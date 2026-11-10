"""Isolated GPU gate and microbenchmark screen for CopyDrafter survival yield and 16-row tile cap.

Verifies on sm_121a (NVIDIA GB10):
1. copy_apply expected-yield gate computes cumulative DSpark survival yield
   sum_{j=0}^{B-1} prod_{i=0}^j p_i instead of raw sum_{j=0}^{B-1} p_j, admitting
   profitable 2..4-token copy continuations without adding kernel latency.
2. AdaptiveVerificationManager._step_cost caps single-request (c1) initial 0.95-prior
   copy proposals at 15 drafts (16 verification rows = 1 Tensor Core MMA tile in the
   16-row CUDA graph) instead of spilling 1 row into a second 16-row MMA wave and the
   24-row CUDA graph, while admitting 16 drafts once copy accuracy reaches >= 0.97 or
   when multiple requests share the batch (c2+).
3. AdaptiveVerificationManager.dead_row_args activates a zero-confidence floor
   (log_tau = -1e30) on wide copy steps (step_steps > num_draft_steps) even when
   SPARK_DS41F_DSPARK_DEAD_ROWS_TAU == 0, so combine_sampled_and_draft_tokens marks
   trailing -1 placeholder rows (confidence == 0) as padding (is_padding = True,
   dead_cut = offered) and routed MoE (VLLM_MOE_SKIP_PADDING=1) skips them across
   all 58 MoE layers.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

import vllm  # noqa: F401
from vllm.models.deepseek_v4_1.tilelang import gemm, linear, moe, prepare
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.tilelang_copy import (
    copy_alloc,
    copy_apply,
)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--boot", required=True, help="Boot identifier (e.g. boot1, boot2)")
p.add_argument("--repeats", type=int, default=20, help="Timed graph replays per case")
a = p.parse_args()
assert a.repeats >= 5


def emit(**record):
    print(
        json.dumps(
            {
                "boot": a.boot,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **record,
            }
        ),
        flush=True,
    )


def timings(call, wash_buf: torch.Tensor | None):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if wash_buf is not None:
            wash_buf.add_(1.0)
        begin.record()
        call()
        end.record()
    samples = []
    for _ in range(a.repeats):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def make_manager(
    confidences: np.ndarray, row_cost: float = 0.035
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager.num_draft_steps = 5
    manager.step_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences.copy())]
    manager._stale_idx = 0
    manager.req_states = SimpleNamespace(
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.row_cost = row_cost
    manager.dead_rows_tau = 0.0
    return manager


def main() -> int:
    device = torch.device("cuda")
    flush = torch.zeros(128 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    emit(
        kind="identity",
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        seed=20261108,
        repeats=a.repeats,
        cache_modes=["warm", "128MiB_wash"],
        timer="external CUDA event nodes inside CUDAGraph replay",
    )

    # -------------------------------------------------------------------------
    # 1. Gate & timing: copy_apply cumulative survival yield vs raw sum
    # -------------------------------------------------------------------------
    block = 5
    lookahead = 16
    initial = 8
    max_reqs = 8
    max_len = 512

    history = torch.full((max_reqs, max_len), 42, dtype=torch.int32, device=device)
    total_len = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    idx = torch.tensor([0], dtype=torch.int64, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    dspark_ids = torch.tensor(
        [[101, 102, 103, 104, 105]], dtype=torch.int64, device=device
    )
    dspark_confs = torch.tensor(
        [[0.70, 0.65, 0.60, 0.55, 0.50]], dtype=torch.float32, device=device
    )
    raw_sum = float(dspark_confs[0].sum().item())
    cum_survival = float(
        np.cumprod(dspark_confs[0].cpu().numpy(), dtype=np.float32).sum()
    )

    min_match = 16
    levels = 5  # 16, 32, 64, 128, 256
    for offered in (2, 3, 4, 8):
        total_len[0] = 100 + offered
        sources = torch.full((1, 32, levels), -1, dtype=torch.int32, device=device)
        sources[0, 0, :] = offered
        int_state = torch.zeros((max_reqs, 2), dtype=torch.int32, device=device)
        float_state = torch.zeros((max_reqs, 2), dtype=torch.float32, device=device)
        chosen = torch.zeros((1, 3), dtype=torch.int32, device=device)
        estimates = torch.zeros(1, dtype=torch.float32, device=device)
        drafts = torch.full((1, lookahead), -1, dtype=torch.int64, device=device)
        conf = torch.zeros((1, lookahead), dtype=torch.float32, device=device)
        planned = torch.zeros((1, lookahead), dtype=torch.float32, device=device)
        alloc = torch.zeros(1, dtype=torch.int32, device=device)

        copy_alloc(
            idx,
            total_len,
            sources,
            zero,
            zero,
            zero,
            int_state,
            float_state,
            chosen,
            estimates,
            alloc,
            min_match=min_match,
            lookahead=lookahead,
            block=block,
            initial=initial,
        )
        copy_apply(
            idx,
            history,
            chosen,
            estimates,
            int_state,
            dspark_ids,
            dspark_confs,
            drafts,
            conf,
            planned,
        )
        copy_yield = min(offered, initial) * 0.95
        parent_admits = copy_yield >= raw_sum
        cand_admits = int(int_state[0, 1].item()) == min(offered, initial)
        assert cand_admits is (copy_yield >= cum_survival)
        emit(
            kind="gate_copy_apply_yield",
            offered=min(offered, initial),
            copy_yield=round(copy_yield, 4),
            dspark_raw_sum=round(raw_sum, 4),
            dspark_cum_survival=round(cum_survival, 6),
            parent_admits=parent_admits,
            candidate_admits=cand_admits,
            drafts_head=drafts[0, : min(offered, initial)].tolist(),
        )

    # Time copy_apply kernel in a CUDA graph across 8 requests
    idx8 = torch.arange(8, dtype=torch.int64, device=device)
    chosen8 = torch.tensor([[256, 8, 16]] * 8, dtype=torch.int32, device=device)
    est8 = torch.full((8,), 0.95, dtype=torch.float32, device=device)
    dsp_ids8 = torch.full((8, block), 101, dtype=torch.int64, device=device)
    dsp_conf8 = dspark_confs.expand(8, block).contiguous()
    int8 = torch.zeros((8, 2), dtype=torch.int32, device=device)
    dr8 = torch.empty((8, lookahead), dtype=torch.int64, device=device)
    cf8 = torch.empty((8, lookahead), dtype=torch.float32, device=device)
    pl8 = torch.empty((8, lookahead), dtype=torch.float32, device=device)

    def run_copy_apply():
        copy_apply(
            idx8,
            history,
            chosen8,
            est8,
            int8,
            dsp_ids8,
            dsp_conf8,
            dr8,
            cf8,
            pl8,
        )

    for cache_mode, wash_buf in (("warm", None), ("128MiB_wash", flush)):
        samples = timings(run_copy_apply, wash_buf)
        emit(
            kind="timing_copy_apply_kernel",
            num_reqs=8,
            cache_mode=cache_mode,
            mean_us=round(statistics.mean(samples) * 1000.0, 3),
            median_us=round(statistics.median(samples) * 1000.0, 3),
            stdev_us=round(statistics.stdev(samples) * 1000.0, 3),
        )

    # -------------------------------------------------------------------------
    # 2. Gate: AdaptiveVerificationManager c1 16-row tile cap + dead-row masking
    # -------------------------------------------------------------------------
    mgr_c1 = make_manager(np.full((1, 16), 0.95, dtype=np.float32), row_cost=0.035)
    # Parent behaviour (single_req_tile_spill_scale = 0.55) vs Candidate (2.0)
    mgr_c1.single_req_tile_spill_scale = 0.55
    parent_c1_tokens = mgr_c1.get_num_tokens({"r0": 17}, {"r0": list(range(16))})
    parent_c1_budget = mgr_c1._batch_budget[2]

    mgr_c1.single_req_tile_spill_scale = 2.0
    cand_c1_tokens = mgr_c1.get_num_tokens({"r0": 17}, {"r0": list(range(16))})
    cand_c1_budget = mgr_c1._batch_budget[2]

    assert parent_c1_tokens == 17 and parent_c1_budget == 16
    assert cand_c1_tokens == 16 and cand_c1_budget == 15

    # Verify GPU combine_sampled_and_draft_tokens with dead_row_args on shrunk wide step
    # (step_steps=16, 5 live drafts followed by -1 placeholder drafts with conf=0.0)
    conf_gpu = torch.zeros((1, 16), dtype=torch.float32, device=device)
    conf_gpu[0, :5] = torch.tensor([0.85, 0.80, 0.75, 0.70, 0.65], device=device)
    mgr_c1._confidence_probs = conf_gpu
    mgr_c1._batch_draft_capacity = torch.tensor([15], dtype=torch.int32, device=device)
    mgr_c1._dead_row_cut = torch.zeros(1, dtype=torch.int32, device=device)
    mgr_c1._has_admitted_drafts = True
    mgr_c1.step_steps = 16

    input_ids = torch.zeros(16, dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    last_sampled = torch.tensor([999], dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 16], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([116], dtype=torch.int32, device=device)
    prefill_len = torch.tensor([100], dtype=torch.int32, device=device)
    draft_tokens = torch.full((1, 16), -1, dtype=torch.int64, device=device)
    draft_tokens[0, :5] = torch.tensor([11, 12, 13, 14, 15], device=device)
    cu_num_logits = torch.tensor([0, 16], dtype=torch.int32, device=device)
    logits_indices = torch.zeros(16, dtype=torch.int64, device=device)

    is_padding_parent = torch.zeros(24, dtype=torch.bool, device=device)
    tilelang_inputs.padding_rows(is_padding_parent, 17, 24)
    tilelang_inputs.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled,
        torch.tensor([0, 17], dtype=torch.int32, device=device),
        seq_lens,
        prefill_len,
        draft_tokens,
        torch.tensor([0, 17], dtype=torch.int32, device=device),
        torch.zeros(17, dtype=torch.int64, device=device),
        1,
        dead_rows=None,
    )

    is_padding_cand = torch.zeros(16, dtype=torch.bool, device=device)
    tilelang_inputs.padding_rows(is_padding_cand, 16, 16)
    dead_args = mgr_c1.dead_row_args(is_padding_cand, 1)
    assert dead_args is not None
    tilelang_inputs.combine_sampled_and_draft_tokens(
        input_ids,
        idx_mapping,
        last_sampled,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        logits_indices,
        1,
        dead_rows=dead_args,
    )
    verified_counts = mgr_c1.get_verified_draft_counts(1, dead_args.cut)
    assert int(dead_args.cut[0].item()) == 5
    assert int(verified_counts[0].item()) == 5
    assert is_padding_cand[:6].tolist() == [False] * 6
    assert is_padding_cand[6:16].tolist() == [True] * 10
    assert is_padding_parent[:17].tolist() == [False] * 17
    assert is_padding_parent[17:24].tolist() == [True] * 7

    emit(
        kind="gate_adaptive_tile_and_dead_rows",
        parent_c1_tokens=parent_c1_tokens,
        parent_c1_graph_bucket=24,
        candidate_c1_tokens=cand_c1_tokens,
        candidate_c1_graph_bucket=16,
        shrunk_step_live_rows=6,
        parent_shrunk_unmasked_rows=17,
        candidate_shrunk_unmasked_rows=int((~is_padding_cand).sum().item()),
        candidate_dead_cut=int(dead_args.cut[0].item()),
    )

    # -------------------------------------------------------------------------
    # 3. Timing: 58-layer routed MoE + 3-layer dense MLP decode step on GB10
    # -------------------------------------------------------------------------
    gen = torch.Generator(device=device).manual_seed(20261108)
    hidden = 5120
    num_experts = 384
    topk = 6
    num_moe_layers = 58
    num_dense_layers = 3

    for world, inter in ((3, 768), (4, 576)):
        dense_inter = 18432 // world
        plan = moe.MoeExpertsV41Plan(
            num_experts,
            topk,
            hidden,
            inter,
            swiglu_limit=10.0,
            out_dtype=torch.float32,
            idx_dtypes=(torch.int64,),
            max_rows=96,
        )
        for name, kernel in linear._launch(plan.kernels()).items():
            setattr(plan, name, kernel)
        for dispatch in plan._dispatches():
            dispatch.bind()

        w13 = torch.randint(
            0,
            255,
            (num_experts, 2 * inter, hidden // 2),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        )
        w13_scale = torch.randint(
            124,
            128,
            (num_experts, 2 * inter, hidden // 32),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        )
        w2 = torch.randint(
            0,
            255,
            (num_experts, hidden, inter // 2),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        )
        w2_scale = torch.randint(
            124,
            128,
            (num_experts, hidden, inter // 32),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        )
        moe_weights = plan.prepare_weights(w13, w13_scale, w2, w2_scale)

        # Dense / shared projection GEMMs at 16-row (1 MMA tile) vs 24-row (32-row, 2 MMA tiles)
        dense_gemm_16 = gemm.mxfp8_gemm(
            dense_inter,
            hidden,
            block_M=16,
            block_N=64,
            block_K=128,
            num_stages=2,
            threads=128,
            shards=1,
            groups=1,
            padded_rows=True,
            tile_contiguous=False,
        )
        dense_gemm_32 = gemm.mxfp8_gemm(
            dense_inter,
            hidden,
            block_M=32,
            block_N=64,
            block_K=128,
            num_stages=2,
            threads=128,
            shards=1,
            groups=1,
            padded_rows=True,
            tile_contiguous=False,
        )
        dw = torch.randint(
            0,
            120,
            (dense_inter, hidden),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        dsf = torch.randint(
            125,
            128,
            (dense_inter // 32, 4 * gemm.scale_words(hidden)),
            generator=gen,
            device=device,
            dtype=torch.uint8,
        ).view(torch.uint32)

        def build_step_runner(graph_rows: int, live_routed_rows: int):
            x_val = torch.randint(
                0,
                120,
                (graph_rows, hidden),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            ).view(torch.float8_e4m3fn)
            x_sf = torch.randint(
                125,
                128,
                (graph_rows, 4 * gemm.scale_words(hidden)),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            )
            x_rows = SimpleNamespace(values=x_val, scales=x_sf)
            topk_ids = torch.full(
                (graph_rows, topk), -1, dtype=torch.int64, device=device
            )
            for r in range(live_routed_rows):
                topk_ids[r] = torch.tensor(
                    [(r * 7 + k * 13) % num_experts for k in range(topk)],
                    dtype=torch.int64,
                    device=device,
                )
            topk_weights = torch.zeros(
                (graph_rows, topk), dtype=torch.float32, device=device
            )
            topk_weights[:live_routed_rows] = 1.0 / topk
            shared_out = torch.zeros(
                (graph_rows, hidden), dtype=torch.bfloat16, device=device
            )
            moe_out = torch.empty(
                (graph_rows, hidden), dtype=torch.bfloat16, device=device
            )
            ws_bytes = moe.workspace_bytes(
                plan.workspace_specs(graph_rows, torch.int64)
            )
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)

            dense_rows = 16 if graph_rows <= 16 else 32
            dx = torch.randint(
                0,
                120,
                (dense_rows, hidden),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            ).view(torch.float8_e4m3fn)
            dxsf = torch.randint(
                125,
                128,
                (dense_rows, 4 * gemm.scale_words(hidden)),
                generator=gen,
                device=device,
                dtype=torch.uint8,
            ).view(torch.uint32)
            dout = torch.empty(
                (dense_rows, dense_inter), dtype=torch.bfloat16, device=device
            )
            dgemm = dense_gemm_16 if graph_rows <= 16 else dense_gemm_32

            def run_one_layer_pair():
                plan.run(
                    x_rows,
                    topk_ids,
                    topk_weights,
                    moe_weights,
                    moe_out,
                    workspace=ws,
                    shared=shared_out,
                )
                dgemm(dx, dw, dxsf, dsf, dout)

            return run_one_layer_pair

        scenarios = [
            (
                "c1_initial_copy_step",
                ("parent_17rows_24graph", 24, 17),
                ("candidate_16rows_16graph", 16, 16),
            ),
            (
                "c1_post_reject_shrunk_step",
                ("parent_17unmasked_24graph", 24, 17),
                ("candidate_6live_10masked_16graph", 16, 6),
            ),
        ]

        for scenario_name, parent_spec, cand_spec in scenarios:
            p_name, p_grows, p_lrows = parent_spec
            c_name, c_grows, c_lrows = cand_spec
            fn_p = build_step_runner(p_grows, p_lrows)
            fn_c = build_step_runner(c_grows, c_lrows)
            variants = [(p_name, p_grows, p_lrows, fn_p), (c_name, c_grows, c_lrows, fn_c)]
            if a.boot.endswith("2"):
                variants = list(reversed(variants))

            for cache_mode, wash_buf in (("warm", None), ("128MiB_wash", flush)):
                for vname, grows, lrows, fn in variants:
                    samples = timings(fn, wash_buf)
                    # Each call runs 1 routed MoE layer + 1 dense projection; also report
                    # the 58-MoE-layer + 3-dense-layer equivalent per-step time.
                    mean_layer_ms = statistics.mean(samples)
                    emit(
                        kind="timing_step",
                        world=world,
                        inter=inter,
                        scenario=scenario_name,
                        cache_mode=cache_mode,
                        variant=vname,
                        graph_rows=grows,
                        live_routed_rows=lrows,
                        mean_layer_ms=round(mean_layer_ms, 5),
                        median_layer_ms=round(statistics.median(samples), 5),
                        stdev_layer_ms=round(statistics.stdev(samples), 5),
                        est_58layer_step_ms=round(mean_layer_ms * num_moe_layers, 3),
                        samples_ms=[round(x, 5) for x in samples],
                    )

        del w13, w13_scale, w2, w2_scale, moe_weights, dw, dsf
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
