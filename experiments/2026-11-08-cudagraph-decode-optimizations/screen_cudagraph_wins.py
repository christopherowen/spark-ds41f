#!/usr/bin/env python3
"""Two-boot GPU screening benchmark for CUDA Graph decode optimizations on GB10 (SM121).

Measures:
1. Target CUDAGraphMode.FULL in-graph attention metadata (23 KV-cache groups) + CEDState
   replay and preallocated 4-slot pinned host ring buffer with floor deduplication.
2. MoE shared-expert multi-stream fork-join and post-gate hook deferral in
   DefaultMoERunner._forward_impl across decode batch sizes M=1..32 on TP3 and TP4 using
   the real TileLang MoE + Gate + Shared Expert kernels.
3. DFlash drafter steady-state padding_rows launch deduplication.
4. Lead-chunk / geometric split decode CUDA graph evaluation (measured dead end on GB10
   CUDA 13.0.2 where monolithic 61-layer cudaGraphLaunch takes ~2.7 us on host CPU).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from vllm.models.deepseek_v4_1.ced import CED_WINDOW, CEDState
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
from vllm.models.deepseek_v4_1.tilelang import linear, moe, router
from vllm.models.deepseek_v4_1.tilelang.activation import SiluAndMulClampFP32
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.workspace import current_workspace_manager, init_workspace_manager


def bench_target_full_metadata_and_ced_replay() -> list[dict]:
    device = torch.device("cuda:0")
    try:
        current_workspace_manager()
    except Exception:
        init_workspace_manager(device)

    max_reqs = 16
    max_tokens = 2048

    class DummySchedulerConfig:
        max_num_batched_tokens = max_tokens
        max_num_seqs = max_reqs

    class DummyVllmConfig:
        scheduler_config = DummySchedulerConfig()
        speculative_config = type(
            "S",
            (),
            {
                "num_speculative_tokens": 5,
                "num_verification_tokens": 5,
                "parallel_drafting": False,
                "use_eagle": lambda self=None: True,
            },
        )()
        cache_config = type("C", (), {"block_size": 64})()
        model_config = type("M", (), {"max_model_len": 4096})()
        parallel_config = type("P", (), {"decode_context_parallel_size": 1})()
        compilation_config = type("Comp", (), {"cudagraph_mode": None})()

    vllm_cfg = DummyVllmConfig()

    builders = []
    for i in range(23):
        if i % 3 == 0:
            spec = object.__new__(CircularBufferSpec)
            object.__setattr__(spec, "block_size", 64)
            object.__setattr__(spec, "tokens_per_state", 1)
        elif i % 3 == 1:
            spec = object.__new__(FullAttentionSpec)
            object.__setattr__(spec, "block_size", 64)
            object.__setattr__(spec, "tokens_per_state", 1)
        else:
            spec = object.__new__(FullAttentionSpec)
            object.__setattr__(spec, "block_size", 64)
            object.__setattr__(spec, "tokens_per_state", 2)
        b = DeepseekV41B12xMetadataBuilder(
            spec, [f"layer_{i * 3}", f"layer_{i * 3 + 1}"], vllm_cfg, device
        )
        builders.append(b)

    ced_state = CEDState(max_reqs, max_tokens, (64, 128, 256, 512, 1024, 2048), device)
    ced_state.warmup(5120, 4, pipeline=False, sink_columns=False)

    query_start_loc_gpu = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc_cpu = torch.zeros(max_reqs + 1, dtype=torch.int32, device="cpu")
    seq_lens_gpu = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    seq_lens_cpu = torch.zeros(max_reqs, dtype=torch.int32, device="cpu")
    block_tables = [
        torch.randint(1, 100, (max_reqs, 64), dtype=torch.int32, device=device)
        for _ in range(23)
    ]
    slot_mappings = torch.randint(
        1, 4000, (26, max_tokens), dtype=torch.int64, device=device
    )
    layer_names_per_group = [
        [f"model.layers.{g * 3 + k}.self_attn.attn" for k in range(4)]
        for g in range(23)
    ]

    def build_slot_mappings_90(slot_mappings_2d):
        by_layer = {}
        for g, names in enumerate(layer_names_per_group):
            row = slot_mappings_2d[g]
            for name in names:
                by_layer[name] = row
        return by_layer

    dummy_w = torch.randn((512, 512), dtype=torch.bfloat16, device=device)
    dummy_x = torch.randn((64, 512), dtype=torch.bfloat16, device=device)
    dummy_out = torch.empty((64, 512), dtype=torch.bfloat16, device=device)
    torch.mm(dummy_x, dummy_w, out=dummy_out)
    g_warm = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_warm):
        ced_state.prepare_cudagraph_replay(query_start_loc_gpu, seq_lens_gpu, 1)
        for g_idx, b in enumerate(builders):
            b.prepare_cudagraph_replay(
                query_start_loc_gpu[:2],
                seq_lens_gpu[:1],
                block_tables[g_idx][:1],
                slot_mappings[g_idx, :1],
                1,
                1,
            )
        torch.mm(dummy_x, dummy_w, out=dummy_out)
    del g_warm
    torch.cuda.synchronize()

    shapes = [
        (1, 1),
        (1, 6),
        (2, 12),
        (3, 18),
        (4, 16),
        (4, 24),
        (8, 32),
        (8, 48),
        (16, 64),
    ]
    results: list[dict] = []

    for num_reqs, num_tokens in shapes:
        q_len = num_tokens // num_reqs
        for r in range(num_reqs + 1):
            query_start_loc_cpu[r] = r * q_len
        query_start_loc_gpu[: num_reqs + 1].copy_(query_start_loc_cpu[: num_reqs + 1])
        for r in range(num_reqs):
            seq_lens_cpu[r] = 1024 + r * 17
        seq_lens_gpu[:num_reqs].copy_(seq_lens_cpu[:num_reqs])
        keep_full = [False] * num_reqs
        floors = [0] * num_reqs
        limits = [CED_WINDOW] * num_reqs

        mem_before_base = torch.cuda.memory_allocated()
        g_base = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_base):
            torch.mm(dummy_x, dummy_w, out=dummy_out)
        torch.cuda.synchronize()
        mem_after_base = torch.cuda.memory_allocated()

        def run_baseline():
            _ = build_slot_mappings_90(slot_mappings[:, :num_tokens])
            _ = torch.tensor(floors, dtype=torch.int64, pin_memory=True)
            _ = torch.tensor(limits, dtype=torch.int32, pin_memory=True)
            ced_state.stage(
                query_start_loc_gpu[: num_reqs + 1],
                seq_lens_gpu[:num_reqs],
                [q_len] * num_reqs,
                keep_full,
                floors,
                limits,
            )
            md = {}
            q_cpu = query_start_loc_cpu[: num_reqs + 1]
            q_gpu = query_start_loc_gpu[: num_reqs + 1]
            s_gpu = seq_lens_gpu[:num_reqs]
            for g_idx, b in enumerate(builders):
                cm = CommonAttentionMetadata(
                    query_start_loc=q_gpu,
                    query_start_loc_cpu=q_cpu,
                    seq_lens=s_gpu,
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=q_len,
                    max_seq_len=4096,
                    block_table_tensor=block_tables[g_idx][:num_reqs],
                    slot_mapping=slot_mappings[g_idx, :num_tokens],
                    causal=True,
                )
                m = b.build(0, cm)
                m.swa_replay_start = ced_state.replay_start
                for ln in layer_names_per_group[g_idx]:
                    md[ln] = m
            g_base.replay()

        for _ in range(10):
            run_baseline()
        torch.cuda.synchronize()
        ref_meta = [
            (
                b.starts.clone(),
                b.request_positions.clone(),
                b.counts.clone(),
                b.positions[:num_tokens].clone(),
                b.reqs[:num_tokens].clone(),
                b.slots[:num_tokens].clone(),
                b.lengths[:num_tokens].clone(),
            )
            for b in builders
        ]
        ref_replay_start = ced_state.replay_start[:num_reqs].clone()

        for b in builders:
            b.starts.zero_()
            b.request_positions.zero_()
            b.counts.zero_()
            b.positions.zero_()
            b.reqs.zero_()
            b.slots.zero_()
            b.lengths.zero_()
        ced_state.replay_start.zero_()

        mem_before_opt = torch.cuda.memory_allocated()
        g_opt = torch.cuda.CUDAGraph()
        ced_state.stage_floors(floors)
        with torch.cuda.graph(g_opt):
            ced_state.prepare_cudagraph_replay(
                query_start_loc_gpu, seq_lens_gpu, num_reqs
            )
            for g_idx, b in enumerate(builders):
                b.prepare_cudagraph_replay(
                    query_start_loc_gpu[: num_reqs + 1],
                    seq_lens_gpu[:num_reqs],
                    block_tables[g_idx][:num_reqs],
                    slot_mappings[g_idx, :num_tokens],
                    num_reqs,
                    num_tokens,
                )
            torch.mm(dummy_x, dummy_w, out=dummy_out)
        torch.cuda.synchronize()
        mem_after_opt = torch.cuda.memory_allocated()

        for b in builders:
            b.starts.zero_()
            b.request_positions.zero_()
            b.counts.zero_()
            b.positions.zero_()
            b.reqs.zero_()
            b.slots.zero_()
            b.lengths.zero_()
        ced_state.replay_start.zero_()

        def run_candidate():
            ced_state.stage_floors(floors)
            g_opt.replay()

        run_candidate()
        torch.cuda.synchronize()

        exact = torch.equal(ced_state.replay_start[:num_reqs], ref_replay_start)
        for g_idx, b in enumerate(builders):
            cur = (
                b.starts,
                b.request_positions,
                b.counts,
                b.positions[:num_tokens],
                b.reqs[:num_tokens],
                b.slots[:num_tokens],
                b.lengths[:num_tokens],
            )
            for c_t, r_t in zip(cur, ref_meta[g_idx]):
                if not torch.equal(c_t, r_t):
                    exact = False

        iters = 300
        for _ in range(30):
            run_baseline()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run_baseline()
        torch.cuda.synchronize()
        base_us = (time.perf_counter() - t0) / iters * 1e6

        for _ in range(30):
            run_candidate()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run_candidate()
        torch.cuda.synchronize()
        opt_us = (time.perf_counter() - t0) / iters * 1e6

        results.append(
            {
                "num_reqs": num_reqs,
                "num_tokens": num_tokens,
                "baseline_us": round(base_us, 2),
                "candidate_us": round(opt_us, 2),
                "delta_us": round(opt_us - base_us, 2),
                "bit_exact": bool(exact),
                "mem_delta_bytes": int(
                    (mem_after_opt - mem_before_opt)
                    - (mem_after_base - mem_before_base)
                ),
            }
        )
    return results


def bench_moe_post_gate_fork() -> list[dict]:
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    hidden_size = 5120
    num_experts = 384
    top_k = 6
    scale = 1.5

    class DummyFP8Linear(nn.Module):
        def __init__(self, out_features, in_features, mxfp8_input=True):
            super().__init__()
            self.weight = nn.Parameter(
                (
                    torch.randn(
                        (out_features, in_features),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.05
                ).to(torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.full(
                    (out_features // 32, in_features // 32),
                    127,
                    dtype=torch.uint8,
                    device=device,
                ).view(torch.float8_e8m0fnu),
                requires_grad=False,
            )
            self.tilelang_mxfp8_input = mxfp8_input
            self.quant_method = linear.TileLangFP8LinearMethod(
                type("Q", (), {"weight_block_size": [32, 32]})()
            )
            self.quant_method.process_weights_after_loading(self)
            self.quant_method.declare_layer_kernels(self)

        def forward(self, x):
            return self.quant_method.apply(self, x)

    class DummyGate(nn.Module):
        def __init__(self, n_experts, h_size):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn((n_experts, h_size), dtype=torch.bfloat16, device=device)
                * 0.02,
                requires_grad=False,
            )
            self.out_dtype = torch.float32
            self.quant_method = linear.TileLangLinearMethod()

        def forward(self, x):
            return self.quant_method.apply(self, x), None

    results: list[dict] = []
    for tp_size, inter_tp in [(3, 768), (4, 576)]:
        gate = DummyGate(num_experts, hidden_size)
        rt = router.TileLangGateRouter(
            top_k=top_k,
            global_num_experts=num_experts,
            e_score_correction_bias=torch.randn(
                num_experts, dtype=torch.float32, device=device
            )
            * 0.01,
            routed_scaling_factor=scale,
            scoring_func="sqrtsoftplus",
            renormalize=True,
            gate=gate,
        )
        rt.max_rows = 64
        gate.quant_method.process_weights_after_loading(gate)
        gate.quant_method.declare_layer_kernels(gate)
        rt.declare_kernels()

        sh_gu = DummyFP8Linear(2 * inter_tp, hidden_size, mxfp8_input=True)
        sh_act = SiluAndMulClampFP32(7.0, inter_tp, mxfp8_output=True)
        sh_act.declare_kernels()
        sh_down = DummyFP8Linear(hidden_size, inter_tp, mxfp8_input=True)

        plan = moe.MoeExpertsV41Plan(
            num_experts,
            top_k,
            hidden_size,
            inter_tp,
            swiglu_limit=10.0,
            out_dtype=torch.bfloat16,
            idx_dtypes=(torch.int64,),
            max_rows=64,
        )
        plan.declare_kernels()

        w13_raw = torch.randint(
            0, 256, (num_experts, 2 * inter_tp, hidden_size // 2), dtype=torch.uint8, device=device
        )
        s13_raw = torch.full(
            (num_experts, 2 * inter_tp, hidden_size // 32), 127, dtype=torch.uint8, device=device
        )
        w2_raw = torch.randint(
            0, 256, (num_experts, hidden_size, inter_tp // 2), dtype=torch.uint8, device=device
        )
        s2_raw = torch.full(
            (num_experts, hidden_size, inter_tp // 32), 127, dtype=torch.uint8, device=device
        )
        weights = plan.prepare_weights(w13_raw, s13_raw, w2_raw, s2_raw)
        del w13_raw, s13_raw, w2_raw, s2_raw

        aux_stream = torch.cuda.Stream(device=device)

        for m in [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]:
            x = torch.randn((m, hidden_size), dtype=torch.bfloat16, device=device) * 0.1
            mxfp8_x = linear.cast_mxfp8(x)
            ws_bytes = moe.workspace_bytes(plan.workspace_specs(m, torch.int64))
            ws = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
            out_buf = torch.empty((m, hidden_size), dtype=torch.bfloat16, device=device)
            with linear.provide_mxfp8(mxfp8_x):
                _ = rt._compute_routing(x, x, torch.int64)
            current_workspace_manager().reserve_by_lane()

            def run_shared():
                gu = sh_gu(x)
                act_q = sh_act.forward_mxfp8(gu)
                with linear.provide_mxfp8(act_q):
                    return sh_down(act_q.rows)

            def mode_fork_pre_gate():
                with linear.provide_mxfp8(mxfp8_x):
                    aux_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(aux_stream):
                        sh_out = run_shared()
                    topk_w, topk_i = rt._compute_routing(x, x, torch.int64)

                    def wait_sh():
                        torch.cuda.current_stream().wait_stream(aux_stream)
                        return sh_out

                    plan.run(
                        mxfp8_x, topk_i, topk_w, weights, out_buf, ws, shared=wait_sh
                    )

            def mode_fork_post_gate():
                with linear.provide_mxfp8(mxfp8_x):
                    sh_box = []

                    def launch_sh():
                        aux_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(aux_stream):
                            sh_box.append(run_shared())

                    rt._post_gate_hook = launch_sh
                    topk_w, topk_i = rt._compute_routing(x, x, torch.int64)

                    def wait_sh():
                        torch.cuda.current_stream().wait_stream(aux_stream)
                        return sh_box[0]

                    plan.run(
                        mxfp8_x, topk_i, topk_w, weights, out_buf, ws, shared=wait_sh
                    )

            def mode_serial_sh_first():
                with linear.provide_mxfp8(mxfp8_x):
                    sh_out = run_shared()
                    topk_w, topk_i = rt._compute_routing(x, x, torch.int64)
                    plan.run(
                        mxfp8_x, topk_i, topk_w, weights, out_buf, ws, shared=sh_out
                    )

            ref_out = None
            bit_exact = True
            timings = {}
            for name, fn in [
                ("serial_us_per_layer", mode_serial_sh_first),
                ("pre_gate_us_per_layer", mode_fork_pre_gate),
                ("post_gate_us_per_layer", mode_fork_post_gate),
            ]:
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                if ref_out is None:
                    ref_out = out_buf.clone()
                elif not torch.equal(out_buf, ref_out):
                    bit_exact = False

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                torch.cuda.synchronize()
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                iters = 200
                start.record()
                for _ in range(iters):
                    g.replay()
                end.record()
                torch.cuda.synchronize()
                timings[name] = round((start.elapsed_time(end) * 1000.0) / iters, 2)

            results.append(
                {
                    "tp_size": tp_size,
                    "num_tokens": m,
                    **timings,
                    "delta_post_vs_pre_us_per_layer": round(
                        timings["post_gate_us_per_layer"]
                        - timings["pre_gate_us_per_layer"],
                        2,
                    ),
                    "delta_post_vs_serial_us_per_layer": round(
                        timings["post_gate_us_per_layer"]
                        - timings["serial_us_per_layer"],
                        2,
                    ),
                    "bit_exact": bit_exact,
                }
            )
    return results


def bench_dflash_padding_rows_dedup() -> list[dict]:
    device = torch.device("cuda:0")
    max_tokens = 96
    tilelang_inputs.padding_rows_kernel()
    is_padding_base = torch.zeros(max_tokens, dtype=torch.bool, device=device)
    is_padding_opt = torch.zeros(max_tokens, dtype=torch.bool, device=device)

    results: list[dict] = []
    for live, padded in [(6, 6), (12, 12), (24, 24), (48, 48), (60, 64), (90, 96)]:
        tilelang_inputs.padding_rows(is_padding_base, live, padded)
        tilelang_inputs.padding_rows(is_padding_opt, live, padded)
        torch.cuda.synchronize()
        exact = bool(torch.equal(is_padding_base, is_padding_opt))

        iters = 500
        for _ in range(50):
            tilelang_inputs.padding_rows(is_padding_base, live, padded)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            tilelang_inputs.padding_rows(is_padding_base, live, padded)
        torch.cuda.synchronize()
        base_us = (time.perf_counter() - t0) / iters * 1e6

        last_live = live
        t0 = time.perf_counter()
        for _ in range(iters):
            if last_live != live:
                tilelang_inputs.padding_rows(
                    is_padding_opt, live, max(padded, last_live or 0)
                )
                last_live = live
        torch.cuda.synchronize()
        opt_us = (time.perf_counter() - t0) / iters * 1e6

        results.append(
            {
                "live_tokens": live,
                "padded_tokens": padded,
                "baseline_us": round(base_us, 2),
                "candidate_us": round(opt_us, 2),
                "delta_us": round(opt_us - base_us, 2),
                "bit_exact": exact,
            }
        )
    return results


def bench_lead_chunk_split_dead_end() -> list[dict]:
    device = torch.device("cuda:0")
    num_layers = 61
    hidden = 2048
    weights = [
        [
            torch.randn(hidden, hidden, dtype=torch.bfloat16, device=device) * 0.01
            for _ in range(4)
        ]
        for _ in range(num_layers)
    ]
    draft_w = [
        torch.randn(hidden, hidden, dtype=torch.bfloat16, device=device) * 0.01
        for _ in range(4)
    ]

    configs = {
        "1_mono_61": [61],
        "2_lead_1_60": [1, 60],
        "2_lead_2_59": [2, 59],
        "3_geom_1_10_50": [1, 10, 50],
        "4_geom_1_4_16_40": [1, 4, 16, 40],
    }

    results: list[dict] = []
    for m in [1, 6, 12, 24, 48]:
        x = torch.randn(m, hidden, dtype=torch.bfloat16, device=device)
        dx = torch.randn(max(1, m // 6), hidden, dtype=torch.bfloat16, device=device)

        draft_graphs = []
        for _ in range(5):
            dg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(dg):
                dh = dx
                for w in draft_w:
                    dh = torch.nn.functional.silu(dh @ w)
                dx.copy_(dh)
            draft_graphs.append(dg)

        ref_out = None
        for name, splits in configs.items():
            bufs = [x.clone()] + [torch.empty_like(x) for _ in splits]
            graphs = []
            idx = 0
            for chunk_idx, count in enumerate(splits):
                layer_slice = weights[idx : idx + count]
                idx += count
                inp = bufs[chunk_idx]
                out = bufs[chunk_idx + 1]
                for _ in range(3):
                    h = inp
                    for lw in layer_slice:
                        for w in lw:
                            h = h @ w
                        h = torch.nn.functional.silu(h)
                    out.copy_(h)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    h = inp
                    for lw in layer_slice:
                        for w in lw:
                            h = h @ w
                        h = torch.nn.functional.silu(h)
                    out.copy_(h)
                graphs.append(g)

            for g in graphs:
                g.replay()
            torch.cuda.synchronize()
            if ref_out is None:
                ref_out = bufs[-1].clone()
                exact = True
            else:
                exact = bool(torch.equal(ref_out, bufs[-1]))

            iters = 150
            for _ in range(25):
                for g in graphs:
                    g.replay()
                for dg in draft_graphs:
                    dg.replay()
            torch.cuda.synchronize()

            cpu_launch_ns = 0
            t0 = time.perf_counter()
            for _ in range(iters):
                c0 = time.perf_counter_ns()
                for g in graphs:
                    g.replay()
                cpu_launch_ns += time.perf_counter_ns() - c0
                for dg in draft_graphs:
                    dg.replay()
                torch.cuda.synchronize()
            step_us = (time.perf_counter() - t0) / iters * 1e6
            cpu_us = (cpu_launch_ns / iters) / 1e3

            results.append(
                {
                    "num_tokens": m,
                    "config": name,
                    "num_chunks": len(splits),
                    "cpu_launch_us": round(cpu_us, 2),
                    "step_wall_us": round(step_us, 2),
                    "bit_exact": exact,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "boot": args.boot,
        "device_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "target_full_metadata_and_ced_replay": bench_target_full_metadata_and_ced_replay(),
        "moe_post_gate_fork": bench_moe_post_gate_fork(),
        "dflash_padding_rows_dedup": bench_dflash_padding_rows_dedup(),
        "lead_chunk_split_dead_end": bench_lead_chunk_split_dead_end(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
