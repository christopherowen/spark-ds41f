#!/usr/bin/env python3
"""Two-boot GPU screening benchmark for TP=3 and TP=4 decode recovery (patch 0762).

Measures on SM121 (GB10):
1. KV compressor projection (4 owner layers: 2x ratio-1 512x5120 + 2x ratio-2 1024x5120):
   1107 12-bit pack12 vs 0762 uncompressed replicated BF16 (cold L2 and L2-prefetched).
2. Sparse MLA decode (TP=3 heads=24, 30 indexed layers):
   1107 Stage-2 seg_active branch + split_max_rows=4 vs 0762 branchless online-softmax
   merge + split_max_rows=6.
3. Post-step verification bookkeeping:
   1107 eager PyTorch masks/clamps + duplicate get_verified_draft_counts vs 0762
   fused _accumulate_kernel + deduplicated get_verified_draft_counts.
4. DSpark confidence head + online logistic calibration:
   1107 two-kernel (dspark_confidence + _calibrate_confidence_kernel) vs 0762
   single-kernel dspark_confidence_calibrate.
5. Adaptive verification TP=3 vs TP=4 row-cost knee and calibrated dead-row floor.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

if os.path.isdir("/workspace/vllm/vllm"):
    sys.path.insert(0, "/workspace/vllm")

import numpy as np
import torch


def _bench_us(fn, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / iters


def bench_compressor(device: torch.device) -> dict:
    from vllm.models.deepseek_v4_1.tilelang.gather import gather_columns_into
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangSplitLinearMethod,
        project_parts,
    )

    hidden = 5120
    gen = torch.Generator(device=device).manual_seed(42)
    x_buf = torch.randn((64, hidden), generator=gen, device=device, dtype=torch.bfloat16)

    def make_layer(parts: int, part_cols: int, pack_bf16: bool, tp_size: int):
        n = part_cols * parts
        w = (torch.randn((n, hidden), generator=gen, device=device) * 0.02).bfloat16()
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.pack_bf16 = pack_bf16
        layer.tp_size = tp_size
        layer.out_dtype = torch.float32 if parts == 2 else torch.bfloat16
        method = TileLangSplitLinearMethod(parts)
        layer.quant_method = method
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)
        return layer

    # 1107 TP=3: disable_tp=False -> part_cols=192, pack_bf16=True + gather_columns_into
    # (plus 4 RoCE all_gathers across the 4 compressor layers in multi-rank serving).
    r1_1107 = make_layer(1, 192, pack_bf16=True, tp_size=3)
    r2_1107 = make_layer(2, 192, pack_bf16=True, tp_size=3)
    # 0762 TP<=4: disable_tp=True -> part_cols=512, pack_bf16=False, no gather, no all_gather.
    r1_0762 = make_layer(1, 512, pack_bf16=False, tp_size=1)
    r2_0762 = make_layer(2, 512, pack_bf16=False, tp_size=1)

    out1_bf16 = torch.empty((64, 512), dtype=torch.bfloat16, device=device)
    out2_a = torch.empty((64, 512), dtype=torch.float32, device=device)
    out2_b = torch.empty((64, 512), dtype=torch.float32, device=device)
    local1_shards = torch.empty((64, 192), dtype=torch.bfloat16, device=device)
    gathered1 = torch.empty((3 * 64, 192), dtype=torch.bfloat16, device=device)
    local2_shards = torch.empty((2 * 64, 192), dtype=torch.float32, device=device)
    gathered2 = torch.empty((3 * 2 * 64, 192), dtype=torch.float32, device=device)

    results = {}
    for rows in (1, 4, 6, 8, 16, 24, 32, 48):
        inp = x_buf[:rows]
        loc1 = local1_shards[:rows].view(1, rows, 192)
        gath1 = gathered1[: 3 * rows]
        loc2 = local2_shards[: 2 * rows].view(2, rows, 192)
        gath2 = gathered2[: 6 * rows]

        def run_r1_1107():
            project_parts(r1_1107, inp, tuple(loc1), rows)
            gather_columns_into(gath1, (out1_bf16[:rows],), 3)

        def run_r2_1107():
            project_parts(r2_1107, inp, tuple(loc2), rows)
            gather_columns_into(gath2, (out2_a[:rows], out2_b[:rows]), 3)

        t_r1_1107 = _bench_us(run_r1_1107)
        t_r1_0762 = _bench_us(lambda: project_parts(r1_0762, inp, (out1_bf16,), rows))
        t_r2_1107 = _bench_us(run_r2_1107)
        t_r2_0762 = _bench_us(
            lambda: project_parts(r2_0762, inp, (out2_a, out2_b), rows)
        )
        step_4layers_1107 = 2.0 * t_r1_1107 + 2.0 * t_r2_1107
        step_4layers_0762 = 2.0 * t_r1_0762 + 2.0 * t_r2_0762
        results[str(rows)] = {
            "ratio1_1107_local_us": round(t_r1_1107, 2),
            "ratio1_0762_us": round(t_r1_0762, 2),
            "ratio2_1107_local_us": round(t_r2_1107, 2),
            "ratio2_0762_us": round(t_r2_0762, 2),
            "step_4layers_1107_local_us": round(step_4layers_1107, 2),
            "step_4layers_0762_us": round(step_4layers_0762, 2),
            "step_4layers_local_delta_us": round(
                step_4layers_0762 - step_4layers_1107, 2
            ),
            "eliminated_roce_all_gathers_per_step": 4,
        }
    return results


def bench_sparse_mla_tp3(device: torch.device) -> dict:
    import importlib.util
    import os
    from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
        HEAD_DIM,
        IDX_RECORD_BYTES,
        SWA_RECORD_BYTES,
        SparseMlaV41Plan,
    )

    heads = 24
    swa_width, idx_width, page_size = 128, 512, 128
    max_rows = 48
    plan_0762 = SparseMlaV41Plan(
        heads,
        swa_width,
        idx_width,
        page_size,
        swa_page_size=page_size,
        max_rows=max_rows,
        device=device,
    )

    plan_1107 = plan_0762
    ref_path = "/host_tmp/sparse_mla_1107.py"
    if os.path.exists(ref_path):
        spec = importlib.util.spec_from_file_location(
            "vllm.models.deepseek_v4_1.tilelang._sparse_mla_1107", ref_path
        )
        mod_1107 = importlib.util.module_from_spec(spec)
        mod_1107.__package__ = "vllm.models.deepseek_v4_1.tilelang"
        assert spec.loader is not None
        spec.loader.exec_module(mod_1107)
        plan_1107 = mod_1107.SparseMlaV41Plan(
            heads,
            swa_width,
            idx_width,
            page_size,
            swa_page_size=page_size,
            max_rows=max_rows,
            device=device,
        )

    num_pages = 64
    q = torch.randn((max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    swa_kv = torch.randint(
        0, 255, (num_pages, page_size * SWA_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    idx_kv = torch.randint(
        0, 255, (num_pages, page_size * IDX_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    swa_indices = torch.ones((max_rows, swa_width), dtype=torch.int32, device=device)
    swa_lens = torch.full((max_rows,), 128, dtype=torch.int32, device=device)
    idx_indices = torch.ones((max_rows, idx_width), dtype=torch.int32, device=device)
    idx_lens = torch.tensor(
        [128, 256, 384, 512] * (max_rows // 4), dtype=torch.int32, device=device
    )
    idx_page_table = torch.ones((max_rows, 32), dtype=torch.int32, device=device)
    sinks = torch.zeros((heads,), dtype=torch.float32, device=device)
    positions = torch.full((max_rows,), 1024, dtype=torch.int64, device=device)
    cos_sin = torch.ones((4096, 64), dtype=torch.float32, device=device)
    out = torch.empty((max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    partial = torch.empty(
        (max_rows, 5, heads, HEAD_DIM), dtype=torch.float32, device=device
    )
    partial_stats = torch.empty(
        (max_rows, 5, heads, 2), dtype=torch.float32, device=device
    )

    def run_plan(plan, r: int):
        plan.run(
            q[:r],
            swa_kv,
            swa_indices[:r],
            swa_lens[:r],
            sinks,
            positions[:r],
            cos_sin,
            out[:r],
            idx_cache=idx_kv,
            idx_indices=idx_indices[:r],
            idx_lengths=idx_lens[:r],
            idx_page_table=idx_page_table[:r],
            scratch=(partial[:r], partial_stats[:r]),
        )

    row_timings = {}
    for r in (4, 6, 8, 12, 16, 24, 32, 48):
        t_1107 = _bench_us(lambda r=r: run_plan(plan_1107, r))
        t_0762 = _bench_us(lambda r=r: run_plan(plan_0762, r))
        row_timings[str(r)] = {
            "route_1107": plan_1107.kernel_for(r),
            "route_0762": plan_0762.kernel_for(r),
            "us_1107": round(t_1107, 2),
            "us_0762": round(t_0762, 2),
            "delta_us": round(t_0762 - t_1107, 2),
            "step_30layers_1107_ms": round(30.0 * t_1107 / 1000.0, 3),
            "step_30layers_0762_ms": round(30.0 * t_0762 / 1000.0, 3),
            "step_30layers_delta_ms": round(30.0 * (t_0762 - t_1107) / 1000.0, 3),
        }

    return {
        "tp3_split_max_rows_1107": plan_1107.split_max_rows,
        "tp3_split_max_rows_0762": plan_0762.split_max_rows,
        "tp3_decode_max_rows": plan_0762.decode_max_rows,
        "rows": row_timings,
    }


def bench_post_step_bookkeeping(device: torch.device) -> dict:
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        OnlineAcceptanceEstimator,
    )

    est = OnlineAcceptanceEstimator.for_confidence_head(16, 5, device)
    results = {}
    for num_reqs in (1, 2, 4, 8):
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        num_sampled = torch.full((num_reqs,), 3, dtype=torch.int32, device=device)
        num_rejected = torch.full((num_reqs,), 2, dtype=torch.int32, device=device)
        dead_row_cut = torch.full((16,), 4, dtype=torch.int32, device=device)
        sink_events = torch.zeros((num_reqs, 2), dtype=torch.int32, device=device)
        copy_int_state = torch.zeros((16, 4), dtype=torch.int32, device=device)

        def run_old_1107():
            live_rejected = (
                dead_row_cut[:num_reqs] - (num_sampled - 1).clamp_min(0)
            ).clamp_min(0)
            obs_rej = torch.minimum(num_rejected, live_rejected)
            obs_rej = obs_rej.masked_fill(sink_events[:, 0] == 2, 0)
            obs_idx = torch.where(copy_int_state[idx_mapping, 1] > 0, -1, idx_mapping)
            est._steps_since_refit = 0
            est.step(obs_idx, num_sampled, obs_rej)

        def run_new_0762():
            est._steps_since_refit = 0
            est.step(
                idx_mapping,
                num_sampled,
                num_rejected,
                dead_row_cut=dead_row_cut,
                sink_events=sink_events,
                copy_int_state=copy_int_state,
            )

        t_old = _bench_us(run_old_1107)
        t_new = _bench_us(run_new_0762)
        results[str(num_reqs)] = {
            "unfused_1107_us": round(t_old, 2),
            "fused_0762_us": round(t_new, 2),
            "delta_us": round(t_new - t_old, 2),
        }
    return results


def bench_confidence_calibration(device: torch.device) -> dict:
    from vllm.models.deepseek_v4_1.tilelang import confidence
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        OnlineAcceptanceEstimator,
    )

    hidden_size, rank, steps = 5120, 256, 5
    gen = torch.Generator(device=device).manual_seed(99)
    h_buf = torch.randn(
        (64, hidden_size), generator=gen, device=device, dtype=torch.bfloat16
    )
    e_buf = torch.randn((64, rank), generator=gen, device=device, dtype=torch.bfloat16)
    weight = (
        torch.randn((hidden_size + rank,), generator=gen, device=device) * 0.01
    ).float()
    head = confidence.TileLangConfidenceHead.__new__(confidence.TileLangConfidenceHead)
    torch.nn.Module.__init__(head)
    head.proj = SimpleNamespace(weight=weight.view(1, -1))
    head.hidden_size, head.markov_rank = hidden_size, rank
    est = OnlineAcceptanceEstimator.for_confidence_head(16, steps, device)

    results = {}
    for num_reqs in (1, 2, 4, 8):
        rows = num_reqs * steps
        h = h_buf[:rows]
        e = e_buf[:rows]
        out = torch.empty((num_reqs, steps), dtype=torch.float32, device=device)
        idx_map = (
            torch.arange(num_reqs, dtype=torch.int32, device=device)
            .repeat_interleave(steps)
            .contiguous()
        )

        def run_two_step():
            head.probability(h, e, out=out.view(-1))
            est.calibrate_confidence_probs(out, idx_map)

        def run_fused():
            head.probability_and_calibrate(
                h,
                e,
                out.view(-1),
                est.slope,
                est.intercepts,
                idx_map,
                est.features,
                est.predictions,
            )

        t_two = _bench_us(run_two_step)
        t_fused = _bench_us(run_fused)
        results[str(num_reqs)] = {
            "two_kernel_1107_us": round(t_two, 2),
            "fused_0762_us": round(t_fused, 2),
            "delta_us": round(t_fused - t_two, 2),
        }
    return results


def check_adaptive_verification() -> dict:
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    def eval_budget(conf: np.ndarray, scale: float, floor: float) -> tuple[int, int]:
        num_reqs, num_steps = conf.shape
        mgr = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
        mgr.num_speculative_steps = num_steps
        mgr._stale_confidences = [SimpleNamespace(np=conf)]
        mgr._stale_idx = 0
        req_ids = [f"r{i}" for i in range(num_reqs)]
        mgr.req_states = SimpleNamespace(
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            num_computed_tokens_np=np.ones(num_reqs, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
        )
        mgr._max_total_logits = 1 << 30
        mgr.num_bonus_tokens = 1
        mgr.row_cost = 0.035
        mgr.dead_rows_tau = 0.20
        mgr.calibrated_confidence = True
        mgr.high_row_cost_knee = 16
        mgr.high_row_cost_scale = scale
        mgr.min_calibrated_dead_rows_tau = floor
        total = mgr.get_num_tokens(
            {rid: 6 for rid in req_ids},
            {rid: [1, 2, 3, 4, 5] for rid in req_ids},
        )
        return mgr._batch_budget[2], total

    c1_typical = np.array([[0.677, 0.556, 0.396, 0.460, 0.638]], dtype=np.float32)
    c8_med = np.tile(
        np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32), (8, 1)
    )
    tp4_c1 = eval_budget(c1_typical, scale=0.55, floor=0.0)
    tp3_c1 = eval_budget(c1_typical, scale=0.85, floor=0.12)
    tp4_c8 = eval_budget(c8_med, scale=0.55, floor=0.0)
    tp3_c8 = eval_budget(c8_med, scale=0.85, floor=0.12)
    return {
        "tp4_c1_drafts_total": list(tp4_c1),
        "tp3_c1_drafts_total": list(tp3_c1),
        "tp4_c8_drafts_total": list(tp4_c8),
        "tp3_c8_drafts_total": list(tp3_c8),
    }


def main() -> None:
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda", 0)
    init_workspace_manager(device)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "compressor": bench_compressor(device),
        "sparse_mla_tp3": bench_sparse_mla_tp3(device),
        "post_step_bookkeeping": bench_post_step_bookkeeping(device),
        "confidence_calibration": bench_confidence_calibration(device),
        "adaptive_verification": check_adaptive_verification(),
    }
    out_text = json.dumps(payload, indent=2)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    print(out_text)


if __name__ == "__main__":
    main()
