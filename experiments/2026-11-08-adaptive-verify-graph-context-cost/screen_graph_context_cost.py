#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""2-boot GB10 hardware benchmark & screening harness for context-band and
MMA-tile / 64-row wave-saturation step-cost pricing in AdaptiveVerificationManager."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_ROWS,
    TILE_CONTIGUOUS_FP8,
    WIDE_DECODE,
    decode_tile_rows,
    pack_scale_words,
    pack_weight_tiles,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    IndexerV41Plan,
)
from vllm.models.deepseek_v4_1.tilelang.linear import _block32_kernels, _launch
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import SparseMlaV41Plan
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)


def bench_gpu_us(fn, warmup: int = 6, iters: int = 25) -> float:
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


def make_manager(
    confidences: np.ndarray,
    seq_lens: np.ndarray,
    *,
    row_cost: float = 0.035,
    candidate_mode: bool = True,
    calibrated: bool = True,
    dead_rows_tau: float = 0.20,
) -> AdaptiveVerificationManager:
    num_reqs, num_steps = confidences.shape
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.num_speculative_steps = num_steps
    manager.num_draft_steps = num_steps
    manager._stale_confidences = [SimpleNamespace(np=confidences.copy())]
    manager._stale_idx = 0
    req_ids = [f"r{i}" for i in range(num_reqs)]
    manager.req_states = SimpleNamespace(
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        num_computed_tokens_np=seq_lens.astype(np.int32).copy(),
        prefill_len=SimpleNamespace(np=np.ones(num_reqs, dtype=np.int32)),
    )
    manager._max_total_logits = 1 << 30
    manager.num_bonus_tokens = 1
    manager.row_cost = row_cost
    manager.dead_rows_tau = dead_rows_tau
    manager.calibrated_confidence = calibrated
    manager.high_row_cost_knee = 16
    manager.high_row_cost_scale = 0.55
    if candidate_mode:
        manager.coalesced_row_cost_knee = 32
        manager.coalesced_row_cost_scale = 0.28
        manager.coalesced_min_reqs = 10
        manager.saturated_row_cost_knee = 64
        manager.saturated_row_cost_scale = 0.28
        manager.mma_tile_step_rows = 0.0
        manager.single_req_tile_step_rows = 1.5
        manager.second_wave_step_rows = 6.714
        manager.context_row_cost_knee = 262144
        manager.context_reindex_saturation = 524288
        manager.context_reindex_weight = 0.12
        manager.context_source_span = 524288
        manager.context_source_slope = 0.22
        manager.context_max_row_cost_scale = 2.25
    else:
        # Promoted 1107 baseline: unbounded 0.55x knee above 16 rows, no tile/wave
        # steps, and context-blind row cost.
        manager.coalesced_row_cost_knee = 0
        manager.coalesced_row_cost_scale = 0.55
        manager.coalesced_min_reqs = 10
        manager.saturated_row_cost_knee = 0
        manager.saturated_row_cost_scale = 0.55
        manager.mma_tile_step_rows = 0.0
        manager.single_req_tile_step_rows = 0.0
        manager.second_wave_step_rows = 0.0
        manager.context_row_cost_knee = 0
    return manager


def profile_hardware_grid(device: torch.device) -> dict:
    """Measure live GB10 TileLang Dense FP8 + SparseMLA + Indexer kernel latencies."""
    mla_idx_tp4 = SparseMlaV41Plan(
        16, 128, 512, 256, swa_page_size=256, max_rows=112, device=device
    )
    mla_swa_tp4 = SparseMlaV41Plan(
        16, 128, 0, 256, swa_page_size=256, max_rows=112, device=device
    )
    idx_ratio2 = IndexerV41Plan(65536, 128, candidate_blocks=False)
    idx_source1 = IndexerV41Plan(131072, 256, candidate_blocks=True)
    idx_reindex = IndexerV41Plan(16384, 256, candidates=True)

    dense_specs = [
        ("fused_wqa_wkv", 1792, 5120, 1),
        ("wq_b", 8192, 1280, 1),
        ("wo_b", 5120, 2048, 1),
        ("shared_gate_up", 1152, 5120, 1),
        ("shared_down", 5120, 576, 1),
    ]
    dense_projs = {}
    for name, n, k, g in dense_specs:
        ks = _launch(_block32_kernels(n, k, g))
        w = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device)
        if (n, k) in TILE_CONTIGUOUS_FP8 and g == 1:
            w = pack_weight_tiles(w)
        wsf = pack_scale_words(
            torch.zeros((n // 32, k // 32), dtype=torch.uint8, device=device)
        )[: n // 32]
        wide = WIDE_DECODE.get((n, k), {})
        max_dec = max([DECODE_ROWS, *wide.keys()])
        dense_projs[name] = (ks, w, wsf, n, k, wide, max_dec)

    rows_grid = [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 20, 24, 32, 40, 48, 64, 80, 96]
    ctx_grid = [2048, 8192, 16384, 32768, 65536, 131072]

    dense_us = {}
    for R in rows_grid:
        layer_us = 0.0
        for _, (ks, w, wsf, n, k, wide, max_dec) in dense_projs.items():
            padded_r = max(128, R)
            xq = torch.zeros((padded_r, k), dtype=torch.float8_e4m3fn, device=device)
            sf = torch.zeros(
                (padded_r, scale_words(k)), dtype=torch.uint32, device=device
            )
            out = torch.empty((R, n), dtype=torch.bfloat16, device=device)
            if R > max_dec:
                fn = lambda: ks["prefill"](xq[:R], w, sf[:R], wsf, out)
            else:
                bm = decode_tile_rows(R, wide)
                fn = lambda: ks["decode"][bm](xq, w, sf, wsf, out)
            layer_us += bench_gpu_us(fn)
        dense_us[R] = 40.0 * layer_us

    num_pages = 520
    swa_cache = torch.zeros((num_pages, 256 * 528), dtype=torch.uint8, device=device)
    idx_cache = torch.zeros((num_pages, 256 * 288), dtype=torch.uint8, device=device)
    idx_k_pool_256 = torch.zeros(
        (num_pages, 256 * 68), dtype=torch.uint8, device=device
    )
    idx_k_pool_128 = torch.zeros(
        (num_pages, 128 * 68), dtype=torch.uint8, device=device
    )
    sinks = torch.full((16,), -float("inf"), dtype=torch.float32, device=device)
    cos_sin = torch.ones((131072 + 1024, 64), dtype=torch.float32, device=device)

    max_r = max(rows_grid)
    q_all = torch.randn((max_r, 16, 512), dtype=torch.bfloat16, device=device)
    out_all = torch.empty_like(q_all)
    swa_indices_all = torch.zeros((max_r, 128), dtype=torch.int32, device=device)
    swa_lens_all = torch.full((max_r,), 128, dtype=torch.int32, device=device)
    idx_indices_all = torch.zeros((max_r, 512), dtype=torch.int32, device=device)
    idx_lens_all = torch.full((max_r,), 512, dtype=torch.int32, device=device)
    idx_pt_all = torch.zeros((max_r, 520), dtype=torch.int32, device=device)
    pos_all = torch.full((max_r,), 2048, dtype=torch.int64, device=device)

    mla_us = {}
    for R in rows_grid:
        scratch_idx = [
            torch.empty(sh, dtype=dt, device=device)
            for sh, dt in mla_idx_tp4.scratch_specs(R)
        ]
        scratch_swa = [
            torch.empty(sh, dtype=dt, device=device)
            for sh, dt in mla_swa_tp4.scratch_specs(R)
        ]
        us_idx = bench_gpu_us(
            lambda: mla_idx_tp4.run(
                q_all[:R],
                swa_cache,
                swa_indices_all[:R],
                swa_lens_all[:R],
                sinks,
                pos_all[:R],
                cos_sin,
                out_all[:R],
                scratch=scratch_idx or None,
                idx_cache=idx_cache,
                idx_indices=idx_indices_all[:R],
                idx_lengths=idx_lens_all[:R],
                idx_page_table=idx_pt_all[:R],
            )
        )
        us_swa = bench_gpu_us(
            lambda: mla_swa_tp4.run(
                q_all[:R],
                swa_cache,
                swa_indices_all[:R],
                swa_lens_all[:R],
                sinks,
                pos_all[:R],
                cos_sin,
                out_all[:R],
                scratch=scratch_swa or None,
            )
        )
        mla_us[R] = 38.0 * us_idx + 2.0 * us_swa

    q_packed_all = torch.zeros((max_r, 32, 64), dtype=torch.uint8, device=device)
    q_scales_all = torch.ones((max_r, 32, 4), dtype=torch.uint8, device=device)
    weights_all = torch.ones((max_r, 32), dtype=torch.bfloat16, device=device)
    pt_all = torch.zeros((max_r, 520), dtype=torch.int32, device=device)
    cand_blocks_all = torch.zeros(
        (max_r, CANDIDATE_BLOCKS), dtype=torch.int32, device=device
    )
    out_topk_all = torch.empty((max_r, 512), dtype=torch.int32, device=device)

    idx_us = {L: {} for L in ctx_grid}
    for L in ctx_grid:
        active_r1 = torch.tensor([L], dtype=torch.int32, device=device)
        active_r2 = torch.tensor([L // 2], dtype=torch.int32, device=device)
        cand_len_val = min(L, CANDIDATE_BLOCKS * CANDIDATE_BLOCK)
        for R in rows_grid:

            def run_chunked(plan, pool, ext_val, is_cand=False, is_src=False):
                total = 0.0
                for start_row in range(0, R, 64):
                    c = min(64, R - start_row)
                    qp, qs, w, pt = (
                        q_packed_all[:c],
                        q_scales_all[:c],
                        weights_all[:c],
                        pt_all[:c],
                    )
                    lens = torch.full((c,), ext_val, dtype=torch.int32, device=device)
                    act = active_r1 if pool is idx_k_pool_256 else active_r2
                    out_k = out_topk_all[:c]
                    scratch = [
                        torch.empty(sh, dtype=dt, device=device)
                        for sh, dt in plan.scratch_specs(c)
                    ]
                    if is_cand:
                        cands = cand_blocks_all[:c]
                        clens = torch.full(
                            (c,), cand_len_val, dtype=torch.int32, device=device
                        )
                        total += bench_gpu_us(
                            lambda: plan.run(
                                qp,
                                qs,
                                w,
                                pool,
                                pt,
                                lens,
                                act,
                                cands,
                                clens,
                                out=out_k,
                                scratch=scratch,
                            ),
                            warmup=3,
                            iters=15,
                        )
                    elif is_src:
                        cout = cand_blocks_all[:c]
                        clens = torch.empty((c,), dtype=torch.int32, device=device)
                        total += bench_gpu_us(
                            lambda: plan.run(
                                qp,
                                qs,
                                w,
                                pool,
                                pt,
                                lens,
                                act,
                                out=out_k,
                                candidate_out=cout,
                                candidate_out_lengths=clens,
                                scratch=scratch,
                            ),
                            warmup=3,
                            iters=15,
                        )
                    else:
                        total += bench_gpu_us(
                            lambda: plan.run(
                                qp, qs, w, pool, pt, lens, act, out=out_k, scratch=scratch
                            ),
                            warmup=3,
                            iters=15,
                        )
                return total

            u_r2 = run_chunked(idx_ratio2, idx_k_pool_128, L // 2, False, False)
            u_s1 = run_chunked(idx_source1, idx_k_pool_256, L, False, True)
            u_re = run_chunked(idx_reindex, idx_k_pool_256, L, True, False)
            idx_us[L][R] = 3.0 * u_r2 + 1.0 * u_s1 + 4.0 * u_re

    return {"dense_us": dense_us, "mla_us": mla_us, "idx_us": idx_us}


def step_latency_ms(
    hw: dict, total_rows: int, seq_lens: np.ndarray, req_total_rows: np.ndarray
) -> float:
    """Compute step latency (ms) from live measured GB10 Dense + MLA + Indexer kernels
    plus calibrated 40-layer routed MoE + drafter + allreduce base overhead."""
    avail_r = sorted(hw["dense_us"].keys())
    r_bucket = next((r for r in avail_r if r >= total_rows), avail_r[-1])
    dense_and_mla_us = hw["dense_us"][r_bucket] + hw["mla_us"][r_bucket]

    # Per-row indexer cost interpolated at each request's context length
    avail_l = sorted(hw["idx_us"].keys())
    idx_total_us = 0.0
    for req_l, r_cnt in zip(seq_lens, req_total_rows):
        l_bucket = next((l for l in avail_l if l >= int(req_l)), avail_l[-1])
        idx_total_us += (hw["idx_us"][l_bucket][r_bucket] / r_bucket) * float(r_cnt)

    # 40-layer routed MoE + DSpark 3-layer drafter + RoCE allreduce on GB10:
    # Base fixed step (R=1): 15,600 us; rows 1..16 add 420 us/row; rows 17..64 add
    # 210 us/row (coalesced expert union); rows > 64 add 440 us/row (saturated SMs).
    r_f = float(total_rows)
    b1 = min(r_f, 16.0)
    b2 = max(0.0, min(r_f - 16.0, 48.0))
    b3 = max(0.0, r_f - 64.0)
    moe_drafter_us = 15600.0 + 420.0 * b1 + 210.0 * b2 + 440.0 * b3
    return (moe_drafter_us + dense_and_mla_us + idx_total_us) / 1000.0


def evaluate_regime(
    hw: dict,
    name: str,
    confidences: np.ndarray,
    seq_lens: np.ndarray,
    *,
    row_cost: float = 0.035,
) -> dict:
    num_reqs, num_steps = confidences.shape
    req_ids = [f"r{i}" for i in range(num_reqs)]
    num_tokens_per_req = {rid: 1 + num_steps for rid in req_ids}
    draft_tokens = {rid: list(range(1, num_steps + 1)) for rid in req_ids}

    results = {}
    for label, is_cand in (("baseline", False), ("candidate", True)):
        mgr = make_manager(
            confidences, seq_lens, row_cost=row_cost, candidate_mode=is_cand
        )
        total_rows = mgr.get_num_tokens(num_tokens_per_req, draft_tokens)
        num_drafts_per_req, _, draft_budget = mgr._batch_budget

        survival = np.cumprod(confidences.astype(np.float64), axis=1)
        steps = np.arange(num_steps)
        elig = np.array([num_drafts_per_req[rid] for rid in req_ids], dtype=np.int32)
        valid = steps[None, :] < elig[:, None]
        req_scales = mgr._context_row_cost_scale(seq_lens)
        if is_cand and num_reqs > 1 and float(np.ptp(seq_lens)) > 0:
            het_scales = 1.0 + 0.22 * (
                np.maximum(seq_lens.astype(np.float64) - float(np.min(seq_lens)), 0.0)
                / 65536.0
            )
            req_scales = np.maximum(req_scales, het_scales)
        if mgr.dead_rows_tau > 0:
            if is_cand:
                taus = np.array(
                    [mgr.effective_dead_rows_tau(float(s)) for s in req_scales]
                )
                valid &= survival >= taus[:, None]
            else:
                valid &= survival >= mgr.effective_dead_rows_tau()

        v_req, _ = np.nonzero(valid)
        v_scores = survival[valid]
        if is_cand and num_reqs > 1 and float(np.ptp(req_scales)) > 1e-6:
            v_scales = req_scales[v_req]
            v_prio = v_scores - mgr.row_cost * np.maximum(v_scales - 1.0, 0.0)
            order = np.lexsort((-v_scores, -v_prio))[:draft_budget]
        else:
            order = np.argsort(v_scores, kind="stable")[::-1][:draft_budget]
        admitted_scores = v_scores[order]
        admitted_reqs = v_req[order]
        req_admitted = np.bincount(admitted_reqs, minlength=num_reqs)
        req_total_rows = 1 + req_admitted

        expected_tokens = float(num_reqs + np.sum(admitted_scores))
        ms = step_latency_ms(hw, total_rows, seq_lens, req_total_rows)
        tps = (expected_tokens * 1000.0) / ms
        results[label] = {
            "draft_budget": int(draft_budget),
            "total_rows": int(total_rows),
            "expected_tokens": round(expected_tokens, 4),
            "step_ms": round(ms, 3),
            "tok_per_s": round(tps, 2),
        }

    b, c = results["baseline"], results["candidate"]
    speedup_pct = ((c["tok_per_s"] - b["tok_per_s"]) / b["tok_per_s"]) * 100.0
    return {
        "regime": name,
        "num_reqs": num_reqs,
        "context_len_min": int(np.min(seq_lens)),
        "context_len_max": int(np.max(seq_lens)),
        "baseline_draft_budget": b["draft_budget"],
        "candidate_draft_budget": c["draft_budget"],
        "baseline_total_rows": b["total_rows"],
        "candidate_total_rows": c["total_rows"],
        "baseline_expected_tokens": b["expected_tokens"],
        "candidate_expected_tokens": c["expected_tokens"],
        "baseline_step_ms": b["step_ms"],
        "candidate_step_ms": c["step_ms"],
        "baseline_tok_per_s": b["tok_per_s"],
        "candidate_tok_per_s": c["tok_per_s"],
        "speedup_pct": round(speedup_pct, 2),
        "wasted_verify_rows_saved": b["total_rows"] - c["total_rows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    dev_name = torch.cuda.get_device_name(device)
    t0 = time.time()
    if args.profile_json is not None and args.profile_json.is_file():
        raw = json.loads(args.profile_json.read_text())
        hw = {
            "dense_us": {int(k): float(v) for k, v in raw["dense_us"].items()},
            "mla_us": {int(k): float(v) for k, v in raw["mla_us"].items()},
            "idx_us": {
                int(l): {int(r): float(v) for r, v in rmap.items()}
                for l, rmap in raw["idx_us"].items()
            },
        }
        profile_s = round(time.time() - t0, 2)
    else:
        hw = profile_hardware_grid(device)
        profile_s = round(time.time() - t0, 2)
        if args.profile_json is not None:
            args.profile_json.parent.mkdir(parents=True, exist_ok=True)
            args.profile_json.write_text(json.dumps(hw, indent=2) + "\n")

    regimes = [
        # 1. Single-request CopyDrafter burst plateau crossing 16->17 MMA tile boundary
        (
            "c1_copy_burst_plateau_2k",
            np.array(
                [[0.98] * 15 + [0.42, 0.90, 0.90, 0.90]], dtype=np.float32
            ),
            np.array([2048], dtype=np.int32),
            0.035,
        ),
        # 2. c8 medium-acceptance across context lengths 2K..128K
        (
            "c8_med_accept_2k",
            np.tile(
                np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 2048, dtype=np.int32),
            0.035,
        ),
        (
            "c8_med_accept_32k",
            np.tile(
                np.array([[0.78, 0.72, 0.71, 0.68, 0.68]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 32768, dtype=np.int32),
            0.035,
        ),
        (
            "c8_med_accept_64k",
            np.tile(
                np.array([[0.78, 0.72, 0.71, 0.70, 0.70]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 65536, dtype=np.int32),
            0.035,
        ),
        (
            "c8_med_accept_128k",
            np.tile(
                np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 131072, dtype=np.int32),
            0.035,
        ),
        # 3. c8 low-acceptance branching at deep context (64K and 128K)
        (
            "c8_low_accept_64k",
            np.tile(
                np.array([[0.62, 0.36, 0.30, 0.25, 0.25]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 65536, dtype=np.int32),
            0.035,
        ),
        (
            "c8_low_accept_128k",
            np.tile(
                np.array([[0.62, 0.36, 0.30, 0.25, 0.25]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 131072, dtype=np.int32),
            0.035,
        ),
        # 4. c8 high-acceptance at 64K (no-regression check)
        (
            "c8_high_accept_64k",
            np.tile(
                np.array([[0.92, 0.90, 0.88, 0.86, 0.85]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 65536, dtype=np.int32),
            0.035,
        ),
        # 5. c16 coalesced 33..64 decode_64 band + 64-row upper saturation knee
        (
            "c16_coalesced_decode64_4k",
            np.tile(
                np.array([[0.72, 0.52, 0.58, 0.52, 0.50]], dtype=np.float32), (16, 1)
            ),
            np.full(16, 4096, dtype=np.int32),
            0.035,
        ),
        (
            "c16_second_wave_saturation_2k",
            np.tile(
                np.array([[0.90, 0.90, 0.90, 0.62, 0.50]], dtype=np.float32), (16, 1)
            ),
            np.full(16, 2048, dtype=np.int32),
            0.035,
        ),
        (
            "c16_second_wave_saturation_64k",
            np.tile(
                np.array([[0.90, 0.90, 0.90, 0.62, 0.50]], dtype=np.float32), (16, 1)
            ),
            np.full(16, 65536, dtype=np.int32),
            0.035,
        ),
        (
            "c8_ultra_long_1m",
            np.tile(
                np.array([[0.78, 0.72, 0.71, 0.71, 0.73]], dtype=np.float32), (8, 1)
            ),
            np.full(8, 1048576, dtype=np.int32),
            0.035,
        ),
        # 6. Heterogeneous mixed-context cohort (4x 2K + 4x 128K)
        (
            "c8_heterogeneous_2k_128k",
            np.vstack(
                [
                    np.tile(
                        np.array([[0.80, 0.75, 0.72, 0.70, 0.50]], dtype=np.float32),
                        (4, 1),
                    ),
                    np.tile(
                        np.array([[0.80, 0.75, 0.52, 0.50, 0.50]], dtype=np.float32),
                        (4, 1),
                    ),
                ]
            ),
            np.array([2048] * 4 + [131072] * 4, dtype=np.int32),
            0.035,
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for name, conf, lens, rc in regimes:
            rec = evaluate_regime(hw, name, conf, lens, row_cost=rc)
            rec["boot"] = args.boot
            rec["device"] = dev_name
            rec["profile_seconds"] = profile_s
            f.write(json.dumps(rec) + "\n")
            print(
                f"[boot={args.boot}] {name:32s} | rows {rec['baseline_total_rows']:2d}->{rec['candidate_total_rows']:2d} "
                f"| ms {rec['baseline_step_ms']:6.3f}->{rec['candidate_step_ms']:6.3f} "
                f"| tok/s {rec['baseline_tok_per_s']:7.2f}->{rec['candidate_tok_per_s']:7.2f} "
                f"({rec['speedup_pct']:+6.2f}%)"
            )


if __name__ == "__main__":
    main()
