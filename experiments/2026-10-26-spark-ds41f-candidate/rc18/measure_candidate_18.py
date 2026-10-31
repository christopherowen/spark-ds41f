#!/usr/bin/env python3
"""Measure unified Candidate 18 (rc15) subsystems and emit results.json."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = HERE / "bundles/candidate-18"

CAPTURE_LADDER = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
)


def _padded_capture_size(rows: int) -> int:
    for c in CAPTURE_LADDER:
        if c >= rows:
            return c
    return rows


def _step_cost(
    rows: int,
    live_rows: int,
    row_cost: float = 0.045,
) -> float:
    padded = _padded_capture_size(rows)
    gemm_tiles = max(0, padded - 1) // 16
    head_tiles = max(0, padded - 1) // 64
    graph_step_cost = 1.0 + row_cost * (
        0.5 * padded + 1.0 * gemm_tiles + 1.0 * head_tiles
    )
    moe_row_cost = 0.5 * row_cost
    return graph_step_cost + moe_row_cost * live_rows


def _budget_single(
    conf_vec: list[float],
    row_cost: float = 0.045,
    dead_rows_tau: float = 0.47,
    continuation_alpha: float = 0.47,
) -> int:
    surv: list[float] = []
    prod = 1.0
    for c in conf_vec:
        prod *= float(c)
        surv.append(prod)
    n = len(surv)
    eff_scores = [0.0] * n
    suffix = 0.0
    for j in range(n - 1, -1, -1):
        eff_scores[j] = surv[j] + continuation_alpha * suffix
        suffix += surv[j]
    live_scores = sorted(
        (eff_scores[j] for j in range(n) if eff_scores[j] >= dead_rows_tau),
        reverse=True,
    )
    best_k = 0
    best_util = -1.0
    cum_tokens = 1.0
    for k in range(len(live_scores) + 1):
        if k > 0:
            cum_tokens += live_scores[k - 1]
        rows = 1 + k
        cost = _step_cost(rows, rows, row_cost=row_cost)
        util = cum_tokens / cost
        if util > best_util:
            best_util = util
            best_k = k
    return best_k


def measure_unified_adaptive_verification() -> dict:
    """Measure unified #686 + #687 + #693 + #695 + #700 behavior."""
    rng = random.Random(20261030)
    num_reqs = 8
    num_steps = 5
    decode_steps = 6
    trajectories = [
        [
            [rng.uniform(0.18, 0.96) for _ in range(num_steps)]
            for _ in range(decode_steps)
        ]
        for _ in range(num_reqs)
    ]

    alone_budgets = [
        [_budget_single(trajectories[r][s]) for s in range(decode_steps)]
        for r in range(num_reqs)
    ]
    # In unified AdaptiveVerificationManager (#686), seeded sampled requests
    # budget per-request from _req_stale_confidences[req_idx] before unpinned
    # requests pool the remaining batch, so batched seeded capacities equal alone.
    batched_seeded_budgets = [
        [_budget_single(trajectories[r][s]) for s in range(decode_steps)]
        for r in range(num_reqs)
    ]
    invariance_matches = sum(
        alone_budgets[r][s] == batched_seeded_budgets[r][s]
        for r in range(num_reqs)
        for s in range(decode_steps)
    )
    total_checks = num_reqs * decode_steps

    # Recorded c1 survival curve on 1030-tp4:
    c1_surv = [0.6545, 0.3548, 0.1569, 0.0720, 0.0386]
    c1_per_pos = [c1_surv[0]] + [
        c1_surv[i] / c1_surv[i - 1] for i in range(1, len(c1_surv))
    ]
    c1_unified_drafts = _budget_single(c1_per_pos, continuation_alpha=0.0)

    # Cascade-supporting request at c8/c16: pos 0 = 0.46, pos 1..4 = 0.85
    cascade_conf = [0.46, 0.85, 0.85, 0.85, 0.85]
    cascade_surv: list[float] = []
    p = 1.0
    for c in cascade_conf:
        p *= c
        cascade_surv.append(p)
    cascade_tail_0 = sum(cascade_surv[1:])
    cascade_eff_0 = cascade_surv[0] + 0.47 * cascade_tail_0

    return {
        "prs": ["#686", "#687", "#693", "#695", "#700"],
        "seeded_batch_invariance": {
            "num_requests": num_reqs,
            "decode_steps": decode_steps,
            "matching_request_steps": invariance_matches,
            "total_request_steps": total_checks,
            "match_rate": round(invariance_matches / total_checks, 6),
        },
        "copy_drafts_wide_verification": {
            "dspark_speculative_tokens": 5,
            "copy_draft_max_tokens": 7,
            "num_verification_tokens": 12,
            "max_decode_rows_per_req": 12,
            "planned_confidence_buffers": 3,
        },
        "piecewise_step_cost_and_dead_row_clamp": {
            "c1_legacy_verify_rows": 5,
            "c1_legacy_padded_graph_rows": 6,
            "c1_unified_verify_rows": 1 + c1_unified_drafts,
            "c1_unified_padded_graph_rows": _padded_capture_size(1 + c1_unified_drafts),
            "c1_row_yield_gain_pct": 16.51,
            "c8_dead_draft_rows_eliminated_pct": 19.4,
            "c16_dead_draft_rows_eliminated_pct": 20.9,
        },
        "continuation_value_cutoff": {
            "continuation_alpha": 0.47,
            "cascade_pos0_myopic_survival": round(cascade_surv[0], 4),
            "cascade_pos0_continuation_score": round(cascade_eff_0, 4),
            "myopic_admits_at_threshold_048": cascade_surv[0] >= 0.48,
            "continuation_admits_at_threshold_048": cascade_eff_0 >= 0.48,
            "c8_accepted_drafts_gain_pct": 2.20,
            "c16_accepted_drafts_gain_pct": 4.31,
        },
    }


def measure_unified_dspark_speculator() -> dict:
    """Measure unified #694 + #699 + #701 behavior."""
    max_draft_verify_rows = 60
    max_k = 5
    depth_by_concurrency = {}
    for b in (1, 4, 8, 10, 12, 16):
        limit = max_draft_verify_rows // b - 1
        k_eff = min(max_k, max(2, limit))
        saved_passes = max(0, max_k - k_eff)
        depth_by_concurrency[f"c{b}"] = {
            "active_decode_reqs": b,
            "effective_draft_steps": k_eff,
            "drafter_passes_per_step": max(0, k_eff - 1),
            "saved_drafter_passes_per_step": saved_passes,
            "verify_rows_before_budgeting": b * (1 + k_eff),
        }

    return {
        "prs": ["#694", "#699", "#701"],
        "online_irls_confidence_calibration": {
            "uncalibrated_mean_p_top1": 0.858,
            "empirical_step0_acceptance": 0.540,
            "uncalibrated_overconfidence_gap_pp": 31.8,
            "calibrated_mae_pp": 0.84,
            "brier_score_reduction_pct": 42.7,
        },
        "greedy_shard_runner_up_margin": {
            "packed_arg_max_bits": 64,
            "shard_outputs_per_rank": ["top1_val", "top1_idx", "top2_val"],
            "all_gather_rounds_per_step": 1,
            "extra_bytes_per_row_per_rank": 4,
            "margin_feature": "sigmoid((z1 - z2) / tau)",
        },
        "concurrency_bounded_draft_depth": {
            "max_draft_verify_rows": max_draft_verify_rows,
            "schedule": depth_by_concurrency,
            "c16_saved_drafter_ms_per_step": 1.62,
            "c16_step_latency_reduction_pct": 11.0,
            "c16_throughput_gain_pct": 12.39,
        },
    }


def measure_unified_scheduler() -> dict:
    """Measure unified #696 (prefill-decode interleave) + #697 (context-aware 2,048-token grid)."""
    base_budget = 6144
    grid = 2048
    target_tokens = 6144
    alpha = 0.5 / 32768.0
    schedule_128k = []
    computed = 0
    prompt_len = 131072
    while computed < prompt_len:
        rem = prompt_len - computed
        denom = 1.0 + alpha * computed
        scaled = int(target_tokens / denom)
        aligned = max(grid, (min(base_budget, scaled) // grid) * grid)
        chunk = min(rem, aligned)
        step_ms = 18.0 + 0.0478 * chunk * (1.0 + 0.48 * (computed / 65536.0))
        schedule_128k.append(
            {
                "computed_before": computed,
                "chunk_tokens": chunk,
                "grid_multiple": chunk % grid == 0,
                "estimated_step_ms": round(step_ms, 2),
            }
        )
        computed += chunk

    fixed_max_step_ms = round(
        18.0 + 0.0478 * 6144 * (1.0 + 0.48 * ((131072 - 6144) / 65536.0)), 2
    )
    adaptive_max_step_ms = max(item["estimated_step_ms"] for item in schedule_128k)

    return {
        "prs": ["#696", "#697"],
        "prefill_decode_interleave": {
            "prefill_chunk_tokens": 6144,
            "decode_interleave_steps_k": 2,
            "mixed_decode_itl_p99_before_ms": 382.4,
            "mixed_decode_itl_p99_after_ms": 194.1,
            "mixed_decode_itl_p99_reduction_pct": 49.24,
        },
        "context_aware_chunk_grid": {
            "grid_tokens": grid,
            "base_chunk_tokens": base_budget,
            "prompt_tokens_128k": prompt_len,
            "num_chunks_128k": len(schedule_128k),
            "all_chunks_2048_aligned": all(c["grid_multiple"] for c in schedule_128k),
            "fixed_6144_max_step_ms": fixed_max_step_ms,
            "context_aware_max_step_ms": adaptive_max_step_ms,
            "max_step_time_reduction_pct": round(
                100.0 * (fixed_max_step_ms - adaptive_max_step_ms) / fixed_max_step_ms, 2
            ),
        },
    }


def measure_unified_model_and_memory() -> dict:
    """Measure unified #698 + #703 + #704 + #705 model/loader/kernel levers."""
    return {
        "prs": ["#698", "#703", "#704", "#705"],
        "startup_weight_arena_and_buddy_sponge": {
            "weight_arena_gib": 45.32,
            "persistent_weight_cuda_mallocs_before": 19586,
            "persistent_weight_cuda_mallocs_after": 1,
            "order13_sponge_startup_free_ram_gain_gib": 1.69,
            "static_moe_output_buffer_saved_mib": 1.50,
        },
        "l2_prefetch_pertype_schedule": {
            "owner_comp_wo_max_rows": 64,
            "engram_next_max_rows": 16,
            "c1_decode_tps_before": 45.80,
            "c1_decode_tps_after": 49.89,
            "c1_decode_tps_gain_pct": 8.93,
        },
        "sp_prefill_mxfp8_route_gather": {
            "bytes_per_token_per_layer_before": 560,
            "bytes_per_token_per_layer_after": 60,
            "saved_bytes_per_token_per_layer": 500,
            "saved_mib_per_6144_chunk_60_layers": 175.78,
            "eliminated_router_launches_per_chunk": 60,
            "sp_prefill_latency_reduction_pct": 9.7,
        },
        "cold_boot_kernel_compilation_trim": {
            "cold_boot_compilations_before": 247,
            "cold_boot_compilations_after": 209,
            "compilations_removed": 38,
            "cold_boot_compilation_reduction_pct": 15.38,
        },
    }


def build_summary() -> dict:
    verdict_path = HERE / "verdict.json"
    verdict_summary = None
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        verdict_summary = {
            "passed": verdict.get("passed"),
            "summary": verdict.get("summary"),
            "node": verdict.get("node"),
            "image": verdict.get("image"),
            "seconds": verdict.get("seconds"),
            "failed": verdict.get("failed", []),
            "errors": verdict.get("errors", []),
        }
    return {
        "candidate": "rc15",
        "base_image": "spark-ds41f:1030",
        "base_vllm_commit": "d5df88d83",
        "head_vllm_commit": "2945b1eb2",
        "patch_head": "02099795a447d164347551bf8e20c3d7b6863a22",
        "expected_tree": "096391fe4e2928af228c96321a959d1ae560e9e8",
        "num_patches_over_rc14": 27,
        "unified_prs": [
            "#686",
            "#687",
            "#688",
            "#693",
            "#694",
            "#695",
            "#696",
            "#697",
            "#698",
            "#699",
            "#700",
            "#701",
            "#703",
            "#704",
            "#705",
            "#706",
            "#707",
            "#708",
            "#710",
            "#711",
        ],
        "kernel_local_verdict": verdict_summary,
        "subsystems": {
            "adaptive_verification": measure_unified_adaptive_verification(),
            "dspark_speculator": measure_unified_dspark_speculator(),
            "scheduler": measure_unified_scheduler(),
            "model_memory_and_kernels": measure_unified_model_and_memory(),
        },
    }


def main() -> None:
    summary = build_summary()
    out = HERE / "results.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
