#!/usr/bin/env python3
"""Piecewise CUDA-graph / 16-row GEMM tile / dead-row step-cost model and trace replay
for DSpark AdaptiveVerificationManager.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

TP4_CAPTURE_SIZES: tuple[int, ...] = (
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
TP3_CAPTURE_SIZES: tuple[int, ...] = (
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
)

GRAPH_PAD_COST_FRACTION = 0.5
GEMM_TILE_ROW_EQUIV = 1.0
LM_HEAD_TILE_ROW_EQUIV = 1.0

WORKLOAD_MANIFESTS: dict[str, tuple[str, int]] = {
    "single-stream/c1": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-single-stream/c1.json",
        1,
    ),
    "release/poisson": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/poisson.json",
        6,
    ),
    "release/c8": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c8.json",
        7,
    ),
    "release/c16": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c16.json",
        13,
    ),
    "sampled/c1": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c1.json",
        1,
    ),
    "sampled/c8": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c8.json",
        7,
    ),
    "sampled/c16": (
        "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c16.json",
        13,
    ),
}


def padded_capture_size(
    rows: int, capture_sizes: Sequence[int] = TP4_CAPTURE_SIZES
) -> int:
    """Snap ``rows`` up to its captured CUDA-graph size when within the capture ladder."""
    if not capture_sizes:
        return rows
    idx = bisect.bisect_left(capture_sizes, rows)
    if idx < len(capture_sizes):
        return int(capture_sizes[idx])
    return rows


def step_costs(
    non_draft_rows: int,
    scores: Sequence[float],
    row_cost: float,
    *,
    capture_sizes: Sequence[int] = (),
    dead_rows_tau: float = 0.0,
) -> list[float]:
    """Piecewise step-cost lookup ``step_cost[capture_size(rows)] + moe_row_cost * live_rows``.

    Matches ``AdaptiveVerificationManager.step_costs`` exactly.
    """
    ladder = tuple(sorted(capture_sizes))
    if not ladder and dead_rows_tau <= 0.0:
        return [
            1.0 + row_cost * (non_draft_rows + k) for k in range(len(scores) + 1)
        ]

    costs: list[float] = []
    live_drafts = 0
    moe_row_cost = (1.0 - GRAPH_PAD_COST_FRACTION) * row_cost
    for k in range(len(scores) + 1):
        if k > 0:
            if dead_rows_tau <= 0.0 or scores[k - 1] >= dead_rows_tau:
                live_drafts += 1
        rows = non_draft_rows + k
        padded_rows = padded_capture_size(rows, ladder)
        if ladder:
            gemm_tiles = max(0, padded_rows - 1) // 16
            head_tiles = max(0, padded_rows - 1) // 64
        else:
            gemm_tiles = 0
            head_tiles = 0
        live_rows = non_draft_rows + live_drafts if dead_rows_tau > 0.0 else rows
        graph_step_cost = 1.0 + row_cost * (
            GRAPH_PAD_COST_FRACTION * padded_rows
            + GEMM_TILE_ROW_EQUIV * gemm_tiles
            + LM_HEAD_TILE_ROW_EQUIV * head_tiles
        )
        costs.append(graph_step_cost + moe_row_cost * live_rows)
    return costs


def select_draft_budget(
    num_sampling_requests: int,
    non_draft_rows: int,
    scores: Sequence[float],
    row_cost: float,
    *,
    capture_sizes: Sequence[int] = (),
    dead_rows_tau: float = 0.0,
) -> tuple[int, list[float]]:
    """Select the draft budget maximising ``expected_tokens / step_costs``."""
    sorted_scores = sorted((float(s) for s in scores), reverse=True)
    costs = step_costs(
        non_draft_rows,
        sorted_scores,
        row_cost,
        capture_sizes=capture_sizes,
        dead_rows_tau=dead_rows_tau,
    )
    utilities: list[float] = []
    expected = float(num_sampling_requests)
    for k, cost in enumerate(costs):
        if k > 0:
            s = sorted_scores[k - 1]
            if dead_rows_tau <= 0.0 or s >= dead_rows_tau:
                expected += s
        utilities.append(expected / cost)
    best_k = max(range(len(utilities)), key=lambda i: utilities[i])
    return best_k, utilities


def marginal_survival_threshold(
    num_sampling_requests: int,
    non_draft_rows: int,
    admitted_scores: Sequence[float],
    row_cost: float,
    *,
    capture_sizes: Sequence[int] = (),
) -> float:
    """Minimum survival ``s_{k+1}`` required for draft slot ``k + 1`` to beat budget ``k``."""
    probe_scores = list(admitted_scores) + [1.0]
    costs = step_costs(
        non_draft_rows,
        probe_scores,
        row_cost,
        capture_sizes=capture_sizes,
        dead_rows_tau=0.0,
    )
    k = len(admitted_scores)
    expected_k = float(num_sampling_requests) + sum(admitted_scores)
    return expected_k * (costs[k + 1] / costs[k] - 1.0)


def _interp(curve: dict[int, float], x: float) -> float:
    keys = sorted(curve)
    if x <= keys[0]:
        return curve[keys[0]]
    if x >= keys[-1]:
        return curve[keys[-1]]
    for k0, k1 in zip(keys, keys[1:]):
        if k0 <= x <= k1:
            if k0 == k1:
                return curve[k0]
            return curve[k0] + (curve[k1] - curve[k0]) * (x - k0) / (k1 - k0)
    return curve[keys[-1]]


def hardware_step_ms(
    static_costs: dict,
    num_reqs: int,
    rows: int,
    live_rows: int,
    *,
    capture_sizes: Sequence[int] = TP4_CAPTURE_SIZES,
) -> float:
    """Hardware step time (ms) calibrated to 1030-tp4/grid.json and static-r6v-25x3:
    - Drafter curve by active request count from static-r6v-25x3
    - Padded CUDA-graph dense cost (0.34 ms/row) + 16-row GEMM tile step (+0.60 ms)
      + 64-row LM-head tile step (+0.60 ms)
    - Live-row routed MoE cost (0.38 ms/row, skipped on padding and dead rows)
    """
    draft_map = {int(k): float(v) for k, v in static_costs["draft_curve"]}
    padded = padded_capture_size(rows, capture_sizes)
    gemm_tiles = max(0, padded - 1) // 16
    head_tiles = max(0, padded - 1) // 64
    draft_ms = _interp(draft_map, num_reqs)
    v_base = 12.65
    pad_ms = 0.34 * padded + 0.60 * gemm_tiles + 0.60 * head_tiles
    moe_ms = 0.38 * live_rows
    return draft_ms + v_base + pad_ms + moe_ms


def _quantile_confidence_score(pos: int, q: float, is_sampled: bool) -> float:
    """Calibrated stale confidence running-product score at quantile ``q in (0, 1)``.

    Matches the recorded 1030-tp4 coverage quantiles:
    - Top cov_c16 (~45%, 19%, 9%, 5%) has survival >= 0.25 (live at all concurrencies).
    - Middle band [cov_c16, cov_c8] (~55%, 25%, 11%, 6%) has survival 0.20..0.25.
    - Tail band [cov_c8, cov_c1] (~96%, 63%, 32%, 16%) has stale survival 0.068..0.20
      (admitted by linear row_cost <= 0.035 at c1, but truncated by dead_row_cut on
      device so its empirical marginal acceptance is only 0.0-3.9%).
    """
    scale = 0.96 if is_sampled else 1.0
    if pos == 0:
        return scale * (0.35 + 0.60 * (q**0.7))
    c1_cov = (1.0, 0.959, 0.635, 0.321, 0.163)[pos]
    c8_cov = (1.0, 0.550, 0.246, 0.113, 0.058)[pos]
    c16_cov = (1.0, 0.455, 0.195, 0.092, 0.048)[pos]
    q_c1 = 1.0 - c1_cov
    q_c8 = 1.0 - c8_cov
    q_c16 = 1.0 - c16_cov
    if q < q_c1:
        return scale * 0.066 * (q / max(1e-6, q_c1))
    if q < q_c8:
        t = (q - q_c1) / max(1e-6, q_c8 - q_c1)
        return scale * (0.068 + t * (0.195 - 0.068))
    if q < q_c16:
        t = (q - q_c8) / max(1e-6, q_c16 - q_c8)
        return scale * (0.205 + t * (0.285 - 0.205))
    t = (q - q_c16) / max(1e-6, 1.0 - q_c16)
    return scale * (0.285 + (t**0.65) * (0.90 - 0.285))


def _empirical_slot_yield(pos: int, q: float, is_sampled: bool, tau: float) -> float:
    """Empirical accepted-token probability of position ``pos`` at quantile ``q``
    from the 1030-tp4 multi-concurrency Lorenz curve (where c8 -> c1 marginal yield
    is 3.97% at pos 1, 1.58% at pos 2, and ~0.2% at pos 3-4)."""
    score = _quantile_confidence_score(pos, q, is_sampled)
    if score < tau:
        return 0.003 * (score / max(1e-6, tau))
    c8_cov = (1.0, 0.550, 0.246, 0.113, 0.058)[pos]
    c16_cov = (1.0, 0.455, 0.195, 0.092, 0.048)[pos]
    if pos > 0 and q < 1.0 - c8_cov:
        # Tail between c8 and c1 has measured empirical marginal yield 0.2%..3.9%
        marg = (0.0, 0.0397, 0.0158, 0.0020, 0.0028)[pos]
        return marg * (0.5 if is_sampled else 1.0)
    if pos > 0 and q < 1.0 - c16_cov:
        # Band between c16 and c8 has measured marginal yield ~4.3%..13.5%
        marg = (0.0, 0.0432, 0.0819, 0.1346, 0.0485)[pos]
        return marg * (0.85 if is_sampled else 1.0)
    return score


def replay_workload_trace(
    manifest: dict,
    static_costs: dict,
    num_reqs: int,
    *,
    is_sampled: bool,
    row_cost: float,
    dead_rows_tau: float,
    piecewise: bool,
    steps: int = 400,
    seed: int = 17,
) -> dict[str, float]:
    """Replay a deterministic step trace calibrated to the recorded 1030-tp4
    per-position coverage and survival Lorenz curve."""
    rng = random.Random(seed + num_reqs * 31 + (100 if is_sampled else 0))
    capture_sizes = TP4_CAPTURE_SIZES if piecewise else ()
    host_tau = dead_rows_tau if piecewise else 0.0

    total_ver_rows = 0
    total_padded_rows = 0
    total_live_rows = 0
    total_gemm_tiles = 0
    total_drafts_admitted = 0
    total_accepted_drafts = 0.0
    total_output_tokens = 0.0
    total_step_ms = 0.0
    graph_jumps_past_4 = 0

    for _ in range(steps):
        slots: list[tuple[float, float]] = []
        for _r in range(num_reqs):
            # Correlated per-request quantile q in (0, 1), monotonically decreasing
            # survival across positions 0..4.
            q_req = rng.random()
            prev_s = 1.0
            prev_y = 1.0
            for pos in range(5):
                q_pos = min(0.999, max(0.001, 0.85 * q_req + 0.15 * rng.random()))
                s = min(prev_s * 0.95, _quantile_confidence_score(pos, q_pos, is_sampled))
                y = min(prev_y * 0.95, _empirical_slot_yield(pos, q_pos, is_sampled, dead_rows_tau))
                prev_s = s
                prev_y = y
                slots.append((s, y))

        slots.sort(key=lambda pair: pair[0], reverse=True)
        flat_scores = [s for s, _ in slots]
        budget, _ = select_draft_budget(
            num_reqs,
            num_reqs,
            flat_scores,
            row_cost,
            capture_sizes=capture_sizes,
            dead_rows_tau=host_tau,
        )
        admitted = slots[:budget]
        rows = num_reqs + budget
        padded = padded_capture_size(rows, TP4_CAPTURE_SIZES)
        live_drafts = sum(1 for s, _ in admitted if s >= dead_rows_tau)
        live_rows = num_reqs + live_drafts
        accepted_drafts = sum(y for _, y in admitted)

        total_ver_rows += rows
        total_padded_rows += padded
        total_live_rows += live_rows
        total_gemm_tiles += max(0, padded - 1) // 16 + 1
        total_drafts_admitted += budget
        total_accepted_drafts += accepted_drafts
        total_output_tokens += num_reqs + accepted_drafts
        total_step_ms += hardware_step_ms(static_costs, num_reqs, rows, live_rows)
        if num_reqs == 1 and padded > 4:
            graph_jumps_past_4 += 1

    mean_ver_rows = total_ver_rows / steps
    mean_padded_rows = total_padded_rows / steps
    mean_live_rows = total_live_rows / steps
    mean_gemm_tiles = total_gemm_tiles / steps
    mean_drafts_per_req = total_drafts_admitted / (steps * num_reqs)
    mean_accepted_per_req = total_accepted_drafts / (steps * num_reqs)
    row_yield = (
        total_accepted_drafts / total_drafts_admitted
        if total_drafts_admitted > 0
        else 0.0
    )
    mean_step_ms = total_step_ms / steps
    decode_tps = (total_output_tokens / steps) * (1000.0 / mean_step_ms)

    return {
        "ver_rows_per_step": round(mean_ver_rows, 3),
        "padded_rows_per_step": round(mean_padded_rows, 3),
        "live_rows_per_step": round(mean_live_rows, 3),
        "gemm_16_tiles_per_step": round(mean_gemm_tiles, 3),
        "drafts_per_req": round(mean_drafts_per_req, 3),
        "accepted_per_req": round(mean_accepted_per_req, 3),
        "speculative_row_yield": round(row_yield, 4),
        "step_ms": round(mean_step_ms, 3),
        "decode_tps": round(decode_tps, 2),
        "c1_6row_graph_frac": round(graph_jumps_past_4 / steps, 4)
        if num_reqs == 1
        else 0.0,
    }


def empirical_c8_to_c1_marginal_yield() -> dict[str, list[dict[str, float]]]:
    """Direct empirical proof from 1030-tp4 manifests: extra draft coverage admitted
    at c1 over c8 on the identical 256-prompt set yields < 4% at pos 1 and ~0% at
    positions 2, 3, and 4 because those slots lie below dead_rows_tau."""
    out: dict[str, list[dict[str, float]]] = {}
    for label, c1_rel, c8_rel, c16_rel in (
        (
            "greedy",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-single-stream/c1.json",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c8.json",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4/c16.json",
        ),
        (
            "sampled",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c1.json",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c8.json",
            "manifests/suites/2026-10-30-spark-ds41f-1030-tp4-sampled/c16.json",
        ),
    ):
        c1_srv = json.loads((ROOT / c1_rel).read_text())["summary"]["server"]
        c8_srv = json.loads((ROOT / c8_rel).read_text())["summary"]["server"]
        c16_srv = json.loads((ROOT / c16_rel).read_text())["summary"]["server"]
        rows: list[dict[str, float]] = []
        for pos in range(5):
            cov1, surv1 = (
                float(c1_srv[f"coverage_at_{pos}"]["mean"]),
                float(c1_srv[f"survival_at_{pos}"]["mean"]),
            )
            cov8, surv8 = (
                float(c8_srv[f"coverage_at_{pos}"]["mean"]),
                float(c8_srv[f"survival_at_{pos}"]["mean"]),
            )
            cov16, surv16 = (
                float(c16_srv[f"coverage_at_{pos}"]["mean"]),
                float(c16_srv[f"survival_at_{pos}"]["mean"]),
            )
            d_cov = cov1 - cov8
            d_surv = surv1 - surv8
            rows.append(
                {
                    "position": pos,
                    "c16_coverage": round(cov16, 4),
                    "c16_survival": round(surv16, 4),
                    "c8_coverage": round(cov8, 4),
                    "c8_survival": round(surv8, 4),
                    "c1_coverage": round(cov1, 4),
                    "c1_survival": round(surv1, 4),
                    "delta_coverage_c8_to_c1": round(d_cov, 4),
                    "delta_survival_c8_to_c1": round(d_surv, 4),
                    "marginal_yield_c8_to_c1": round(
                        d_surv / d_cov if d_cov > 1e-6 else 0.0, 4
                    ),
                }
            )
        out[label] = rows
    return out


def build_summary() -> dict:
    static_costs = json.loads(
        (
            ROOT
            / "experiments/2026-10-18-dspark-static-costs/static-r6v-25x3.json"
        ).read_text()
    )
    manifests = {
        name: (json.loads((ROOT / rel_path).read_text()), num_reqs)
        for name, (rel_path, num_reqs) in WORKLOAD_MANIFESTS.items()
    }

    # 1. Marginal survival thresholds at c1 across row_cost values
    c1_surv = [0.6545, 0.3548, 0.1569, 0.0720, 0.0386]
    c1_thresholds: dict[str, dict[str, float]] = {}
    for rc in (0.02, 0.035, 0.04, 0.045, 0.05):
        c1_thresholds[f"{rc:.3f}"] = {
            "linear_pos3_r3_to_4": round(
                marginal_survival_threshold(1, 1, c1_surv[:2], rc), 4
            ),
            "linear_pos4_r4_to_5": round(
                marginal_survival_threshold(1, 1, c1_surv[:3], rc), 4
            ),
            "piecewise_pos3_r3_to_4": round(
                marginal_survival_threshold(
                    1, 1, c1_surv[:2], rc, capture_sizes=TP4_CAPTURE_SIZES
                ),
                4,
            ),
            "piecewise_pos4_r4_to_5_pad6": round(
                marginal_survival_threshold(
                    1, 1, c1_surv[:3], rc, capture_sizes=TP4_CAPTURE_SIZES
                ),
                4,
            ),
        }

    # 2. Policy comparison across workloads
    policies = {
        "linear_rc020_tau020": (0.020, 0.20, False),
        "linear_rc035_tau020_baseline": (0.035, 0.20, False),
        "linear_rc045_tau025": (0.045, 0.25, False),
        "piecewise_rc045_tau025": (0.045, 0.25, True),
        "piecewise_rc050_tau030": (0.050, 0.30, True),
    }

    workloads_out: dict[str, dict] = {}
    for w_name, (manifest, num_reqs) in manifests.items():
        srv = manifest["summary"]["server"]
        is_sampled = w_name.startswith("sampled/")
        recorded = {
            "recorded_output_tps": round(
                float(manifest["summary"]["output_tps"]["mean"]), 2
            ),
            "recorded_drafts_per_req": round(
                float(srv["draft_tokens_per_draft"]["mean"]), 3
            ),
            "recorded_accepted_per_req": round(
                float(srv["accepted_per_draft"]["mean"]), 3
            ),
            "recorded_coverage": [
                round(float(srv[f"coverage_at_{i}"]["mean"]), 4) for i in range(5)
            ],
            "recorded_survival": [
                round(float(srv[f"survival_at_{i}"]["mean"]), 4) for i in range(5)
            ],
        }
        replays = {
            p_name: replay_workload_trace(
                manifest,
                static_costs,
                num_reqs,
                is_sampled=is_sampled,
                row_cost=rc,
                dead_rows_tau=tau,
                piecewise=pw,
            )
            for p_name, (rc, tau, pw) in policies.items()
        }
        base_tps = replays["linear_rc035_tau020_baseline"]["decode_tps"]
        base_yield = replays["linear_rc035_tau020_baseline"]["speculative_row_yield"]
        for p_name, rep in replays.items():
            rep["tps_vs_baseline_pct"] = round(
                100.0 * (rep["decode_tps"] / base_tps - 1.0), 2
            )
            rep["row_yield_vs_baseline_pct"] = round(
                100.0 * (rep["speculative_row_yield"] / base_yield - 1.0), 2
            )
        workloads_out[w_name] = {
            "recorded": recorded,
            "policies": replays,
        }

    return {
        "capture_sizes_tp4": list(TP4_CAPTURE_SIZES),
        "c1_marginal_survival_thresholds": c1_thresholds,
        "empirical_c8_to_c1_marginal_yield": empirical_c8_to_c1_marginal_yield(),
        "workloads": workloads_out,
    }


def main() -> None:
    summary = build_summary()
    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")
    for w_name, w_data in summary["workloads"].items():
        base = w_data["policies"]["linear_rc035_tau020_baseline"]
        pw = w_data["policies"]["piecewise_rc045_tau025"]
        print(
            f"  {w_name:18s}: ver_rows {base['ver_rows_per_step']:5.2f} (pad {base['padded_rows_per_step']:5.2f}) -> "
            f"{pw['ver_rows_per_step']:5.2f} (pad {pw['padded_rows_per_step']:5.2f}) | "
            f"row_yield {base['speculative_row_yield']:.3f} -> {pw['speculative_row_yield']:.3f} ({pw['row_yield_vs_baseline_pct']:+.1f}%) | "
            f"decode_tps {pw['tps_vs_baseline_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
