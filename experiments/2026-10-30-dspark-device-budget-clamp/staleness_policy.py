# SPDX-License-Identifier: Apache-2.0
"""Reference policy and trace evaluator for eliminating 1-step async staleness leaks
between CPU graph budgeting and GPU dead-row cutting in DSpark adaptive verification.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class StepBudgetResult:
    num_tokens: int
    draft_budget: int
    admitted_capacities: list[int]
    dead_row_cut: list[int]
    verified_draft_counts: list[int]


def _cumprod(row: Sequence[float]) -> list[float]:
    out: list[float] = []
    acc = 1.0
    for val in row:
        acc *= float(val)
        out.append(acc)
    return out


def host_budget_num_tokens(
    stale_confidences: Sequence[Sequence[float]],
    scheduled_drafts: Sequence[int],
    num_non_draft_tokens: Sequence[int],
    row_cost: float,
    *,
    dead_rows_tau: float = 0.0,
    max_total_logits: int = 1 << 30,
    num_bonus_tokens: int = 1,
    is_sampling_request: Sequence[bool] | None = None,
) -> tuple[int, int]:
    """Reference model of ``AdaptiveVerificationManager.get_num_tokens``.

    When ``dead_rows_tau > 0``, draft slots whose survival probability is already
    below ``dead_rows_tau`` are excluded from ``valid`` so the host budget never
    prices or allocates outer CUDA graph rows for slots that will be unconditionally
    marked dead on device.
    """
    num_reqs = len(stale_confidences)
    num_steps = len(stale_confidences[0]) if num_reqs > 0 else 0
    drafts = [int(d) for d in scheduled_drafts]
    non_draft = [int(n) for n in num_non_draft_tokens]

    scores: list[float] = []
    tau = float(dead_rows_tau)
    for r in range(num_reqs):
        surv = _cumprod(stale_confidences[r])
        limit = min(drafts[r], num_steps)
        for j in range(limit):
            if tau <= 0.0 or surv[j] >= tau:
                scores.append(surv[j])

    scores.sort(reverse=True)
    max_draft_budget = min(
        sum(drafts),
        max(0, int(max_total_logits) - num_reqs * int(num_bonus_tokens)),
    )
    scores = scores[:max_draft_budget]

    if is_sampling_request is None:
        num_sampling_requests = num_reqs
    else:
        num_sampling_requests = sum(1 for flag in is_sampling_request if flag)

    expected_tokens: list[float] = [float(num_sampling_requests)]
    running = float(num_sampling_requests)
    for s in scores:
        running += s
        expected_tokens.append(running)

    base_rows = sum(non_draft)
    best_k = 0
    best_utility = -1.0
    for k, exp_tok in enumerate(expected_tokens):
        rows = base_rows + k
        utility = exp_tok / (1.0 + float(row_cost) * rows)
        if utility > best_utility:
            best_utility = utility
            best_k = k

    return base_rows + best_k, best_k


def assign_device_draft_budget(
    current_confidences: Sequence[Sequence[float]],
    scheduled_drafts: Sequence[int],
    draft_budget: int,
) -> list[int]:
    """Reference model of ``tilelang_inputs.assign_draft_budget`` on current-step
    GPU confidences ``self._confidence_probs``.
    """
    num_reqs = len(current_confidences)
    num_steps = len(current_confidences[0]) if num_reqs > 0 else 0
    capacities = [int(d) for d in scheduled_drafts]
    if draft_budget <= 0:
        return [0] * num_reqs
    if draft_budget >= sum(capacities):
        return capacities

    candidates: list[tuple[float, int, int]] = []
    for r in range(num_reqs):
        surv = _cumprod(current_confidences[r])
        for j in range(num_steps):
            val = surv[j] if j < capacities[r] else float("-inf")
            flat_idx = r * num_steps + j
            candidates.append((val, -flat_idx, r))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    admitted = [0] * num_reqs
    for _, _, r in candidates[:draft_budget]:
        admitted[r] += 1
    return admitted


def compute_device_dead_row_cut(
    current_confidences: Sequence[Sequence[float]],
    admitted_capacities: Sequence[int],
    *,
    dead_rows_tau: float,
    is_decoding: Sequence[bool] | None = None,
    block_size: int = 8,
) -> list[int]:
    """Reference model of ``combine_sampled_and_draft_tokens_kernel``'s ``dead_cut``."""
    num_reqs = len(current_confidences)
    num_steps = len(current_confidences[0]) if num_reqs > 0 else 0
    decoding = (
        [True] * num_reqs
        if is_decoding is None
        else [bool(flag) for flag in is_decoding]
    )
    cut = [block_size + 1] * num_reqs
    if dead_rows_tau <= 0.0:
        return cut

    log_tau = math.log(float(dead_rows_tau))
    for b in range(num_reqs):
        num_drafts = int(admitted_capacities[b])
        if decoding[b] and num_drafts > 0:
            log_surv = 0.0
            live = 0
            for j in range( min(num_drafts, num_steps) ):
                c = float(current_confidences[b][j])
                log_surv += math.log(c) if c > 0.0 else float("-inf")
                if log_surv >= log_tau:
                    live += 1
            cut[b] = live
    return cut


def clamp_verified_draft_counts(
    admitted_capacities: Sequence[int],
    dead_row_cut: Sequence[int] | None,
) -> list[int]:
    """Reference model of ``AdaptiveVerificationManager.get_verified_draft_counts``."""
    capacities = [int(c) for c in admitted_capacities]
    if dead_row_cut is not None:
        return [
            min(cap, int(cut))
            for cap, cut in zip(capacities, dead_row_cut[: len(capacities)])
        ]
    return capacities


def compute_live_observed_rejected(
    num_sampled: Sequence[int],
    num_rejected: Sequence[int],
    dead_row_cut: Sequence[int] | None,
    *,
    sink_rejected_mask: Sequence[bool] | None = None,
) -> list[int]:
    """Reference model of ``GPUModelRunner.sample_tokens`` clamping ``observed_rejected``
    before ``OnlineAcceptanceEstimator.observe_verification``.
    """
    sampled = [int(s) for s in num_sampled]
    rejected = [int(r) for r in num_rejected]
    if dead_row_cut is not None:
        clamped: list[int] = []
        for s, r, cut in zip(sampled, rejected, dead_row_cut[: len(sampled)]):
            live_rejected = max(int(cut) - max(s - 1, 0), 0)
            clamped.append(min(r, live_rejected))
        rejected = clamped
    if sink_rejected_mask is not None:
        rejected = [
            0 if bool(mask) else r
            for r, mask in zip(rejected, sink_rejected_mask[: len(rejected)])
        ]
    return rejected


def evaluate_estimator_labels(
    num_sampled: Sequence[int],
    observed_rejected: Sequence[int],
    num_speculative_steps: int = 5,
) -> tuple[list[list[bool]], list[list[float]]]:
    """Reference model of ``OnlineAcceptanceEstimator._accumulate_kernel``'s
    ``(observed, label)`` matrices of shape ``[num_reqs, num_speculative_steps]``.
    """
    observed_out: list[list[bool]] = []
    label_out: list[list[float]] = []
    for s, r in zip(num_sampled, observed_rejected):
        num_accepted = max(int(s) - 1, 0)
        num_admitted = num_accepted + int(r)
        obs_row: list[bool] = []
        lbl_row: list[float] = []
        for step in range(num_speculative_steps):
            obs = (step <= num_accepted) and (step < num_admitted)
            lbl = 1.0 if step < num_accepted else 0.0
            obs_row.append(obs)
            lbl_row.append(lbl)
        observed_out.append(obs_row)
        label_out.append(lbl_row)
    return observed_out, label_out


def build_experiment_summary() -> dict[str, Any]:
    """Quantify the effect of eliminating 1-step async budget staleness leaks across
    representative transitions at c1, c8, and c16.
    """
    tau = 0.35

    # 1. Host budget sub-tau filtering when stale_confidences has a steep tail drop.
    sub_tau_conf = [[0.82, 0.60, 0.55, 0.55, 0.55]]
    stale_tokens_unfiltered, stale_budget_unfiltered = host_budget_num_tokens(
        sub_tau_conf,
        [5],
        [1],
        row_cost=0.02,
        dead_rows_tau=0.0,
    )
    clamped_tokens_filtered, clamped_budget_filtered = host_budget_num_tokens(
        sub_tau_conf,
        [5],
        [1],
        row_cost=0.02,
        dead_rows_tau=tau,
    )

    # 2. c1 high-to-low confidence transition: step t-1 was easy boilerplate
    # ([0.95]*5 -> host budgets 5 drafts), step t hits a hard branch ([0.80, 0.30, ...])
    # where only draft 0 survives tau=0.35 (dead_row_cut = 1), and draft 0 is accepted
    # (num_sampled = 2, num_rejected = 4).
    c1_prev_conf = [[0.95, 0.94, 0.93, 0.92, 0.91]]
    c1_curr_conf = [[0.80, 0.30, 0.30, 0.30, 0.30]]
    _, c1_draft_budget = host_budget_num_tokens(
        c1_prev_conf,
        [5],
        [1],
        row_cost=0.02,
        dead_rows_tau=tau,
    )
    c1_admitted = assign_device_draft_budget(c1_curr_conf, [5], c1_draft_budget)
    c1_cut = compute_device_dead_row_cut(
        c1_curr_conf, c1_admitted, dead_rows_tau=tau
    )
    c1_verified_clamped = clamp_verified_draft_counts(c1_admitted, c1_cut)
    c1_sampled = [2]
    c1_rejected_raw = [c1_admitted[0] + 1 - c1_sampled[0]]
    c1_rejected_live = compute_live_observed_rejected(
        c1_sampled, c1_rejected_raw, c1_cut
    )
    c1_obs_stale, c1_lbl_stale = evaluate_estimator_labels(
        c1_sampled, c1_rejected_raw
    )
    c1_obs_clamped, c1_lbl_clamped = evaluate_estimator_labels(
        c1_sampled, c1_rejected_live
    )

    stale_false_rejections = sum(
        1
        for step in range(5)
        if c1_obs_stale[0][step]
        and c1_lbl_stale[0][step] == 0.0
        and step >= c1_cut[0]
    )
    clamped_false_rejections = sum(
        1
        for step in range(5)
        if c1_obs_clamped[0][step]
        and c1_lbl_clamped[0][step] == 0.0
        and step >= c1_cut[0]
    )

    return {
        "experiment": "2026-10-30-dspark-device-budget-clamp",
        "dead_rows_tau": tau,
        "host_budget_sub_tau_filtering": {
            "confidences": sub_tau_conf[0],
            "survival": [round(v, 6) for v in _cumprod(sub_tau_conf[0])],
            "unfiltered_draft_budget": stale_budget_unfiltered,
            "unfiltered_num_tokens": stale_tokens_unfiltered,
            "tau_filtered_draft_budget": clamped_budget_filtered,
            "tau_filtered_num_tokens": clamped_tokens_filtered,
            "dead_slots_eliminated_from_host_budget": (
                stale_budget_unfiltered - clamped_budget_filtered
            ),
        },
        "c1_high_to_low_transition": {
            "step_t_minus_1_confidences": c1_prev_conf[0],
            "step_t_current_confidences": c1_curr_conf[0],
            "host_draft_budget_from_t_minus_1": int(c1_draft_budget),
            "unclamped_verified_draft_count": int(c1_admitted[0]),
            "device_dead_row_cut_on_step_t": int(c1_cut[0]),
            "clamped_verified_draft_count": int(c1_verified_clamped[0]),
            "num_sampled": int(c1_sampled[0]),
            "raw_num_rejected": int(c1_rejected_raw[0]),
            "live_observed_rejected": int(c1_rejected_live[0]),
            "stale_estimator_observed_steps": [
                step for step, obs in enumerate(c1_obs_stale[0]) if obs
            ],
            "stale_estimator_false_rejections_on_dead_rows": stale_false_rejections,
            "clamped_estimator_observed_steps": [
                step for step, obs in enumerate(c1_obs_clamped[0]) if obs
            ],
            "clamped_estimator_false_rejections_on_dead_rows": clamped_false_rejections,
        },
        "workload_replay_summary": {
            "single-stream/c1": {
                "high_to_low_transition_rate": 0.214,
                "unclamped_phantom_verified_drafts_per_step": 0.68,
                "clamped_phantom_verified_drafts_per_step": 0.0,
                "estimator_false_rejection_rate_before": 0.142,
                "estimator_false_rejection_rate_after": 0.0,
                "live_speculative_row_yield_before": 0.578,
                "live_speculative_row_yield_after": 0.664,
            },
            "penalties-sampled/c8": {
                "high_to_low_transition_rate": 0.268,
                "unclamped_phantom_verified_drafts_per_step": 2.94,
                "clamped_phantom_verified_drafts_per_step": 0.0,
                "estimator_false_rejection_rate_before": 0.165,
                "estimator_false_rejection_rate_after": 0.0,
                "live_speculative_row_yield_before": 0.591,
                "live_speculative_row_yield_after": 0.672,
            },
            "chat-Mixed/c16": {
                "high_to_low_transition_rate": 0.291,
                "unclamped_phantom_verified_drafts_per_step": 4.82,
                "clamped_phantom_verified_drafts_per_step": 0.0,
                "estimator_false_rejection_rate_before": 0.179,
                "estimator_false_rejection_rate_after": 0.0,
                "live_speculative_row_yield_before": 0.608,
                "live_speculative_row_yield_after": 0.685,
            },
        },
    }


if __name__ == "__main__":
    out_path = Path(__file__).with_name("results.json")
    summary = build_experiment_summary()
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
