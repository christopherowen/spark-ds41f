#!/usr/bin/env python3
"""Reference online IRLS confidence-head calibrator and trace replay evaluator.

Mirrors the GPU-resident ``_calibrate_confidence_kernel``, ``_accumulate_kernel``
and ``_refit_kernel`` in ``vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py``
using only the Python standard library so host-side tests and offline replay
can run without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
from typing import Sequence

MAX_LOG_ODDS = 40.0
INIT_CONFIDENCE_HEAD_SLOPE = 1.0
INIT_CONFIDENCE_HEAD_BIAS = 0.0


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def logit(prob: float, max_log_odds: float = MAX_LOG_ODDS) -> float:
    p = clamp(float(prob), 1e-7, 1.0 - 1e-7)
    return clamp(math.log(p / (1.0 - p)), -max_log_odds, max_log_odds)


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(x, 80.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -80.0))
    return z / (1.0 + z)


def cumprod(probs: Sequence[float]) -> list[float]:
    out: list[float] = []
    prod = 1.0
    for p in probs:
        prod *= float(p)
        out.append(prod)
    return out


def conditional_from_survival(survival: Sequence[float]) -> list[float]:
    """Convert cumulative survival P(accept 0..k) into conditional P(accept k | 0..k-1)."""
    cond: list[float] = []
    prev = 1.0
    for s in survival:
        s_val = float(s)
        cond.append(clamp(s_val / prev, 1e-6, 1.0 - 1e-6) if prev > 1e-9 else 0.0)
        prev = s_val
    return cond


@dataclass
class OnlineConfidenceCalibrator:
    """Pure-Python reference for ``OnlineAcceptanceEstimator`` in confidence-head mode.

    Implements the exact arrowhead Newton-IRLS system solved on GPU by
    ``_calibrate_confidence_kernel``, ``_accumulate_kernel`` and ``_refit_kernel``:
      * Cold start: ``slope = 1.0``, ``intercepts = [0.0, ...]`` so
        ``sigma(slope * logit(p_head) + intercepts[k]) == p_head`` identically.
      * Each step: accumulates per-position normal equations ``(a_k, b_k, c_k)``
        and gradients ``(g0_k, g1_k)`` over graded positions ``step <= num_accepted``
        and ``step < num_admitted``.
      * Every ``refit_interval`` steps: solves the arrowhead system
        ``(X^T W X + L2 * I) delta_theta = X^T (Y - P)`` with sample-count damping
        ``n / (n + damping_observations)`` and trust-region clamping.
    """

    max_num_reqs: int
    num_speculative_steps: int = 5
    init_slope: float = INIT_CONFIDENCE_HEAD_SLOPE
    init_bias: float = INIT_CONFIDENCE_HEAD_BIAS
    refit_interval: int = 100
    damping_observations: float = 50.0
    l2: float = 1e-3
    max_step: float = 1.0
    max_slope: float = 4.0
    max_abs_intercept: float = 8.0

    slope: float = field(init=False)
    intercepts: list[float] = field(init=False)
    features: list[list[float]] = field(init=False)
    predictions: list[list[float]] = field(init=False)
    info: list[list[float]] = field(init=False)
    grad: list[list[float]] = field(init=False)
    counts: list[float] = field(init=False)
    steps_since_refit: int = field(default=0, init=False)
    refits: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.reset_for_confidence_head(
            init_slope=self.init_slope, init_bias=self.init_bias
        )

    def reset_for_confidence_head(
        self,
        init_slope: float = INIT_CONFIDENCE_HEAD_SLOPE,
        init_bias: float = INIT_CONFIDENCE_HEAD_BIAS,
    ) -> None:
        self.init_slope = float(init_slope)
        self.init_bias = float(init_bias)
        self.slope = self.init_slope
        self.intercepts = [self.init_bias for _ in range(self.num_speculative_steps)]
        self.features = [
            [0.0 for _ in range(self.num_speculative_steps)]
            for _ in range(self.max_num_reqs)
        ]
        self.predictions = [
            [0.0 for _ in range(self.num_speculative_steps)]
            for _ in range(self.max_num_reqs)
        ]
        self.info = [[0.0, 0.0, 0.0] for _ in range(self.num_speculative_steps)]
        self.grad = [[0.0, 0.0] for _ in range(self.num_speculative_steps)]
        self.counts = [0.0 for _ in range(self.num_speculative_steps)]
        self.steps_since_refit = 0
        self.refits = 0

    def calibrate_confidence_probs(
        self,
        confidence_probs: list[list[float]],
        idx_mapping: Sequence[int],
    ) -> list[list[float]]:
        """Calibrate raw confidence_head probabilities in place and store features/preds."""
        num_reqs = len(confidence_probs)
        if num_reqs == 0:
            return confidence_probs
        num_tokens = num_reqs * self.num_speculative_steps
        per_token = len(idx_mapping) == num_tokens and self.num_speculative_steps > 1

        for r in range(num_reqs):
            row = confidence_probs[r]
            for k in range(self.num_speculative_steps):
                feat = logit(row[k])
                prob = sigmoid(self.slope * feat + self.intercepts[k])
                row[k] = prob
                slot = (
                    int(idx_mapping[r * self.num_speculative_steps + k])
                    if per_token
                    else int(idx_mapping[r])
                )
                if slot >= 0:
                    self.features[slot][k] = feat
                    self.predictions[slot][k] = prob
        return confidence_probs

    def step(
        self,
        idx_mapping: Sequence[int],
        num_sampled: Sequence[int],
        num_rejected: Sequence[int],
    ) -> None:
        """Accumulate graded drafts and run the arrowhead Newton-IRLS refit every interval."""
        num_reqs = len(idx_mapping)
        for k in range(self.num_speculative_steps):
            a_k = b_k = c_k = g0_k = g1_k = count_k = 0.0
            for r in range(num_reqs):
                slot = int(idx_mapping[r])
                if slot < 0:
                    continue
                accepted = max(int(num_sampled[r]) - 1, 0)
                rejected = int(num_rejected[r])
                admitted = accepted + rejected
                if k <= accepted and k < admitted:
                    feat = self.features[slot][k]
                    pred = self.predictions[slot][k]
                    label = 1.0 if k < accepted else 0.0
                    w = pred * (1.0 - pred)
                    resid = label - pred
                    a_k += w * feat * feat
                    b_k += w * feat
                    c_k += w
                    g0_k += resid * feat
                    g1_k += resid
                    count_k += 1.0
            self.info[k][0] += a_k
            self.info[k][1] += b_k
            self.info[k][2] += c_k
            self.grad[k][0] += g0_k
            self.grad[k][1] += g1_k
            self.counts[k] += count_k

        self.steps_since_refit += 1
        if self.steps_since_refit < self.refit_interval:
            return
        self.steps_since_refit = 0
        self._refit()

    def _refit(self) -> None:
        K = self.num_speculative_steps
        a_k = [self.info[k][0] for k in range(K)]
        b_k = [self.info[k][1] for k in range(K)]
        c_k = [self.info[k][2] + self.l2 for k in range(K)]
        a = sum(a_k) + self.l2
        g0 = sum(self.grad[k][0] for k in range(K))
        g1_k = [self.grad[k][1] for k in range(K)]
        n_k = [self.counts[k] for k in range(K)]

        shrink = sum((b_k[k] * b_k[k]) / c_k[k] for k in range(K))
        coupling = sum((b_k[k] * g1_k[k]) / c_k[k] for k in range(K))
        denom = a - shrink
        step_w = (g0 - coupling) / denom if abs(denom) > 1e-12 else 0.0

        total_n = sum(n_k)
        if total_n > 0.0:
            step_w *= total_n / (total_n + self.damping_observations)
        else:
            step_w = 0.0

        if not math.isfinite(step_w) or (self.slope + step_w < 0.0):
            step_w = 0.0
        step_w = clamp(step_w, -self.max_step, self.max_step)
        new_w = min(self.slope + step_w, self.max_slope)

        new_intercepts: list[float] = []
        for k in range(K):
            if n_k[k] > 0.0:
                sb = ((g1_k[k] - b_k[k] * step_w) / c_k[k]) * (
                    n_k[k] / (n_k[k] + self.damping_observations)
                )
            else:
                sb = 0.0
            if not math.isfinite(sb):
                sb = 0.0
            sb = clamp(sb, -self.max_step, self.max_step)
            new_intercepts.append(
                clamp(
                    self.intercepts[k] + sb,
                    -self.max_abs_intercept,
                    self.max_abs_intercept,
                )
            )

        self.slope = new_w
        self.intercepts = new_intercepts
        for k in range(K):
            self.info[k] = [0.0, 0.0, 0.0]
            self.grad[k] = [0.0, 0.0]
            self.counts[k] = 0.0
        self.refits += 1


# Empirical static confidence_head means from 2026-09-26-dspark-policy (trained
# offline at temperature 0) vs empirical serving survival from 2026-10-29-copy-drafts-c12.
STATIC_HEAD_MEAN_PROBS: tuple[float, ...] = (0.78, 0.72, 0.71, 0.71, 0.73)

EMPIRICAL_WORKLOADS: dict[str, dict[str, object]] = {
    "single-stream/c1": {
        "num_reqs": 1,
        "accept_by_pos": [0.6545, 0.3548, 0.1569, 0.0720, 0.0445],
        "marginal_threshold": 0.1239,
    },
    "penalties-sampled/c8": {
        "num_reqs": 8,
        "accept_by_pos": [0.5877, 0.3125, 0.1286, 0.0457, 0.0235],
        "marginal_threshold": 0.2580,
    },
    "chat-Mixed/c16": {
        "num_reqs": 16,
        "accept_by_pos": [0.7393, 0.5204, 0.3144, 0.2202, 0.1726],
        "marginal_threshold": 0.4850,
    },
}


def simulate_online_calibration(
    workload_name: str,
    *,
    num_steps: int = 240,
    refit_interval: int = 20,
    seed: int = 20261030,
) -> dict[str, object]:
    """Replay online IRLS calibration on a workload with realistic per-step
    confidence spread and prefix-stopped target rejection sampling."""
    spec = EMPIRICAL_WORKLOADS[workload_name]
    target_survival = [float(x) for x in spec["accept_by_pos"]]  # type: ignore[index]
    target_cond = conditional_from_survival(target_survival)
    threshold = float(spec["marginal_threshold"])  # type: ignore[arg-type]

    # Use 16 concurrent request slots in replay so each refit round gathers realistic
    # graded observations while pricing the workload's marginal threshold.
    batch_reqs = max(int(spec["num_reqs"]), 16)  # type: ignore[arg-type]
    calibrator = OnlineConfidenceCalibrator(
        max_num_reqs=batch_reqs,
        num_speculative_steps=5,
        refit_interval=refit_interval,
    )
    rng = random.Random(seed)
    slots = list(range(batch_reqs))

    # True conditional logit offsets relative to static head logit(STATIC_HEAD_MEAN_PROBS).
    head_logits_mean = [logit(p) for p in STATIC_HEAD_MEAN_PROBS]
    target_logits_mean = [logit(p) for p in target_cond]
    true_slope = 0.92
    true_intercepts = [
        target_logits_mean[k] - true_slope * head_logits_mean[k] for k in range(5)
    ]

    for _ in range(num_steps):
        batch_probs: list[list[float]] = []
        batch_true_cond: list[list[float]] = []
        for _r in range(batch_reqs):
            # Shared request-level difficulty shift + small per-position noise.
            req_shift = rng.gauss(0.0, 0.35)
            row_probs: list[float] = []
            row_true: list[float] = []
            for k in range(5):
                z_head = head_logits_mean[k] + req_shift + rng.gauss(0.0, 0.15)
                row_probs.append(sigmoid(z_head))
                row_true.append(sigmoid(true_slope * z_head + true_intercepts[k]))
            batch_probs.append(row_probs)
            batch_true_cond.append(row_true)

        calibrator.calibrate_confidence_probs(batch_probs, slots)

        num_sampled: list[int] = []
        num_rejected: list[int] = []
        for r in range(batch_reqs):
            accepted = 0
            for k in range(5):
                if rng.random() < batch_true_cond[r][k]:
                    accepted += 1
                else:
                    break
            num_sampled.append(accepted + 1)
            num_rejected.append(5 - accepted)

        calibrator.step(slots, num_sampled, num_rejected)

    # Evaluate calibrated vs static survival at the mean head probabilities.
    eval_probs = [list(STATIC_HEAD_MEAN_PROBS)]
    calibrator.calibrate_confidence_probs(eval_probs, [0])
    calibrated_cond = eval_probs[0]
    calibrated_survival = cumprod(calibrated_cond)
    static_survival = cumprod(STATIC_HEAD_MEAN_PROBS)

    static_mae = sum(abs(static_survival[k] - target_survival[k]) for k in range(5)) / 5.0
    calibrated_mae = (
        sum(abs(calibrated_survival[k] - target_survival[k]) for k in range(5)) / 5.0
    )

    static_admitted = sum(1 for s in static_survival if s >= threshold)
    calibrated_admitted = sum(1 for s in calibrated_survival if s >= threshold)
    oracle_admitted = sum(1 for s in target_survival if s >= threshold)

    return {
        "workload": workload_name,
        "static_head_cond_probs": [round(x, 4) for x in STATIC_HEAD_MEAN_PROBS],
        "static_survival": [round(x, 4) for x in static_survival],
        "empirical_target_cond_probs": [round(x, 4) for x in target_cond],
        "empirical_target_survival": [round(x, 4) for x in target_survival],
        "calibrated_cond_probs": [round(x, 4) for x in calibrated_cond],
        "calibrated_survival": [round(x, 4) for x in calibrated_survival],
        "fitted_slope": round(calibrator.slope, 4),
        "fitted_intercepts": [round(b, 4) for b in calibrator.intercepts],
        "pos3_overconfidence_ratio_static": round(
            static_survival[3] / max(target_survival[3], 1e-6), 2
        ),
        "pos4_overconfidence_ratio_static": round(
            static_survival[4] / max(target_survival[4], 1e-6), 2
        ),
        "survival_mae_static": round(static_mae, 4),
        "survival_mae_calibrated": round(calibrated_mae, 4),
        "marginal_threshold": threshold,
        "admitted_depth_static": static_admitted,
        "admitted_depth_calibrated": calibrated_admitted,
        "admitted_depth_empirical_optimal": oracle_admitted,
    }


def evaluate_all() -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in EMPIRICAL_WORKLOADS:
        summary[name] = simulate_online_calibration(name)
    return summary


if __name__ == "__main__":
    results = evaluate_all()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
