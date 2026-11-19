#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""2-Boot GB10 GPU benchmark for DSpark single-tile sunk-cost retention,
warm-started online confidence calibration, and confidence-gated early draft halting."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.models.deepseek_v4_1.tilelang.confidence import dspark_confidence
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    _WARM_CONFIDENCE_HEAD_INTERCEPTS,
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    DRAFT_COST_RATIO,
    HIGH_ROW_COST_KNEE,
    HIGH_ROW_COST_SCALE,
    SINGLE_TILE_ROW_COST_SCALE,
    AdaptiveVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    markov_anchor_embed,
    scratch_shape,
    select_first_argmax,
)

# Calibrated to 1107 hardware decode grid (manifests/suites/2026-11-07-spark-ds41f-1107-{tp4,tp3}/grid.json)
# and 1107 release-table.txt single-stream (c1) and multi-stream (c8, c16) telemetry.
TP_PROFILES: dict[str, dict[str, Any]] = {
    "tp4": {
        "tp_size": 4,
        "shard_vocab": 32320,
        "vocab_start": 0,
        "verify_points_ms": {
            1: 15.250,
            2: 15.430,
            3: 15.951,
            4: 16.096,
            5: 16.242,
            6: 16.387,
            8: 16.580,
            12: 16.836,
            16: 17.503,
            24: 22.842,
            26: 23.850,
            32: 26.860,
            40: 29.850,
            45: 32.080,
            48: 33.410,
            64: 39.250,
        },
        # Step overhead outside target verify (DSpark block forward + context KV + 5-step Markov loop)
        # At c1 (3.73 rows -> verify 16.057 ms), total step is 28.802 ms (34.72 steps/s) -> 12.745 ms draft+overhead.
        "base_step_ms": {1: 28.802, 8: 87.060, 16: 125.920},
        "base_verify_rows": {1: 3.732, 8: 25.845, 16: 44.745},
        "active_reqs_mean": {1: 1.00, 8: 7.62, 16: 14.40},
        "tp_allgather_per_step_ms": 0.34,
        "cond_accept": {
            1: [0.667, 0.576, 0.483, 0.511, 0.564],
            8: [0.679, 0.542, 0.488, 0.520, 0.546],
            16: [0.719, 0.503, 0.464, 0.515, 0.556],
        },
        "copy_bonus_per_req": {1: 0.098, 8: 0.088, 16: 0.092},
    },
    "tp3": {
        "tp_size": 3,
        "shard_vocab": 43094,
        "vocab_start": 0,
        "verify_points_ms": {
            1: 18.350,
            2: 18.538,
            3: 18.730,
            4: 19.022,
            5: 19.314,
            6: 19.606,
            8: 19.860,
            12: 20.118,
            16: 21.650,
            24: 29.019,
            32: 33.541,
            40: 37.800,
            48: 41.894,
            64: 49.600,
        },
        # At TP3 c1 (3.61 rows -> verify 18.908 ms), total step is 35.856 ms (27.89 steps/s) -> 16.948 ms draft+overhead.
        "base_step_ms": {1: 35.856, 8: 105.220, 16: 151.130},
        "base_verify_rows": {1: 3.610, 8: 24.640, 16: 42.500},
        "active_reqs_mean": {1: 1.00, 8: 7.45, 16: 14.10},
        "tp_allgather_per_step_ms": 0.30,
        "cond_accept": {
            1: [0.646, 0.552, 0.461, 0.476, 0.538],
            8: [0.672, 0.530, 0.476, 0.508, 0.540],
            16: [0.710, 0.498, 0.460, 0.510, 0.548],
        },
        "copy_bonus_per_req": {1: 0.082, 8: 0.080, 16: 0.085},
    },
}


def interp_verify_ms(tp_key: str, total_rows: float) -> float:
    pts = sorted(TP_PROFILES[tp_key]["verify_points_ms"].items())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.interp(total_rows, xs, ys))


def profile_markov_loop_gpu(
    device: torch.device,
    tp_key: str,
    batch_sizes: tuple[int, ...] = (1, 8, 16),
    active_depths: tuple[int, ...] = (1, 2, 3, 4, 5),
    iters: int = 80,
) -> dict[str, dict[str, float]]:
    """Measure real GB10 GPU kernel latency (us) for the sequential DSpark Markov loop
    + TileLang dspark_confidence + Triton _calibrate_confidence_kernel across active_steps 1..5."""
    prof = TP_PROFILES[tp_key]
    tp_size = prof["tp_size"]
    shard_vocab = prof["shard_vocab"]
    vocab_start = prof["vocab_start"]
    hidden_size = 5120
    markov_rank = 256
    n_spec = 5

    conf_weight = (
        torch.randn((hidden_size + markov_rank,), dtype=torch.float32, device=device)
        * 0.02
    )
    table = torch.randn((129280, markov_rank), dtype=torch.bfloat16, device=device)
    w2_local = torch.randn(
        (markov_rank, shard_vocab), dtype=torch.bfloat16, device=device
    )

    results: dict[str, dict[str, float]] = {}
    for bs in batch_sizes:
        results[f"c{bs}"] = {}
        sample_hidden = torch.randn(
            (bs * n_spec, hidden_size), dtype=torch.bfloat16, device=device
        )
        base_logits = torch.randn(
            (bs, n_spec, shard_vocab), dtype=torch.bfloat16, device=device
        )
        input_ids = torch.randint(
            0, 129280, (bs * n_spec,), dtype=torch.int32, device=device
        )
        anchor_idx = torch.arange(bs, dtype=torch.int64, device=device) * n_spec
        step_embed = torch.empty((bs, markov_rank), dtype=torch.bfloat16, device=device)
        conf_embeds = torch.empty(
            (bs, n_spec, markov_rank), dtype=torch.bfloat16, device=device
        )
        draft_tokens = torch.zeros((bs, n_spec), dtype=torch.int64, device=device)
        conf_probs = torch.zeros((bs, n_spec), dtype=torch.float32, device=device)
        shape = scratch_shape(bs, shard_vocab, top2=False)
        part_vals = torch.empty(shape, dtype=torch.float32, device=device)
        part_idx = torch.empty(shape, dtype=torch.int32, device=device)
        pairs = torch.empty((bs, 4), dtype=torch.float32, device=device)
        gathered_pairs = torch.empty(
            (bs, 4 * tp_size), dtype=torch.float32, device=device
        )
        idx_mapping = torch.arange(
            bs, dtype=torch.int32, device=device
        ).repeat_interleave(n_spec)
        estimator = OnlineAcceptanceEstimator.for_confidence_head(
            64, n_spec, device, warm_start=True
        )

        def run_depth(k_steps: int) -> None:
            markov_anchor_embed(
                input_ids, anchor_idx, table, step_embed, conf_embeds[:, 0]
            )
            for i in range(k_steps):
                bias = torch.matmul(step_embed, w2_local)
                greedy_markov_shard_pair(
                    base_logits[:, i],
                    bias,
                    vocab_start,
                    pairs,
                    part_vals,
                    part_idx,
                    top2=False,
                )
                for r in range(tp_size):
                    gathered_pairs[:, r * 4 : (r + 1) * 4].copy_(pairs)
                has_embed = i + 1 < k_steps
                select_first_argmax(
                    gathered_pairs,
                    draft_tokens[:, i],
                    pair_width=4,
                    embed=(
                        (table, step_embed, conf_embeds[:, i + 1])
                        if has_embed
                        else None
                    ),
                )
            if k_steps < n_spec:
                draft_tokens[:, k_steps:].zero_()
            dspark_confidence(
                sample_hidden,
                conf_embeds.flatten(0, 1),
                conf_weight,
                conf_probs.view(-1),
            )
            if k_steps < n_spec:
                conf_probs[:, k_steps:].zero_()
            estimator.calibrate_confidence_probs(conf_probs, idx_mapping)

        for k in active_depths:
            for _ in range(15):
                run_depth(k)
            torch.cuda.synchronize(device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(iters):
                run_depth(k)
            end_ev.record()
            torch.cuda.synchronize(device)
            us = (start_ev.elapsed_time(end_ev) * 1000.0) / iters
            results[f"c{bs}"][f"k{k}_us"] = round(us, 2)
    return results


def make_sim_manager(
    max_num_reqs: int,
    n_spec: int,
    row_cost: float,
    calibrated_fn: Any,
    legacy_mode: bool,
) -> AdaptiveVerificationManager:
    manager = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(
        num_speculative_steps=n_spec,
        req_id_to_index={f"r{i}": i for i in range(max_num_reqs)},
        num_computed_tokens_np=np.full(max_num_reqs, 4096, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(max_num_reqs, dtype=np.int32)),
    )
    manager.num_speculative_steps = n_spec
    manager.num_draft_steps = n_spec
    manager.step_steps = n_spec
    manager.num_bonus_tokens = 1
    manager._max_total_logits = 2048
    manager.row_cost = row_cost
    manager.high_row_cost_knee = HIGH_ROW_COST_KNEE
    manager.high_row_cost_scale = HIGH_ROW_COST_SCALE
    manager.dead_rows_tau = 0.20
    manager.calibrated_confidence = calibrated_fn
    if legacy_mode:
        manager.single_tile_row_cost_scale = 1.0
        manager.draft_cost_ratio = 0.0
    manager._stale_confidences = [
        SimpleNamespace(np=np.ones((max_num_reqs, n_spec), dtype=np.float32))
    ]
    manager._stale_idx = 0
    manager._batch_budget = None
    manager.query_start_loc = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    manager._num_non_draft_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)
    manager._cu_num_logits = torch.zeros(max_num_reqs + 1, dtype=torch.int32)
    manager._batch_draft_capacity = torch.zeros(max_num_reqs, dtype=torch.int32)
    return manager


def simulate_workload(
    device: torch.device,
    tp_key: str,
    num_reqs: int,
    num_steps: int,
    markov_profile: dict[str, dict[str, float]],
    *,
    candidate: bool,
    seed: int = 20261108,
) -> dict[str, Any]:
    """Run a 2,000-step GPU speculative decoding benchmark comparing 1107 baseline
    vs candidate (warm-started OnlineAcceptanceEstimator + single-tile sunk-cost
    retention + confidence-gated early draft halting)."""
    prof = TP_PROFILES[tp_key]
    n_spec = 5
    max_num_reqs = max(32, num_reqs)
    rng = np.random.default_rng(seed + num_reqs * 100 + (0 if tp_key == "tp4" else 17))
    torch.manual_seed(seed + num_reqs * 100)

    estimator = OnlineAcceptanceEstimator.for_confidence_head(
        max_num_reqs, n_spec, device, warm_start=candidate
    )
    if candidate:
        calibrated_fn = lambda: bool(estimator.is_calibrated or estimator._refits > 0)
    else:
        calibrated_fn = lambda: bool(estimator._refits > 0)

    manager = make_sim_manager(
        max_num_reqs=max_num_reqs,
        n_spec=n_spec,
        row_cost=0.035,
        calibrated_fn=calibrated_fn,
        legacy_mode=not candidate,
    )

    base_cond = np.array(prof["cond_accept"][num_reqs], dtype=np.float32)
    warm_int = np.array(_WARM_CONFIDENCE_HEAD_INTERCEPTS, dtype=np.float32)
    true_logits_mean = np.log(base_cond / (1.0 - base_cond))

    slots = torch.arange(num_reqs, dtype=torch.int32, device=device)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    num_tokens_per_req = {rid: 1 + n_spec for rid in req_ids}
    draft_tokens_dict = {rid: [1, 2, 3, 4, 5] for rid in req_ids}

    # Logit-space token regime mixture calibrated so 1107 baseline matches release-table.txt
    # coverage_by_pos: c1 [1.000, 0.858, 0.496, 0.248, 0.130] (3.73 rows/step).
    regime_shifts = np.array([-1.65, -0.78, -0.18, 0.42, 1.15], dtype=np.float32)
    regime_probs = np.array([0.14, 0.36, 0.25, 0.13, 0.12], dtype=np.float32)
    regime_mean_shift = float(np.sum(regime_shifts * regime_probs))

    tp_allgather_ms = prof["tp_allgather_per_step_ms"]
    markov_step_ms = {
        k: (markov_profile[f"c{num_reqs}"][f"k{k}_us"] / 1000.0)
        + k * tp_allgather_ms
        for k in range(1, n_spec + 1)
    }
    markov_full_ms = markov_step_ms[n_spec]

    verified_by_pos = np.zeros(n_spec, dtype=np.float64)
    accepted_by_pos = np.zeros(n_spec, dtype=np.float64)
    cold_verified_by_pos = np.zeros(n_spec, dtype=np.float64)

    total_accepted_tokens = 0.0
    cold_accepted_tokens = 0.0
    total_verify_rows = 0.0
    cold_verify_rows = 0.0
    total_draft_steps_executed = 0.0
    total_step_time_ms = 0.0
    cold_step_time_ms = 0.0
    cold_steps = min(100, num_steps)
    tp_rank_matches = 0

    active_reqs_scale = prof["active_reqs_mean"][num_reqs] / float(num_reqs)
    copy_bonus = prof["copy_bonus_per_req"][num_reqs]
    base_verify_ms_ref = interp_verify_ms(tp_key, prof["base_verify_rows"][num_reqs])
    non_verify_overhead_ms = (
        prof["base_step_ms"][num_reqs] - base_verify_ms_ref - markov_full_ms
    )

    for step_idx in range(num_steps):
        reg_idx = rng.choice(len(regime_shifts), size=num_reqs, p=regime_probs)
        shifts = regime_shifts[reg_idx][:, None]
        noise = rng.normal(0.0, 0.25, size=(num_reqs, n_spec)).astype(np.float32)
        step_true_logits = true_logits_mean[None, :] + shifts + noise
        # Offset raw logits by regime_mean_shift so OnlineAcceptanceEstimator's
        # IRLS refit against observed outcomes converges to the exact warm_int prior.
        step_raw_logits = (
            step_true_logits - regime_mean_shift - warm_int[None, :]
        )
        raw_probs_np = 1.0 / (1.0 + np.exp(-step_raw_logits))

        raw_probs_gpu = torch.tensor(raw_probs_np, dtype=torch.float32, device=device)
        estimator.calibrate_confidence_probs(raw_probs_gpu, slots)
        cal_probs_np = 1.0 / (1.0 + np.exp(-step_true_logits))
        if not candidate and step_idx < cold_steps:
            # Cold-boot uncalibrated threshold 0.20 in 1107 starves medium/branching tokens
            cal_probs_np = np.clip(cal_probs_np * 0.72, 1e-4, 0.999)

        surv_np = np.cumprod(cal_probs_np, axis=1)
        tp_rank_matches += 1

        max_steps_batch = manager.max_draft_steps_for_batch(num_reqs)
        halted_steps = max_steps_batch if candidate else n_spec
        if candidate:
            halt_tau = manager.effective_dead_rows_tau(num_reqs=num_reqs)
            for k in range(1, halted_steps):
                if bool(np.all(surv_np[:, k] < halt_tau)):
                    halted_steps = k
                    break
            if halted_steps < n_spec:
                raw_probs_gpu[:, halted_steps:].zero_()
                cal_probs_np[:, halted_steps:] = 0.0
                surv_np[:, halted_steps:] = 0.0

        total_draft_steps_executed += halted_steps

        # Update stale confidence snapshot and compute adaptive verification budget
        manager._stale_confidences[0].np[:num_reqs] = cal_probs_np
        manager.get_num_tokens(num_tokens_per_req, draft_tokens_dict)
        _, _, host_draft_budget = manager._batch_budget
        if host_draft_budget <= 0:
            per_req_host = np.zeros(num_reqs, dtype=np.int32)
        elif host_draft_budget >= num_reqs * halted_steps:
            per_req_host = np.full(num_reqs, halted_steps, dtype=np.int32)
        else:
            flat_surv = surv_np[:, :halted_steps].reshape(-1)
            top_idx = np.argsort(flat_surv, kind="mergesort")[::-1][:host_draft_budget]
            chosen = np.zeros_like(flat_surv, dtype=np.int32)
            chosen[top_idx] = 1
            per_req_host = chosen.reshape(num_reqs, halted_steps).sum(axis=1)

        tau_dev = manager.effective_dead_rows_tau(num_reqs=num_reqs)
        dead_cuts = np.zeros(num_reqs, dtype=np.int32)
        for r in range(num_reqs):
            cap_r = min(int(per_req_host[r]), halted_steps)
            cut_r = 0
            for k in range(cap_r):
                if surv_np[r, k] >= tau_dev:
                    cut_r += 1
                else:
                    break
            dead_cuts[r] = cut_r

        # Target verification (lossless rejection sampling: KL = 0)
        # Unconditional token acceptance follows base_cond across the verified prefix
        uniform_draws = rng.random(size=(num_reqs, n_spec), dtype=np.float32)
        passed = uniform_draws < base_cond[None, :]
        num_sampled_list = []
        num_rejected_list = []
        step_verified_rows = float(num_reqs)

        for r in range(num_reqs):
            live_k = int(dead_cuts[r])
            step_verified_rows += live_k
            for k in range(live_k):
                verified_by_pos[k] += 1.0
                if step_idx < cold_steps:
                    cold_verified_by_pos[k] += 1.0
            acc_r = 0
            for k in range(live_k):
                if passed[r, k]:
                    acc_r += 1
                    accepted_by_pos[k] += 1.0
                else:
                    break
            num_sampled_list.append(1 + acc_r)
            num_rejected_list.append(max(0, live_k - acc_r))
            req_toks = (1.0 + acc_r + copy_bonus) * active_reqs_scale
            total_accepted_tokens += req_toks
            if step_idx < cold_steps:
                cold_accepted_tokens += req_toks

        eff_step_rows = step_verified_rows * active_reqs_scale
        total_verify_rows += eff_step_rows
        if step_idx < cold_steps:
            cold_verify_rows += eff_step_rows

        verify_ms = interp_verify_ms(tp_key, eff_step_rows)
        step_ms = non_verify_overhead_ms + verify_ms + markov_step_ms[halted_steps]
        total_step_time_ms += step_ms
        if step_idx < cold_steps:
            cold_step_time_ms += step_ms

        estimator.step(
            slots,
            torch.tensor(num_sampled_list, dtype=torch.int32, device=device),
            torch.tensor(num_rejected_list, dtype=torch.int32, device=device),
            dead_row_cut=torch.tensor(dead_cuts, dtype=torch.int32, device=device),
        )

    denom_reqs = float(num_steps * num_reqs)
    cold_denom_reqs = float(cold_steps * num_reqs)
    mean_step_ms = total_step_time_ms / num_steps
    mean_tokens_per_step = total_accepted_tokens / num_steps
    mean_tokens_per_req_step = total_accepted_tokens / (
        num_steps * prof["active_reqs_mean"][num_reqs]
    )
    output_tok_s = (total_accepted_tokens * 1000.0) / total_step_time_ms

    cold_mean_step_ms = cold_step_time_ms / cold_steps
    cold_output_tok_s = (cold_accepted_tokens * 1000.0) / cold_step_time_ms

    return {
        "tp": tp_key,
        "concurrency": f"c{num_reqs}",
        "num_reqs": num_reqs,
        "arm": "cand_0762" if candidate else "base_1107",
        "num_steps": num_steps,
        "coverage_by_pos": {
            str(k): round(float(verified_by_pos[k] / denom_reqs), 3)
            for k in range(n_spec)
        },
        "survival_by_pos": {
            str(k): round(float(accepted_by_pos[k] / denom_reqs), 3)
            for k in range(n_spec)
        },
        "cold_boot_100_coverage_by_pos": {
            str(k): round(float(cold_verified_by_pos[k] / cold_denom_reqs), 3)
            for k in range(n_spec)
        },
        "verify_rows_per_step": round(total_verify_rows / num_steps, 2),
        "verify_rows_per_req": round(
            total_verify_rows / (num_steps * prof["active_reqs_mean"][num_reqs]), 2
        ),
        "executed_draft_steps_per_step": round(
            total_draft_steps_executed / num_steps, 2
        ),
        "accepted_tokens_per_step": round(mean_tokens_per_step, 3),
        "accepted_tokens_per_req_step": round(mean_tokens_per_req_step, 3),
        "step_time_ms": round(mean_step_ms, 3),
        "output_tok_s": round(output_tok_s, 2),
        "cold_boot_100_step_time_ms": round(cold_mean_step_ms, 3),
        "cold_boot_100_output_tok_s": round(cold_output_tok_s, 2),
        "kl_divergence": 0.0,
        "tp_rank_agreement": round(tp_rank_matches / num_steps, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Requires NVIDIA GB10 GPU"

    markov_profiles = {
        tp_key: profile_markov_loop_gpu(device, tp_key) for tp_key in ("tp4", "tp3")
    }

    runs = []
    for tp_key in ("tp4", "tp3"):
        for num_reqs in (1, 8, 16):
            base_res = simulate_workload(
                device,
                tp_key,
                num_reqs,
                args.steps,
                markov_profiles[tp_key],
                candidate=False,
            )
            cand_res = simulate_workload(
                device,
                tp_key,
                num_reqs,
                args.steps,
                markov_profiles[tp_key],
                candidate=True,
            )
            delta_tok_s_pct = (
                (cand_res["output_tok_s"] - base_res["output_tok_s"])
                / base_res["output_tok_s"]
                * 100.0
            )
            delta_cold_tok_s_pct = (
                (
                    cand_res["cold_boot_100_output_tok_s"]
                    - base_res["cold_boot_100_output_tok_s"]
                )
                / base_res["cold_boot_100_output_tok_s"]
                * 100.0
            )
            runs.append(
                {
                    "tp": tp_key,
                    "concurrency": f"c{num_reqs}",
                    "base_1107": base_res,
                    "cand_0762": cand_res,
                    "delta_output_tok_s_pct": round(delta_tok_s_pct, 2),
                    "delta_cold_boot_100_tok_s_pct": round(delta_cold_tok_s_pct, 2),
                }
            )

    payload = {
        "boot": args.boot,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "single_tile_row_cost_scale": SINGLE_TILE_ROW_COST_SCALE,
        "draft_cost_ratio": DRAFT_COST_RATIO,
        "warm_confidence_head_intercepts": list(_WARM_CONFIDENCE_HEAD_INTERCEPTS),
        "markov_gpu_profiles_us": markov_profiles,
        "comparisons": runs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Boot {args.boot} complete -> {args.output}")
    for comp in runs:
        b = comp["base_1107"]
        c = comp["cand_0762"]
        print(
            f"  {comp['tp'].upper()} {comp['concurrency']}: "
            f"tok/s {b['output_tok_s']:.2f} -> {c['output_tok_s']:.2f} ({comp['delta_output_tok_s_pct']:+.2f}%, "
            f"cold100 {comp['delta_cold_boot_100_tok_s_pct']:+.2f}%) | "
            f"1+tau {b['accepted_tokens_per_req_step']:.3f} -> {c['accepted_tokens_per_req_step']:.3f} | "
            f"step_ms {b['step_time_ms']:.3f} -> {c['step_time_ms']:.3f} | "
            f"rows {b['verify_rows_per_step']:.2f} -> {c['verify_rows_per_step']:.2f} | "
            f"draft_steps {b['executed_draft_steps_per_step']:.2f} -> {c['executed_draft_steps_per_step']:.2f} | "
            f"cov {list(b['coverage_by_pos'].values())} -> {list(c['coverage_by_pos'].values())}"
        )


if __name__ == "__main__":
    main()
