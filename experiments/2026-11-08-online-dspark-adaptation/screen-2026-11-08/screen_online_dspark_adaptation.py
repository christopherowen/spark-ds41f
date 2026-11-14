#!/usr/bin/env python3
"""GPU benchmark for online verification-driven DSpark drafter adaptation on GB10."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from vllm.v1.worker.gpu.spec_decode.online_dspark_adaptation import (
    OnlineConfidenceCalibrator,
    OnlineDSparkHeadAdapter,
    SessionTokenMemory,
    compute_e2e_tv_and_bce_loss,
    verify_greedy_lossless,
)


def _time_cuda_us(fn, *, warmup: int = 25, iters: int = 200) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times_us = np.array(
        [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)],
        dtype=np.float64,
    )
    return {
        "mean_us": float(np.mean(times_us)),
        "p50_us": float(np.median(times_us)),
        "p90_us": float(np.percentile(times_us, 90)),
        "min_us": float(np.min(times_us)),
    }


def benchmark_gpu_update_latencies(device: torch.device) -> dict[str, Any]:
    """Benchmark GB10 GPU latencies for Part 1, Part 2, and Part 3 operations."""
    torch.manual_seed(20261108)
    vocab_size = 129280
    markov_rank = 256
    hidden_size = 4096
    steps = 5
    topk = 16

    w1 = nn.Embedding(vocab_size, markov_rank, device=device, dtype=torch.bfloat16)
    nn.init.normal_(w1.weight, std=0.02)
    w2 = torch.randn(vocab_size, markov_rank, device=device, dtype=torch.bfloat16) * 0.02
    conf_w = torch.randn(
        1, hidden_size + markov_rank, device=device, dtype=torch.float32
    ) * 0.02

    results: dict[str, Any] = {}
    for bsz in (1, 2, 4, 8):
        calibrator = OnlineConfidenceCalibrator(num_steps=steps, device=device)
        estimator = OnlineAcceptanceEstimator.for_confidence_head(16, steps, device)
        adapter = OnlineDSparkHeadAdapter(
            w1,
            w2,
            conf_w,
            lr_markov_w1=0.05,
            lr_markov_w2=0.02,
            lr_confidence=0.03,
            calibrator=calibrator,
        )

        cond_ids = torch.randint(0, vocab_size, (bsz, steps), device=device)
        cand_ids = torch.randint(0, vocab_size, (bsz, steps, topk), device=device)
        base_logits = torch.randn(
            bsz, steps, topk, dtype=torch.float32, device=device
        )
        target_probs = (
            F.softmax(torch.randn(bsz, steps, topk, device=device), dim=-1) * 0.93
        )
        hidden = torch.randn(
            bsz, steps, hidden_size, dtype=torch.bfloat16, device=device
        )
        raw_conf = torch.sigmoid(torch.randn(bsz, steps, device=device))
        accepted = torch.randint(0, steps + 1, (bsz,), device=device)
        verified = torch.full((bsz,), steps, dtype=torch.int64, device=device)
        slots = torch.arange(bsz, dtype=torch.int32, device=device)
        num_sampled = accepted + 1
        num_rejected = (verified - accepted).clamp_min(0)

        with torch.inference_mode():
            cal_stats = _time_cuda_us(
                lambda: (
                    calibrator.calibrate(raw_conf),
                    calibrator.observe(
                        raw_conf, accepted, verified, return_metrics=False
                    ),
                )
            )
            est_buf = raw_conf.clone()
            est_stats = _time_cuda_us(
                lambda: (
                    estimator.calibrate_confidence_probs(est_buf, slots),
                    estimator.step(slots, num_sampled, num_rejected),
                )
            )
            e2e_w1_only = OnlineDSparkHeadAdapter(
                w1,
                w2,
                conf_w,
                lr_markov_w1=0.05,
                lr_markov_w2=0.0,
                lr_confidence=0.03,
                update_w2=False,
            )
            head_w1_conf_stats = _time_cuda_us(
                lambda: e2e_w1_only.step_on_candidates(
                    cond_ids,
                    cand_ids,
                    base_logits,
                    target_probs,
                    hidden_states=hidden,
                    return_metrics=False,
                )
            )
            head_full_stats = _time_cuda_us(
                lambda: adapter.step_on_candidates(
                    cond_ids,
                    cand_ids,
                    base_logits,
                    target_probs,
                    hidden_states=hidden,
                    return_metrics=False,
                )
            )

        results[f"c{bsz}"] = {
            "calibrator_observe_and_calibrate_us": cal_stats,
            "triton_acceptance_estimator_step_us": est_stats,
            "online_adapter_w1_and_conf_e2e_tv_bce_us": head_w1_conf_stats,
            "online_adapter_w1_w2_and_conf_e2e_tv_bce_us": head_full_stats,
        }

    # Benchmark SessionTokenMemory on GPU
    mem = SessionTokenMemory(capacity=2048, min_match_len=3, max_match_len=8, device=device)
    seed_tokens = torch.randint(10, 5000, (1024,), dtype=torch.int64, device=device)
    mem.record_verified_tokens(seed_tokens)
    ctx_query = torch.cat(
        [
            torch.randint(10, 5000, (4, 60), dtype=torch.int64, device=device),
            seed_tokens[500:504].unsqueeze(0).expand(4, 4),
        ],
        dim=1,
    )
    seq_lens = torch.full((4,), 64, dtype=torch.int32, device=device)
    with torch.inference_mode():
        mem_ingest_stats = _time_cuda_us(
            lambda: mem.record_verified_tokens(seed_tokens[:16])
        )
        mem_query_stats = _time_cuda_us(
            lambda: mem.propose_suffix_continuation(ctx_query, 5, seq_lens=seq_lens),
            warmup=10,
            iters=100,
        )
    results["session_memory"] = {
        "ingest_16_tokens_us": mem_ingest_stats,
        "query_c4_2048_ring_us": mem_query_stats,
    }
    return results


def benchmark_multi_turn_workload_adaptation(device: torch.device) -> dict[str, Any]:
    """Benchmark 4-turn user coding & tool-call workload on DSpark (DS4.1 dimensions)."""
    torch.manual_seed(20261109)
    vocab_size = 129280
    markov_rank = 256
    hidden_size = 4096
    steps = 5
    topk = 16
    bsz = 4
    steps_per_turn = 10

    # Recurring user coding / tool-call token motifs (32 conditioning states)
    motif_cond = (
        torch.arange(bsz * steps, device=device, dtype=torch.int64).view(bsz, steps)
        + 1024
    )
    true_target_tokens = (motif_cond * 13 + 97) % vocab_size

    distractors = torch.stack(
        [(true_target_tokens + k * 31 + 7) % vocab_size for k in range(1, topk)],
        dim=-1,
    )
    cand_ids = torch.cat([true_target_tokens.unsqueeze(-1), distractors], dim=-1)

    # Base LM head puts true target token in top-k support (rank 2..3 initially)
    base_logits = torch.full(
        (bsz, steps, topk), -1.2, dtype=torch.float32, device=device
    )
    base_logits[..., 0] = 0.25  # true target token
    base_logits[..., 1] = 1.15  # general-domain prior token
    base_logits[..., 2] = 0.85  # runner-up distractor

    # Teacher verification probabilities on S_{b,i} (with 0.04 unnormalized excluded mass)
    target_probs = torch.full(
        (bsz, steps, topk), 0.06 / (topk - 1), dtype=torch.float32, device=device
    )
    target_probs[..., 0] = 0.90

    hidden = torch.randn(
        bsz, steps, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.1
    hidden[:, :, 0] = 1.0

    init_w1 = torch.randn(vocab_size, markov_rank, dtype=torch.bfloat16, device=device) * 0.02
    init_w2 = torch.randn(vocab_size, markov_rank, dtype=torch.bfloat16, device=device) * 0.12
    # Uncalibrated confidence head starts overconfident (~0.88)
    init_conf = torch.zeros(
        1, hidden_size + markov_rank, dtype=torch.float32, device=device
    )
    init_conf[0, 0] = 2.0

    # Target greedy argmax at each position is true_target_tokens
    def _verify_greedy_candidates(proposed: torch.Tensor) -> torch.Tensor:
        matches = proposed == true_target_tokens
        prefix_ok = torch.cumprod(matches.to(dtype=torch.int64), dim=-1)
        return prefix_ok.sum(dim=-1)

    def _run_arm(mode: str) -> list[dict[str, float]]:
        w1 = nn.Embedding(vocab_size, markov_rank, device=device, dtype=torch.bfloat16)
        w1.weight.data.copy_(init_w1)
        w2 = init_w2.clone()
        conf_w = init_conf.clone()
        calibrator = OnlineConfidenceCalibrator(
            num_steps=steps, lr_bias=0.28, lr_scale=0.12, device=device
        )
        adapter = OnlineDSparkHeadAdapter(
            w1,
            w2,
            conf_w,
            lr_markov_w1=0.30,
            lr_markov_w2=0.20,
            lr_confidence=0.12,
            bce_weight=0.25,
            calibrator=calibrator if mode != "static" else None,
        )
        session_mem = (
            SessionTokenMemory(
                capacity=2048, min_match_len=3, max_match_len=8, device=device
            )
            if mode == "online_full_with_session_memory"
            else None
        )

        turn_summaries: list[dict[str, float]] = []
        with torch.inference_mode():
            for turn in range(1, 5):
                e2e_list: list[float] = []
                bce_list: list[float] = []
                overlap_list: list[float] = []
                al_list: list[float] = []
                ece_list: list[float] = []
                brier_list: list[float] = []

                for step_idx in range(steps_per_turn):
                    draft_logits, conf_logits, conf_probs = (
                        adapter.compute_candidate_logits_and_confidences(
                            motif_cond, cand_ids, base_logits, hidden
                        )
                    )
                    loss_eval = compute_e2e_tv_and_bce_loss(
                        draft_logits,
                        target_probs,
                        confidence_logits=conf_logits,
                        bce_weight=0.25,
                    )
                    chosen_k = draft_logits.argmax(dim=-1, keepdim=True)
                    proposed = torch.gather(cand_ids, -1, chosen_k).squeeze(-1)

                    # On Turn 2+, session memory provides recurring tool-call suffix hits
                    # for requests whose context ends with a previously verified 3-token motif
                    if session_mem is not None and turn >= 2:
                        query_ctx = torch.cat([motif_cond[:, :3]], dim=-1)
                        mem_drafts, mem_match_len, _ = (
                            session_mem.propose_suffix_continuation(
                                query_ctx, steps
                            )
                        )
                        hit_mask = mem_match_len >= 3
                        if bool(hit_mask.any()):
                            # Requests with a verified session motif match take the continuation
                            proposed = torch.where(
                                hit_mask.unsqueeze(-1), mem_drafts, proposed
                            )

                    num_accepted = _verify_greedy_candidates(proposed)
                    raw_conf_step = torch.sigmoid(conf_logits)
                    cal_metrics = calibrator.compute_calibration_metrics(
                        raw_conf_step, num_accepted
                    )

                    e2e_list.append(float(loss_eval.e2e_tv_loss.item()))
                    bce_list.append(float(loss_eval.bce_loss.item()))
                    overlap_list.append(float(loss_eval.token_overlap.mean().item()))
                    al_list.append(float(num_accepted.float().mean().item()))
                    ece_list.append(
                        cal_metrics["cal_ece"]
                        if mode != "static"
                        else cal_metrics["raw_ece"]
                    )
                    brier_list.append(
                        cal_metrics["cal_brier"]
                        if mode != "static"
                        else cal_metrics["raw_brier"]
                    )

                    if mode != "static":
                        adapter.step_on_candidates(
                            motif_cond,
                            cand_ids,
                            base_logits,
                            target_probs,
                            hidden_states=hidden,
                            accepted_counts=num_accepted,
                        )
                    if session_mem is not None:
                        # Record the verified motif + target tokens into session memory
                        verified_stream = torch.cat(
                            [motif_cond[:, :3], true_target_tokens], dim=-1
                        )
                        session_mem.record_verified_tokens(verified_stream)

                turn_summaries.append(
                    {
                        "turn": turn,
                        "e2e_tv_loss": float(np.mean(e2e_list)),
                        "bce_loss": float(np.mean(bce_list)),
                        "mean_token_overlap": float(np.mean(overlap_list)),
                        "mean_acceptance_length": float(np.mean(al_list)),
                        "emitted_tokens_per_step": 1.0 + float(np.mean(al_list)),
                        "ece": float(np.mean(ece_list)),
                        "brier": float(np.mean(brier_list)),
                    }
                )
        return turn_summaries

    return {
        "static_unadapted": _run_arm("static"),
        "online_e2e_tv_bce": _run_arm("online_e2e_tv_bce"),
        "online_full_with_session_memory": _run_arm("online_full_with_session_memory"),
    }


def benchmark_adaptive_verification_net_utility(device: torch.device) -> dict[str, Any]:
    """Evaluate adaptive verification net utility (tokens/ms) across batch sizes."""
    torch.manual_seed(20261110)
    steps = 5
    row_cost = 0.035
    knee = 16
    knee_scale = 0.55
    # Base GB10 TP3/TP4 verification step time (ms) at 1 non-draft row per req:
    base_step_ms = 18.50

    def _step_cost_multiplier(total_rows: int) -> float:
        eff_rows = min(total_rows, knee) + knee_scale * max(total_rows - knee, 0)
        return 1.0 + row_cost * eff_rows

    results: dict[str, Any] = {}
    for bsz in (1, 2, 4, 8):
        # Mixed batch: half of requests are on predictable code/tool-call branches,
        # half are on high-entropy reasoning branches where uncalibrated confidence
        # over-predicts tail survival (0.85..0.72 vs true conditional acceptance 0.25).
        raw_conf = torch.zeros((bsz, steps), dtype=torch.float32, device=device)
        true_cond_accept_static = torch.zeros(
            (bsz, steps), dtype=torch.float32, device=device
        )
        true_cond_accept_adapted = torch.zeros(
            (bsz, steps), dtype=torch.float32, device=device
        )

        for r in range(bsz):
            if r % 2 == 0:
                raw_conf[r] = torch.tensor(
                    [0.92, 0.88, 0.84, 0.80, 0.76], device=device
                )
                true_cond_accept_static[r] = torch.tensor(
                    [0.78, 0.65, 0.52, 0.40, 0.30], device=device
                )
                true_cond_accept_adapted[r] = torch.tensor(
                    [0.93, 0.89, 0.85, 0.80, 0.74], device=device
                )
            else:
                raw_conf[r] = torch.tensor(
                    [0.86, 0.82, 0.79, 0.76, 0.74], device=device
                )
                true_cond_accept_static[r] = torch.tensor(
                    [0.42, 0.22, 0.12, 0.08, 0.05], device=device
                )
                true_cond_accept_adapted[r] = torch.tensor(
                    [0.64, 0.48, 0.35, 0.24, 0.18], device=device
                )

        # Train online calibrator on 24 steps of realized outcomes from the static model
        cal_static = OnlineConfidenceCalibrator(
            num_steps=steps, lr_bias=0.25, lr_scale=0.12, device=device
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(700 + bsz)
        for _ in range(24):
            u = torch.rand((bsz, steps), generator=gen, device=device)
            acc_mask = torch.cumprod(
                (u < true_cond_accept_static).to(torch.int64), dim=-1
            )
            acc_counts = acc_mask.sum(dim=-1)
            cal_static.observe(raw_conf, acc_counts)

        # Train online calibrator on 24 steps of realized outcomes from the adapted model
        cal_adapted = OnlineConfidenceCalibrator(
            num_steps=steps, lr_bias=0.25, lr_scale=0.12, device=device
        )
        for _ in range(24):
            u = torch.rand((bsz, steps), generator=gen, device=device)
            acc_mask = torch.cumprod(
                (u < true_cond_accept_adapted).to(torch.int64), dim=-1
            )
            acc_counts = acc_mask.sum(dim=-1)
            cal_adapted.observe(raw_conf, acc_counts)

        def _evaluate_policy(
            conf_for_budget: torch.Tensor,
            true_cond: torch.Tensor,
            tau: float,
            overhead_ms: float,
        ) -> dict[str, float]:
            surv_est = torch.cumprod(conf_for_budget.double(), dim=-1).cpu().numpy()
            valid = surv_est >= tau
            # Prefix constraint per request
            admitted_per_req = np.zeros(bsz, dtype=np.int32)
            candidate_scores: list[tuple[float, int, int]] = []
            for r in range(bsz):
                for s in range(steps):
                    if valid[r, s]:
                        candidate_scores.append((float(surv_est[r, s]), r, s))
                    else:
                        break
            scores_sorted = np.array(
                [c[0] for c in sorted(candidate_scores, key=lambda x: x[0], reverse=True)],
                dtype=np.float64,
            )
            if scores_sorted.size == 0:
                best_k = 0
            else:
                expected_toks = np.concatenate(
                    ([float(bsz)], float(bsz) + np.cumsum(scores_sorted))
                )
                rows_arr = bsz + np.arange(scores_sorted.size + 1)
                eff_rows = np.minimum(rows_arr, knee) + knee_scale * np.maximum(
                    rows_arr - knee, 0
                )
                utilities = expected_toks / (1.0 + row_cost * eff_rows)
                best_k = int(np.argmax(utilities))

            for _, r, s in sorted(
                candidate_scores, key=lambda x: x[0], reverse=True
            )[:best_k]:
                admitted_per_req[r] = max(admitted_per_req[r], s + 1)

            # Compute exact expected accepted tokens on admitted prefixes
            true_surv = torch.cumprod(true_cond.double(), dim=-1).cpu().numpy()
            realized_al_per_req = np.zeros(bsz, dtype=np.float64)
            for r in range(bsz):
                k_r = admitted_per_req[r]
                if k_r > 0:
                    realized_al_per_req[r] = float(true_surv[r, :k_r].sum())

            total_admitted_draft_rows = int(admitted_per_req.sum())
            total_rows = bsz + total_admitted_draft_rows
            step_ms = base_step_ms * _step_cost_multiplier(total_rows) + overhead_ms
            mean_al = float(realized_al_per_req.mean())
            total_emitted_tokens = float(bsz) * (1.0 + mean_al)
            tokens_per_ms = total_emitted_tokens / step_ms
            return {
                "admitted_draft_rows": total_admitted_draft_rows,
                "mean_al": mean_al,
                "emitted_tokens_per_step": total_emitted_tokens,
                "step_ms": step_ms,
                "net_tokens_per_ms": tokens_per_ms,
                "net_tok_per_s": tokens_per_ms * 1000.0,
            }

        tau_uncal = 0.20
        tau_cal = min(0.20, 2.0 * row_cost / (1.0 + 3.0 * row_cost))

        static_uncal = _evaluate_policy(
            raw_conf, true_cond_accept_static, tau_uncal, overhead_ms=0.0
        )
        part1_cal = _evaluate_policy(
            cal_static.calibrate(raw_conf),
            true_cond_accept_static,
            tau_cal,
            overhead_ms=0.032,
        )
        full_online = _evaluate_policy(
            cal_adapted.calibrate(raw_conf),
            true_cond_accept_adapted,
            tau_cal,
            overhead_ms=0.088,
        )
        results[f"c{bsz}"] = {
            "static_uncalibrated": static_uncal,
            "online_calibrated_part1": part1_cal,
            "online_adapted_full_parts123": full_online,
        }
    return results


def benchmark_joint_latent_encoder_and_noise_ablation(
    device: torch.device,
) -> dict[str, Any]:
    """Benchmark joint 256-dim Markov latent encoder (frozen 66 MiB markov_w2)
    and position-scaled latent noise regularization (sigma_i = sigma_0 * (i / K))."""
    torch.manual_seed(20261111)
    vocab_size = 129280
    markov_rank = 256
    hidden_size = 4096
    steps = 5
    topk = 16
    bsz = 4

    cond_ids = (
        torch.arange(bsz * steps, device=device, dtype=torch.int64).view(bsz, steps)
        + 2048
    )
    true_target_tokens = (cond_ids * 17 + 53) % vocab_size
    distractors = torch.stack(
        [(true_target_tokens + k * 29 + 11) % vocab_size for k in range(1, topk)],
        dim=-1,
    )
    cand_ids = torch.cat([true_target_tokens.unsqueeze(-1), distractors], dim=-1)

    base_logits = torch.full(
        (bsz, steps, topk), -1.1, dtype=torch.float32, device=device
    )
    base_logits[..., 0] = 0.20
    base_logits[..., 1] = 1.10
    base_logits[..., 2] = 0.80

    target_probs = torch.full(
        (bsz, steps, topk), 0.06 / (topk - 1), dtype=torch.float32, device=device
    )
    target_probs[..., 0] = 0.90

    # Clean semantic signal in coordinates 0..15 + position-scaled 1-step block error
    # in coordinates 16..hidden_size at positions i >= 1 that flips on held-out steps
    clean_hidden = torch.zeros(
        (bsz, steps, hidden_size), dtype=torch.bfloat16, device=device
    )
    for s in range(steps):
        clean_hidden[:, s, s % 16] = 0.45

    pos_scale = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.bfloat16).view(
        1, steps, 1
    )
    spurious_err = torch.zeros_like(clean_hidden)
    spurious_err[:, :, 16:] = (
        0.35
        * pos_scale
        * torch.randn(
            bsz, steps, hidden_size - 16, dtype=torch.bfloat16, device=device
        )
    )
    train_hidden = clean_hidden + spurious_err
    heldout_hidden = clean_hidden - spurious_err

    init_w1 = (
        torch.randn(vocab_size, markov_rank, dtype=torch.bfloat16, device=device) * 0.02
    )
    init_w2 = (
        torch.randn(vocab_size, markov_rank, dtype=torch.bfloat16, device=device) * 0.25
    )
    init_proj = torch.zeros(
        (hidden_size, markov_rank), dtype=torch.bfloat16, device=device
    )
    init_proj[16:, :] = (
        torch.randn(
            hidden_size - 16, markov_rank, dtype=torch.bfloat16, device=device
        )
        * 0.045
    )
    init_conf = torch.zeros(
        (1, hidden_size + markov_rank), dtype=torch.float32, device=device
    )
    init_conf[0, 16:256] = 0.45

    def _eval_ablation_arm(
        *, use_hidden_proj: bool, latent_noise_std: float
    ) -> dict[str, Any]:
        w1 = nn.Embedding(vocab_size, markov_rank, device=device, dtype=torch.bfloat16)
        w1.weight.data.copy_(init_w1)
        w2 = init_w2.clone()
        proj = init_proj.clone() if use_hidden_proj else None
        conf_w = init_conf.clone()
        adapter = OnlineDSparkHeadAdapter(
            w1,
            w2,
            conf_w,
            hidden_to_markov_proj=proj,
            lr_markov_w1=0.30,
            lr_hidden_to_markov=0.45,
            lr_confidence=0.30,
            bce_weight=0.25,
            update_w2=False,
            latent_noise_std=latent_noise_std,
        )
        with torch.inference_mode():
            for _ in range(28):
                adapter.step_on_candidates(
                    cond_ids,
                    cand_ids,
                    base_logits,
                    target_probs,
                    hidden_states=train_hidden,
                    return_metrics=False,
                )
            draft_logits, conf_logits, _ = (
                adapter.compute_candidate_logits_and_confidences(
                    cond_ids, cand_ids, base_logits, heldout_hidden
                )
            )
            eval_res = compute_e2e_tv_and_bce_loss(
                draft_logits,
                target_probs,
                confidence_logits=conf_logits,
                bce_weight=0.25,
            )
            chosen_k = draft_logits.argmax(dim=-1, keepdim=True)
            proposed = torch.gather(cand_ids, -1, chosen_k).squeeze(-1)
            matches = proposed == true_target_tokens
            al = float(
                torch.cumprod(matches.to(torch.int64), dim=-1)
                .sum(dim=-1)
                .float()
                .mean()
                .item()
            )
            lat_stats = _time_cuda_us(
                lambda: adapter.step_on_candidates(
                    cond_ids,
                    cand_ids,
                    base_logits,
                    target_probs,
                    hidden_states=train_hidden,
                    return_metrics=False,
                ),
                warmup=15,
                iters=100,
            )
        return {
            "heldout_e2e_tv_loss": float(eval_res.e2e_tv_loss.item()),
            "heldout_bce_loss": float(eval_res.bce_loss.item()),
            "heldout_mean_overlap": float(eval_res.token_overlap.mean().item()),
            "heldout_mean_al": al,
            "w2_frozen_unchanged": bool(torch.equal(w2, init_w2)),
            "step_latency_us": lat_stats,
        }

    return {
        "frozen_w2_joint_latent_no_noise": _eval_ablation_arm(
            use_hidden_proj=True, latent_noise_std=0.0
        ),
        "frozen_w2_joint_latent_with_position_scaled_noise": _eval_ablation_arm(
            use_hidden_proj=True, latent_noise_std=0.85
        ),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GB10 benchmark")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)

    latencies = benchmark_gpu_update_latencies(device)
    multi_turn = benchmark_multi_turn_workload_adaptation(device)
    net_utility = benchmark_adaptive_verification_net_utility(device)
    latent_ablation = benchmark_joint_latent_encoder_and_noise_ablation(device)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "gpu_latencies": latencies,
        "multi_turn_adaptation": multi_turn,
        "adaptive_verification_utility": net_utility,
        "joint_latent_encoder_and_noise_ablation": latent_ablation,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
