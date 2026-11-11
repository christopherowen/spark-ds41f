#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for GPUModelRunner per-step decode fastpath on GB10 (SM121).

Measures both baseline (spark-ds41f:1107) and candidate across concurrency
levels c1, c8, and c16:
  1. add_requests + update_requests + block_tables.apply_staged_writes on a
     steady-state decode step (no new requests, no new KV block allocations).
  2. prepare_inputs H2D / offset preparation on an adaptive-verification
     decode step ( skipping redundant cu_num_logits and query_start_loc H2D
     transfers overwritten by adaptive_verification.reallocate_drafts).
  3. sample_tokens post-sample critical path before speculator.propose
     (fusing dead_row_cut clamping and CopyDrafter slot masking into
     OnlineAcceptanceEstimator._accumulate_kernel, reusing
     num_verified_draft_tokens, and skipping 0-element idx_mapping tail fill).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    OnlineAcceptanceEstimator,
    _accumulate_kernel,
)
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)
from vllm.triton_utils import triton


MAX_NUM_REQS = 64
NUM_SPEC_STEPS = 5
VOCAB_SIZE = 151936


class _BaselineSamplerStateStub:
    """Mimics the 15 UvaBackedTensors + 7 StagedWriteTensors rotated by
    Sampler.apply_staged_writes() when called unconditionally on decode steps."""

    def __init__(self, max_num_reqs: int, device: torch.device) -> None:
        self.uva_tensors = [
            UvaBackedTensor(max_num_reqs, dtype=torch.float32) for _ in range(7)
        ] + [
            UvaBackedTensor(max_num_reqs, dtype=torch.int32) for _ in range(7)
        ] + [
            UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        ]
        self.staged_tensors = [
            StagedWriteTensor((max_num_reqs, 16), dtype=torch.int32, device=device)
            for _ in range(7)
        ]

    def apply_staged_writes(self) -> None:
        for t in self.uva_tensors:
            t.copy_to_uva()
        for s in self.staged_tensors:
            s.apply_write()


def _make_block_tables(device: torch.device, dirty_guard: bool) -> BlockTables:
    bt = BlockTables(
        block_sizes=[256, 512],
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=4096,
        max_num_blocks_per_group=[64, 32],
        device=device,
        kernel_block_sizes=[64, 512],
    )
    for r in range(16):
        bt.append_block_ids(r, ([r * 2 + 1], [r + 1]), overwrite=True)
    bt.apply_staged_writes()
    torch.cuda.synchronize(device)
    bt._use_dirty_guard = dirty_guard
    return bt


def _make_adaptive_verification(
    device: torch.device,
) -> tuple[AdaptiveVerificationManager, torch.Tensor]:
    query_start_loc = torch.zeros(MAX_NUM_REQS + 1, dtype=torch.int32, device=device)
    req_states = SimpleNamespace(
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=4096,
        num_speculative_steps=NUM_SPEC_STEPS,
        device=device,
        req_id_to_index={f"req-{i}": i for i in range(MAX_NUM_REQS)},
        num_computed_tokens_np=np.full(MAX_NUM_REQS, 128, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.full(MAX_NUM_REQS, 32, dtype=np.int32)),
        expanded_idx_mapping=torch.zeros(4096, dtype=torch.int32, device=device),
        expanded_local_pos=torch.zeros(4096, dtype=torch.int32, device=device),
    )
    manager = AdaptiveVerificationManager(
        req_states=req_states,
        query_start_loc=query_start_loc,
        num_bonus_tokens=1,
        max_total_logits=4096,
        row_cost=0.02,
        planned_confidences=True,
        num_draft_steps=NUM_SPEC_STEPS,
        tp_invariant_confidence=True,
        calibrated_confidence=True,
    )
    manager.dead_rows_tau = 0.2
    manager._confidence_probs.fill_(0.82)
    manager._planned_probs.fill_(0.82)
    return manager, query_start_loc


def _run_staged_writes_step(
    mode: str,
    num_reqs: int,
    sampler_stub: _BaselineSamplerStateStub,
    bt: BlockTables,
) -> None:
    scheduled_new_reqs = []
    if mode == "baseline":
        sampler_stub.apply_staged_writes()
        for r in range(num_reqs):
            for i in range(bt.num_kv_cache_groups):
                start = bt.num_blocks.np[i, r]
                bt.block_tables[i].stage_write(r, start, [])
                bt.num_blocks.np[i, r] = start
        assert bt.fused_writer is not None
        bt.fused_writer.apply(
            bt.block_tables, bt._block_table_ptrs_i64, bt.block_table_strides
        )
        bt.num_blocks.copy_to_uva()
    else:
        if scheduled_new_reqs:
            sampler_stub.apply_staged_writes()
        for r in range(num_reqs):
            bt.append_block_ids(r, ([], []), overwrite=False)
        bt.apply_staged_writes()


def _run_prepare_inputs_step(
    mode: str,
    num_reqs: int,
    req_ids: list[str],
    num_tokens_per_req: dict[str, int],
    draft_tokens: dict[str, list[int]],
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens_orig: np.ndarray,
    manager: AdaptiveVerificationManager,
    query_start_loc_buf: torch.Tensor,
    device: torch.device,
):
    num_tokens = manager.get_num_tokens(num_tokens_per_req, draft_tokens)
    idx_mapping = async_tensor_h2d(idx_mapping_np, device=device)
    num_draft_tokens_per_req = np.full(num_reqs, NUM_SPEC_STEPS, dtype=np.int32)
    num_bonus_tokens = 1
    num_logits = num_draft_tokens_per_req + num_bonus_tokens
    cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
    cu_num_logits_np[0] = 0
    np.cumsum(num_logits, out=cu_num_logits_np[1:])
    if mode == "baseline":
        cu_num_logits = async_tensor_h2d(cu_num_logits_np, device=device)

    num_scheduled_tokens_np, cu_num_logits_np = manager.compact_batch(
        num_draft_tokens_per_req,
        num_scheduled_tokens_orig,
        cu_num_logits_np,
    )
    num_reqs_padded = num_reqs
    if mode == "baseline":
        query_start_loc_np = np.empty(MAX_NUM_REQS + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_tensor_h2d(query_start_loc_np, out=query_start_loc_buf)
    else:
        query_start_loc_np = np.empty(num_reqs_padded + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_np, out=query_start_loc_np[1 : num_reqs + 1])
        query_start_loc_np[num_reqs + 1 :] = num_tokens

    (
        cu_num_logits,
        query_start_loc,
        total_num_draft_tokens,
        expanded_idx_mapping,
        expanded_local_pos,
    ) = manager.reallocate_drafts(req_ids, idx_mapping)
    query_start_loc_np = query_start_loc_np[: num_reqs_padded + 1]
    query_start_loc = query_start_loc[: num_reqs_padded + 1]
    return (
        cu_num_logits,
        query_start_loc,
        query_start_loc_np,
        total_num_draft_tokens,
        expanded_idx_mapping,
        expanded_local_pos,
    )


def _run_post_sample_step(
    mode: str,
    num_reqs: int,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    dead_row_cut: torch.Tensor,
    copy_int_state: torch.Tensor,
    estimator: OnlineAcceptanceEstimator,
    manager: AdaptiveVerificationManager,
    spec_idx_mapping_buf: torch.Tensor,
    max_num_reqs: int,
) -> torch.Tensor:
    # 1. First get_verified_draft_counts (at AsyncOutput creation)
    num_verified_draft_tokens = manager.get_verified_draft_counts(
        num_reqs, dead_row_cut
    )

    # 2. observe_verification
    estimator._steps_since_refit = 0
    if mode == "baseline":
        live_rejected = (
            dead_row_cut[:num_reqs] - (num_sampled - 1).clamp_min(0)
        ).clamp_min(0)
        observed_rejected = torch.minimum(num_rejected, live_rejected)
        observed_idx_mapping = torch.where(
            copy_int_state[idx_mapping, 1] > 0,
            -1,
            idx_mapping,
        )
        # Baseline OnlineAcceptanceEstimator.step also repeated the dead_row_cut clamp
        num_accepted = (num_sampled - 1).clamp_min(0)
        live_rejected_2 = (dead_row_cut[:num_reqs] - num_accepted).clamp_min(0)
        observed_rejected = torch.minimum(observed_rejected, live_rejected_2)
        _accumulate_kernel[(estimator.num_speculative_steps,)](
            estimator.info,
            estimator.grad,
            observed_idx_mapping,
            num_sampled,
            observed_rejected,
             observed_rejected,
            observed_rejected,
            0,
            estimator.features,
            estimator.features.stride(0),
            estimator.predictions,
            estimator.predictions.stride(0),
            estimator.counts,
            num_reqs,
            HAS_DEAD_ROW_CUT=False,
            HAS_COPY_STATE=False,
            BLOCK_R=triton.next_power_of_2(max(num_reqs, 1)),
        )
        # 3. Second get_verified_draft_counts before copy_drafter.prepare_alloc
        verified = manager.get_verified_draft_counts(num_reqs, dead_row_cut)
        # 4. _copy_request_inputs tail fill
        spec_idx_mapping_buf[:num_reqs].copy_(idx_mapping)
        spec_idx_mapping_buf[num_reqs:].fill_(-1)
    else:
        estimator.step(
            idx_mapping,
            num_sampled,
            num_rejected,
            dead_row_cut=dead_row_cut,
            copy_int_state=copy_int_state,
        )
        verified = num_verified_draft_tokens
        spec_idx_mapping_buf[:num_reqs].copy_(idx_mapping)
        if num_reqs < max_num_reqs:
            spec_idx_mapping_buf[num_reqs:].fill_(-1)
    return verified


def benchmark_concurrency(
    num_reqs: int, warmup: int, iters: int, device: torch.device
) -> dict:
    torch.manual_seed(20261108 + num_reqs)
    np.random.seed(20261108 + num_reqs)

    sampler_base = _BaselineSamplerStateStub(MAX_NUM_REQS, device)
    sampler_cand = _BaselineSamplerStateStub(MAX_NUM_REQS, device)
    bt_base = _make_block_tables(device, dirty_guard=False)
    bt_cand = _make_block_tables(device, dirty_guard=True)

    av_base, qsl_base = _make_adaptive_verification(device)
    av_cand, qsl_cand = _make_adaptive_verification(device)

    req_ids = [f"req-{i}" for i in range(num_reqs)]
    num_tokens_per_req = {r: 1 + NUM_SPEC_STEPS for r in req_ids}
    draft_tokens = {r: [10, 11, 12, 13, 14] for r in req_ids}
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    num_scheduled_tokens_orig = np.full(num_reqs, 1 + NUM_SPEC_STEPS, dtype=np.int32)

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    num_sampled = torch.randint(1, NUM_SPEC_STEPS + 1, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = (NUM_SPEC_STEPS + 1 - num_sampled).clamp_min(0).to(torch.int32)
    dead_row_cut = torch.randint(1, NUM_SPEC_STEPS + 1, ( MAX_NUM_REQS,), dtype=torch.int32, device=device)
    copy_int_state = torch.zeros((MAX_NUM_REQS, 2), dtype=torch.int32, device=device)
    copy_int_state[::3, 1] = 8

    est_base = OnlineAcceptanceEstimator.for_confidence_head(
        MAX_NUM_REQS, NUM_SPEC_STEPS, device
    )
    est_cand = OnlineAcceptanceEstimator.for_confidence_head(
        MAX_NUM_REQS, NUM_SPEC_STEPS, device
    )
    feats = torch.randn(MAX_NUM_REQS, NUM_SPEC_STEPS, device=device)
    preds = torch.rand(MAX_NUM_REQS, NUM_SPEC_STEPS, device=device) * 0.8 + 0.1
    est_base.features.copy_(feats)
    est_base.predictions.copy_(preds)
    est_cand.features.copy_(feats)
    est_cand.predictions.copy_(preds)

    spec_idx_base = torch.full(( num_reqs,), -1, dtype=torch.int32, device=device)
    spec_idx_cand = torch.full(( num_reqs,), -1, dtype=torch.int32, device=device)

    # Bit-exactness check before timing
    _run_staged_writes_step("baseline", num_reqs, sampler_base, bt_base)
    _run_staged_writes_step("candidate", num_reqs, sampler_cand, bt_cand)
    out_base = _run_prepare_inputs_step(
        "baseline",
        num_reqs,
        req_ids,
        num_tokens_per_req,
        draft_tokens,
        idx_mapping_np,
        num_scheduled_tokens_orig,
        av_base,
        qsl_base,
        device,
    )
    out_cand = _run_prepare_inputs_step(
        "candidate",
        num_reqs,
        req_ids,
        num_tokens_per_req,
        draft_tokens,
        idx_mapping_np,
        num_scheduled_tokens_orig,
        av_cand,
        qsl_cand,
        device,
    )
    est_base.info.zero_()
    est_base.grad.zero_()
    est_base.counts.zero_()
    est_cand.info.zero_()
    est_cand.grad.zero_()
    est_cand.counts.zero_()
    ver_base = _run_post_sample_step(
        "baseline",
        num_reqs,
        idx_mapping,
        num_sampled,
        num_rejected,
        dead_row_cut,
        copy_int_state,
        est_base,
        av_base,
        spec_idx_base,
        max_num_reqs=num_reqs,
    )
    ver_cand = _run_post_sample_step(
        "candidate",
        num_reqs,
        idx_mapping,
        num_sampled,
        num_rejected,
        dead_row_cut,
        copy_int_state,
        est_cand,
        av_cand,
        spec_idx_cand,
        max_num_reqs=num_reqs,
    )
    torch.cuda.synchronize(device)

    bit_exact = (
        torch.equal(out_base[0], out_cand[0])
        and torch.equal(out_base[1], out_cand[1])
        and np.array_equal(out_base[2], out_cand[2])
        and out_base[3] == out_cand[3]
        and torch.equal(out_base[4], out_cand[4])
        and torch.equal(out_base[5], out_cand[5])
        and torch.equal(ver_base, ver_cand)
        and torch.equal(spec_idx_base, spec_idx_cand)
        and torch.allclose(est_base.counts, est_cand.counts, atol=1e-6, rtol=1e-6)
        and torch.allclose(est_base.info, est_cand.info, atol=1e-6, rtol=1e-6)
        and torch.allclose(est_base.grad, est_cand.grad, atol=1e-6, rtol=1e-6)
    )
    if not bit_exact:
        raise RuntimeError(f"Output mismatch at concurrency={num_reqs}")

    results = {"concurrency": num_reqs, "bit_exact": bit_exact}
    for mode, sampler_s, bt_s, av_s, qsl_s, est_s, spec_idx_s in (
        ("baseline", sampler_base, bt_base, av_base, qsl_base, est_base, spec_idx_base),
        ("candidate", sampler_cand, bt_cand, av_cand, qsl_cand, est_cand, spec_idx_cand),
    ):
        for _ in range(warmup):
            _run_staged_writes_step(mode, num_reqs, sampler_s, bt_s)
            _run_prepare_inputs_step(
                mode,
                num_reqs,
                req_ids,
                num_tokens_per_req,
                draft_tokens,
                idx_mapping_np,
                num_scheduled_tokens_orig,
                av_s,
                qsl_s,
                device,
            )
            _run_post_sample_step(
                mode,
                num_reqs,
                idx_mapping,
                num_sampled,
                num_rejected,
                dead_row_cut,
                copy_int_state,
                est_s,
                av_s,
                spec_idx_s,
                max_num_reqs=num_reqs,
            )
        torch.cuda.synchronize(device)

        staged_times = []
        prep_times = []
        post_times = []
        total_times = []
        pipelined_times = []

        for _ in range(iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _run_staged_writes_step(mode, num_reqs, sampler_s, bt_s)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            _run_prepare_inputs_step(
                mode,
                num_reqs,
                req_ids,
                num_tokens_per_req,
                draft_tokens,
                idx_mapping_np,
                num_scheduled_tokens_orig,
                av_s,
                qsl_s,
                device,
            )
            torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            _run_post_sample_step(
                mode,
                num_reqs,
                idx_mapping,
                num_sampled,
                num_rejected,
                dead_row_cut,
                copy_int_state,
                est_s,
                av_s,
                spec_idx_s,
                max_num_reqs=num_reqs,
            )
            torch.cuda.synchronize(device)
            t3 = time.perf_counter()

            s_us = (t1 - t0) * 1e6
            p_us = (t2 - t1) * 1e6
            o_us = (t3 - t2) * 1e6
            staged_times.append(s_us)
            prep_times.append(p_us)
            post_times.append(o_us)
            total_times.append(s_us + p_us + o_us)

        for _ in range(iters):
            torch.cuda.synchronize(device)
            tp0 = time.perf_counter()
            _run_staged_writes_step(mode, num_reqs, sampler_s, bt_s)
            _run_prepare_inputs_step(
                mode,
                num_reqs,
                req_ids,
                num_tokens_per_req,
                draft_tokens,
                idx_mapping_np,
                num_scheduled_tokens_orig,
                av_s,
                qsl_s,
                device,
            )
            _run_post_sample_step(
                mode,
                num_reqs,
                idx_mapping,
                num_sampled,
                num_rejected,
                dead_row_cut,
                copy_int_state,
                est_s,
                av_s,
                spec_idx_s,
                max_num_reqs=num_reqs,
            )
            torch.cuda.synchronize(device)
            tp1 = time.perf_counter()
            pipelined_times.append((tp1 - tp0) * 1e6)

        def _stats(arr: list[float]) -> dict[str, float]:
            s = sorted(arr)
            return {
                "mean_us": round(statistics.mean(s), 2),
                "median_us": round(statistics.median(s), 2),
                "p95_us": round(s[int(0.95 * (len(s) - 1))], 2),
                "stdev_us": round(statistics.stdev(s), 2),
            }

        results[mode] = {
            "staged_writes": _stats(staged_times),
            "prepare_inputs": _stats(prep_times),
            "post_sample": _stats(post_times),
            "total_step_overhead": _stats(total_times),
            "pipelined_step_overhead": _stats(pipelined_times),
        }

    b_tot = results["baseline"]["total_step_overhead"]["mean_us"]
    c_tot = results["candidate"]["total_step_overhead"]["mean_us"]
    b_pipe = results["baseline"]["pipelined_step_overhead"]["mean_us"]
    c_pipe = results["candidate"]["pipelined_step_overhead"]["mean_us"]
    results["delta_mean_us"] = round(c_tot - b_tot, 2)
    results["speedup_x"] = round(b_tot / c_tot, 3) if c_tot > 0 else 0.0
    results["delta_pipelined_mean_us"] = round(c_pipe - b_pipe, 2)
    results["pipelined_speedup_x"] = round(b_pipe / c_pipe, 3) if c_pipe > 0 else 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    payload = {
        "boot": args.boot,
        "device": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "workloads": {},
    }
    for c in (1, 8, 16):
        res = benchmark_concurrency(c, args.warmup, args.iters, device)
        payload["workloads"][f"c{c}"] = res
        print(
            f"[boot {args.boot}] c{c}: baseline={res['baseline']['total_step_overhead']['mean_us']:.2f} us "
            f"candidate={res['candidate']['total_step_overhead']['mean_us']:.2f} us "
            f"delta={res['delta_mean_us']:+.2f} us ({res['speedup_x']:.2f}x) bit_exact={res['bit_exact']}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
