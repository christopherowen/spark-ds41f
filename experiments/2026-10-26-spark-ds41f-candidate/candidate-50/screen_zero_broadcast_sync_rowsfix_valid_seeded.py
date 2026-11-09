#!/usr/bin/env python3
"""2-boot multi-node TP3 & TP4 hardware verification (1,000 steps) and per-step
latency benchmark for Zero-Broadcast Speculative Sync and Lockstep TP Sampling."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import datetime
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace
from typing import Any

CANDIDATE_NODES = [
    ("sp5", "10.30.0.13"),
    ("sp6", "10.30.0.14"),
    ("sp7", "10.30.0.15"),
    ("sp8", "10.30.0.16"),
]

BASE_ENV = {
    "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
    "TP_SOCKET_IFNAME": "enp1s0f0np0",
    "NCCL_IB_HCA": "=rocep1s0f0,roceP2p1s0f0",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_MERGE_NICS": "1",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_CROSS_NIC": "1",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "NCCL_NET": "IB",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def run_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import torch.distributed as dist

    from sparknet import oneshot as roce
    import vllm.distributed.parallel_state as parallel_state
    from vllm.models.deepseek_v4_1.tilelang.confidence import dspark_confidence
    from vllm.sampling_params import SamplingParams
    from vllm.v1.utils import CpuGpuBuffer
    from vllm.v1.worker.gpu import tilelang_inputs
    from vllm.v1.worker.gpu.sample import tilelang_topk_topp
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu.sample.states import (
        _NP_INT64_MAX,
        _NP_INT64_MIN,
        SamplingStates,
    )
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av_mod
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        OnlineAcceptanceEstimator,
    )
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.greedy import select_first_argmax
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    rank = args.rank
    world = args.world_size
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
        device_id=device,
    )

    rt = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=1 << 20,
        max_gather_bytes=4 << 20,
    )
    rt.prepare((torch.bfloat16, torch.float32), padded_gather=False)
    torch.cuda.synchronize()
    dist.barrier()

    def emit(**payload: Any) -> None:
        if rank == 0:
            rec = {
                "boot": args.boot,
                "tp_size": world,
                "utc": utc_now(),
                **payload,
            }
            print("JSON_RECORD:" + json.dumps(rec), flush=True)

    def make_live_manager(
        max_num_reqs: int,
        num_steps: int,
        planned_confidences: bool,
        tp_invariant: bool,
        row_cost: float = 0.035,
        dead_rows_tau: float = 0.20,
    ) -> AdaptiveVerificationManager:
        mgr = AdaptiveVerificationManager.__new__(AdaptiveVerificationManager)
        mgr.num_speculative_steps = num_steps
        mgr.num_draft_steps = num_steps
        mgr.step_steps = num_steps
        mgr._copy_stream = torch.cuda.Stream(device)
        mgr.num_bonus_tokens = 1
        mgr._max_total_logits = 1 << 20
        mgr.row_cost = row_cost
        mgr.high_row_cost_knee = 16
        mgr.high_row_cost_scale = 0.55
        mgr.tp_invariant_confidence = tp_invariant
        mgr.calibrated_confidence = False
        mgr.dead_rows_tau = dead_rows_tau
        mgr._has_admitted_drafts = False
        mgr._dead_row_cut = torch.empty(max_num_reqs, dtype=torch.int32, device=device)
        mgr.query_start_loc = torch.arange(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        mgr._batch_budget = None
        mgr._confidence_buffer = torch.ones(
            (1 + int(planned_confidences), max_num_reqs, num_steps),
            dtype=torch.float32,
            device=device,
        )
        mgr._confidence_probs = mgr._confidence_buffer[0]
        mgr._planned_probs = mgr._confidence_buffer[-1]
        mgr._batch_draft_capacity = torch.empty(
            max_num_reqs, dtype=torch.int32, device=device
        )
        mgr._num_non_draft_tokens = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )
        mgr._cu_num_logits = torch.arange(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        mgr._stale_confidences = [
            CpuGpuBuffer(max_num_reqs, num_steps, dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        mgr._copy_events = [torch.cuda.Event(blocking=True) for _ in range(2)]
        mgr._pending_resets = []
        mgr._new_rows = []
        mgr._stale_idx = 0
        for slot in mgr._stale_confidences:
            slot.np.fill(1.0)
            slot.gpu.fill_(1.0)
        mgr.req_states = SimpleNamespace(
            device=device,
            max_num_reqs=max_num_reqs,
            num_speculative_steps=num_steps,
            req_id_to_index={f"r{i}": i for i in range(max_num_reqs)},
            num_computed_tokens_np=np.full(max_num_reqs, 64, dtype=np.int32),
            prefill_len=SimpleNamespace(np=np.full(max_num_reqs, 16, dtype=np.int32)),
        )
        return mgr

    # Shared model weights replicated identically across TP ranks (seed=42)
    num_reqs = 8
    num_steps = 5
    hidden_size = 5120
    markov_rank = 256
    vocab_size = 4096 * world  # divisible by world_size
    shard_vocab = vocab_size // world
    vocab_start = rank * shard_vocab

    gen_shared = torch.Generator(device=device).manual_seed(42)
    conf_weight = torch.randn(
        hidden_size + markov_rank,
        dtype=torch.float32,
        device=device,
        generator=gen_shared,
    ) / ((hidden_size + markov_rank) ** 0.5)
    markov_table = torch.randn(
        vocab_size,
        markov_rank,
        dtype=torch.bfloat16,
        device=device,
        generator=gen_shared,
    )

    def check_tensor_equal_across_ranks(t: torch.Tensor) -> tuple[bool, float]:
        flat = t.contiguous().view(-1)
        gathered = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        ref = gathered[0]
        exact = all(torch.equal(g, ref) for g in gathered[1:])
        if flat.is_floating_point():
            max_diff = max(
                float((g.float() - ref.float()).abs().max().item())
                for g in gathered[1:]
            )
        else:
            max_diff = max(
                float((g.to(torch.int64) != ref.to(torch.int64)).sum().item())
                for g in gathered[1:]
            )
        return exact, max_diff

    # =========================================================================
    # PART 1: 1,000-Step Multi-Rank Determinism Verification
    # =========================================================================
    steps_verify = args.verify_steps
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    req_ids = [f"r{i}" for i in range(num_reqs)]
    num_tokens_per_req = {rid: 1 + num_steps for rid in req_ids}
    draft_tokens_dict = {rid: [1, 2, 3, 4, 5] for rid in req_ids}
    batch_obj = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        has_prefill=False,
    )

    def run_1000_step_simulation(mode: str) -> dict[str, Any]:
        """Run 1,000 steps of speculative decoding and check cross-rank bit identity."""
        broadcast_calls = 0

        def fake_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
            nonlocal broadcast_calls
            broadcast_calls += 1
            flat = tensor.contiguous().view(-1)
            gathered = rt.all_gather(flat, dim=0)
            if rank != src:
                flat.copy_(gathered.view(world, -1)[src])
            return tensor

        av_mod.get_tp_group = lambda: SimpleNamespace(
            world_size=world,
            broadcast=fake_broadcast,
            device_communicator=SimpleNamespace(rank_order_reduce_scatter=True),
        )
        parallel_state.model_parallel_is_initialized = lambda: True
        parallel_state.get_tensor_model_parallel_world_size = lambda: world

        mgr = make_live_manager(
            max_num_reqs=num_reqs,
            num_steps=num_steps,
            planned_confidences=(mode == "candidate_greedy"),
            tp_invariant=True,
        )
        est = (
            OnlineAcceptanceEstimator(num_reqs, num_steps, device)
            if mode == "candidate_dflash_estimator"
            else OnlineAcceptanceEstimator.for_confidence_head(
                num_reqs, num_steps, device
            )
        )
        mgr.calibrated_confidence = lambda: est._refits > 0

        # Set up SamplingStates
        states = (
            SamplingStates(num_reqs, vocab_size)
            if args.sampling_seed is None
            else SamplingStates(num_reqs, vocab_size, seed=args.sampling_seed)
        )
        # Intentionally perturb process-global np.random asymmetrically per rank!
        np.random.seed(1000 + rank * 7919)
        _ = np.random.randint(0, 100000, size=rank * 17 + 3)

        is_greedy = mode == "candidate_greedy"
        for r_idx in range(num_reqs):
            sp = SamplingParams(
                temperature=0.0 if is_greedy else 0.8,
                top_p=1.0 if is_greedy else 0.95,
                top_k=-1 if is_greedy else 50,
                seed=None,
            )
            if mode == "control_unseeded_global_rng":
                # Baseline 1107 behavior before 0762: global np.random.randint
                states.temperature.np[r_idx] = sp.temperature
                states.top_p.np[r_idx] = sp.top_p
                states.top_k.np[r_idx] = 50
                states.seeds_set[r_idx] = False
                states.seeds.np[r_idx] = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
            else:
                # Candidate 0762 behavior: isolated lockstep _seed_rng
                states.add_request(r_idx, sp)
            # Additional rank-asymmetric global np.random perturbation between requests
            _ = np.random.random(rank * 5 + 1)
        states.apply_staged_writes()

        seeds_equal, _ = check_tensor_equal_across_ranks(states.seeds.gpu)

        hist_draft = torch.empty((steps_verify, num_reqs, num_steps), dtype=torch.int64, device=device)
        hist_conf = torch.empty((steps_verify, num_reqs, num_steps), dtype=torch.float32, device=device)
        hist_budget = torch.empty((steps_verify, 2), dtype=torch.int32, device=device)
        hist_cap = torch.empty((steps_verify, num_reqs), dtype=torch.int32, device=device)
        hist_dead = torch.empty((steps_verify, num_reqs), dtype=torch.int32, device=device)
        hist_sampled = torch.empty((steps_verify, num_reqs, 1 + num_steps), dtype=torch.int64, device=device)
        hist_sampled_raw = torch.empty((steps_verify, num_reqs, 1 + num_steps), dtype=torch.int64, device=device)
        hist_nsampled = torch.empty((steps_verify, num_reqs), dtype=torch.int32, device=device)
        hist_slope = torch.empty((steps_verify, 1), dtype=torch.float32, device=device)
        hist_inter = torch.empty((steps_verify, num_steps), dtype=torch.float32, device=device)

        draft_tokens = torch.zeros(
            (num_reqs, num_steps), dtype=torch.int64, device=device
        )
        confidence_probs = torch.empty(
            (num_reqs, num_steps), dtype=torch.float32, device=device
        )
        planned_probs = torch.empty(
            (num_reqs, num_steps), dtype=torch.float32, device=device
        )
        draft_logits = torch.empty(
            (num_reqs, num_steps, vocab_size), dtype=torch.float32, device=device
        )
        step_cols = [
            torch.tensor(s, dtype=torch.int32, device=device) for s in range(num_steps)
        ]
        is_padding = torch.zeros(
            num_reqs * (1 + num_steps), dtype=torch.bool, device=device
        )

        # Per-rank generator for local shards (different per rank, reduced via RoCE!)
        gen_rank = torch.Generator(device=device).manual_seed(9000 + rank * 101)
        # Shared generator for target verification logits
        gen_target = torch.Generator(device=device).manual_seed(77777)

        for step in range(steps_verify):
            # Perturb global np.random asymmetrically on every step
            if step % 7 == rank:
                _ = np.random.randint(0, 1000)

            # 1. Layer hidden state shard -> Sparknet one-shot RoCE all-reduce (rank-ordered)
            hidden_shard = torch.randn(
                (num_reqs * num_steps, hidden_size),
                dtype=torch.bfloat16,
                device=device,
                generator=gen_rank,
            )
            sample_hidden = rt.all_reduce(hidden_shard)

            # 2. Draft tokens and confidences
            if is_greedy:
                # Vocab-parallel greedy pair reduction via Sparknet one-shot RoCE all-gather
                local_scores = torch.randn(
                    (num_reqs, num_steps, shard_vocab),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=gen_rank,
                )
                local_max, local_idx = local_scores.max(dim=-1)
                local_global_idx = local_idx.to(torch.int32) + vocab_start
                for s in range(num_steps):
                    pairs = torch.zeros(
                        (num_reqs, 4), dtype=torch.float32, device=device
                    )
                    pairs[:, 0] = local_max[:, s].float()
                    pairs.view(torch.int32)[:, 1] = local_global_idx[:, s]
                    gathered_pairs = (
                        rt.all_gather(pairs.view(-1), dim=0)
                        .view(world, num_reqs, 4)
                        .permute(1, 0, 2)
                        .contiguous()
                        .view(num_reqs, world * 4)
                    )
                    select_first_argmax(
                        gathered_pairs,
                        draft_tokens[:, s],
                        pair_width=4,
                    )
                markov_embeds = markov_table[draft_tokens.view(-1)]
                dspark_confidence(
                    sample_hidden,
                    markov_embeds,
                    conf_weight,
                    confidence_probs.view(-1),
                )
                est.calibrate_confidence_probs(confidence_probs, idx_mapping)
                planned_probs.copy_(confidence_probs * 0.95)
            elif mode == "candidate_dflash_estimator":
                local_logits = torch.randn(
                    (num_reqs, shard_vocab),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=gen_rank,
                )
                full_logits = (
                    rt.all_gather(local_logits.contiguous(), dim=0)
                    .view(world, num_reqs, shard_vocab)
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(num_reqs, vocab_size)
                )
                for s in range(num_steps):
                    step_logits = full_logits + float(s) * 0.05
                    est.predict(
                        step_logits,
                        idx_mapping,
                        step_cols[s],
                        confidence_probs,
                        states.temperature.gpu,
                    )
                    pos_s = torch.full(
                        (num_reqs,), 64 + step * 6 + s, dtype=torch.int64, device=device
                    )
                    draft_tokens[:, s] = gumbel_sample(
                        step_logits,
                        idx_mapping,
                        states.temperature.gpu,
                        states.seeds.gpu,
                        pos_s,
                        apply_temperature=True,
                        is_drafting=True,
                        logits_cache=draft_logits,
                        logits_cache_col=step_cols[s],
                    )
            else:
                # Probabilistic DSpark drafting (control vs candidate)
                local_logits = torch.randn(
                    (num_reqs, shard_vocab),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=gen_rank,
                )
                full_logits = (
                    rt.all_gather(local_logits.contiguous(), dim=0)
                    .view(world, num_reqs, shard_vocab)
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(num_reqs, vocab_size)
                )
                for s in range(num_steps):
                    step_logits = full_logits + float(s) * 0.05
                    pos_s = torch.full(
                        (num_reqs,), 64 + step * 6 + s, dtype=torch.int64, device=device
                    )
                    draft_tokens[:, s] = gumbel_sample(
                        step_logits,
                        idx_mapping,
                        states.temperature.gpu,
                        states.seeds.gpu,
                        pos_s,
                        apply_temperature=True,
                        is_drafting=True,
                        logits_cache=draft_logits,
                        logits_cache_col=step_cols[s],
                    )
                markov_embeds = markov_table[draft_tokens.view(-1)]
                dspark_confidence(
                    sample_hidden,
                    markov_embeds,
                    conf_weight,
                    confidence_probs.view(-1),
                )
                est.calibrate_confidence_probs(confidence_probs, idx_mapping)

            # 3. Record confidences in AdaptiveVerificationManager (0 broadcasts!)
            mgr.record_confidences(
                confidence_probs,
                batch_obj,
                planned=planned_probs if is_greedy else None,
            )

            # 4. Host-side budget planning & device-side dead-row pruning
            total_tokens = mgr.get_num_tokens(num_tokens_per_req, draft_tokens_dict)
            assert mgr._batch_budget is not None
            _, scheduled_map, draft_budget = mgr._batch_budget
            capacities_t = torch.tensor(
                [scheduled_map[rid] for rid in req_ids],
                dtype=torch.int32,
                device=device,
            )
            dead_info = mgr.dead_row_args(is_padding, num_reqs)
            dead_cut = (
                dead_info.cut
                if dead_info is not None
                else torch.full((num_reqs,), num_steps, dtype=torch.int32, device=device)
            )

            # 5. Target verification & OnlineAcceptanceEstimator step
            num_verify_rows = num_reqs * (1 + num_steps)
            target_logits = torch.randn(
                (num_verify_rows, vocab_size),
                dtype=torch.float32,
                device=device,
                generator=gen_target,
            )
            expanded_idx = idx_mapping.repeat_interleave(1 + num_steps)
            expanded_local_pos = torch.arange(
                1 + num_steps, dtype=torch.int32, device=device
            ).repeat(num_reqs)
            cu_num_logits = torch.arange(
                0,
                (num_reqs + 1) * (1 + num_steps),
                1 + num_steps,
                dtype=torch.int32,
                device=device,
            )
            pos_verify = torch.arange(
                num_verify_rows, dtype=torch.int64, device=device
            ) + (64 + step * 6)
            if is_greedy:
                sampled_tokens = target_logits.argmax(dim=-1).view(
                    num_reqs, 1 + num_steps
                )
                matches = sampled_tokens[:, :num_steps] == draft_tokens
                # Count consecutive accepted drafts
                all_prev = torch.cumprod(matches.to(torch.int32), dim=1)
                num_accepted = all_prev.sum(dim=1).to(torch.int32)
                num_accepted = torch.minimum(num_accepted, dead_cut)
                num_sampled = num_accepted + 1
                num_rejected = (dead_cut - num_accepted).clamp_min(0)
            else:
                top_k_t, top_p_t = states.get_top_k_top_p(expanded_idx, idx_mapping_np)
                proc_logits = tilelang_topk_topp.apply_top_k_top_p(
                    target_logits, top_k_t, top_p_t
                )
                # rejection_sample reads draft_sampled as [num_logits], each
                # request's drafts at its rows 1..num_steps (the worker passes
                # input_ids[logits_indices]); draft_tokens.view(-1) is only
                # num_reqs * num_steps long.
                draft_rows = torch.zeros(
                    (num_reqs, 1 + num_steps), dtype=draft_tokens.dtype, device=device
                )
                draft_rows[:, 1:] = draft_tokens
                sampled_tokens, num_sampled = rejection_sample(
                    proc_logits,
                    draft_logits,
                    draft_rows.view(-1),
                    cu_num_logits,
                    pos_verify,
                    idx_mapping,
                    expanded_idx,
                    expanded_local_pos,
                    states.temperature.gpu,
                    states.seeds.gpu,
                    num_steps,
                    None,
                    use_fp64=False,
                    use_block_verification=False,
                    dead_row_cut=dead_cut,
                )
                raw_sampled = sampled_tokens.clone()
                written = (
                    torch.arange(1 + num_steps, device=device)[None, :]
                    < num_sampled.to(torch.int64)[:, None]
                )
                sampled_tokens = torch.where(
                    written, sampled_tokens, torch.full_like(sampled_tokens, -1)
                )
                num_accepted = (num_sampled - 1).clamp_min(0)
                num_rejected = (dead_cut - num_accepted).clamp_min(0)

            est.step(idx_mapping, num_sampled, num_rejected, dead_row_cut=dead_cut)

            hist_draft[step].copy_(draft_tokens)
            hist_conf[step].copy_(mgr._confidence_probs)
            hist_budget[step, 0] = draft_budget
            hist_budget[step, 1] = total_tokens
            hist_cap[step].copy_(capacities_t)
            hist_dead[step].copy_(dead_cut)
            hist_sampled[step].copy_(sampled_tokens.view(num_reqs, 1 + num_steps))
            hist_sampled_raw[step].copy_(
                (sampled_tokens if is_greedy else raw_sampled).view(num_reqs, 1 + num_steps)
            )
            hist_nsampled[step].copy_(num_sampled)
            hist_slope[step].copy_(est.slope)
            hist_inter[step].copy_(est.intercepts)

        mgr._copy_stream.synchronize()
        torch.cuda.synchronize()
        dist.barrier()

        def count_step_mismatches(hist: torch.Tensor) -> tuple[int, float]:
            flat = hist.contiguous().view(steps_verify, -1)
            gathered = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(gathered, flat)
            ref = gathered[0]
            step_mismatches = torch.zeros(steps_verify, dtype=torch.bool, device=device)
            max_diff = 0.0
            for g in gathered[1:]:
                step_mismatches |= (g != ref).any(dim=1)
                if flat.is_floating_point():
                    diff = float((g.float() - ref.float()).abs().max().item())
                    max_diff = max(max_diff, diff)
            return int(step_mismatches.sum().item()), max_diff

        mismatch_draft_steps, _ = count_step_mismatches(hist_draft)
        mismatch_conf_steps, max_conf_abs_diff = count_step_mismatches(hist_conf)
        mismatch_budget_steps, _ = count_step_mismatches(hist_budget)
        mismatch_capacities_steps, _ = count_step_mismatches(hist_cap)
        mismatch_dead_cut_steps, _ = count_step_mismatches(hist_dead)
        mm_s1, _ = count_step_mismatches(hist_sampled)
        mismatch_sampled_raw_steps, _ = count_step_mismatches(hist_sampled_raw)
        mm_s2, _ = count_step_mismatches(hist_nsampled)
        mismatch_sampled_steps = max(mm_s1, mm_s2)
        mm_e1, max_slope_abs_diff = count_step_mismatches(hist_slope)
        mm_e2, max_intercept_abs_diff = count_step_mismatches(hist_inter)
        mismatch_estimator_steps = max(mm_e1, mm_e2)

        return {
            "mode": mode,
            "sampling_seed": args.sampling_seed,
            "steps": steps_verify,
            "broadcast_calls": broadcast_calls,
            "seeds_equal_across_ranks": seeds_equal,
            "refits_completed": est._refits,
            "mismatch_draft_steps": mismatch_draft_steps,
            "mismatch_conf_steps": mismatch_conf_steps,
            "mismatch_budget_steps": mismatch_budget_steps,
            "mismatch_capacities_steps": mismatch_capacities_steps,
            "mismatch_dead_cut_steps": mismatch_dead_cut_steps,
            "mismatch_sampled_steps": mismatch_sampled_steps,
            "unchecked_sampled_raw_steps": mismatch_sampled_raw_steps,
            "mismatch_estimator_steps": mismatch_estimator_steps,
            "max_conf_abs_diff": round(max_conf_abs_diff, 8),
            "max_slope_abs_diff": round(max_slope_abs_diff, 8),
            "max_intercept_abs_diff": round(max_intercept_abs_diff, 8),
        }

    for sim_mode in (
        "control_unseeded_global_rng",
        "candidate_greedy",
        "candidate_sampled_unseeded",
        "candidate_dflash_estimator",
    ):
        torch.cuda.synchronize()
        dist.barrier()
        res = run_1000_step_simulation(sim_mode)
        emit(kind="determinism_1000_steps", **res)
        if sim_mode.startswith("candidate_"):
            assert res["broadcast_calls"] == 0, f"Expected 0 broadcasts in {sim_mode}"
            assert res["seeds_equal_across_ranks"] is True
            assert res["mismatch_draft_steps"] == 0
            assert res["mismatch_conf_steps"] == 0
            assert res["mismatch_budget_steps"] == 0
            assert res["mismatch_capacities_steps"] == 0
            assert res["mismatch_dead_cut_steps"] == 0
            assert res["mismatch_sampled_steps"] == 0
            assert res["mismatch_estimator_steps"] == 0

    # =========================================================================
    # PART 2: Per-Step Sync & Estimator Latency Benchmark (200 warmup + 1000 iters)
    # =========================================================================
    warmup_iters = 200
    bench_iters = args.bench_iters

    for payload_name, n_steps, has_planned in (
        ("128B_1x8x4", 4, False),
        ("160B_1x8x5", 5, False),
        ("320B_2x8x5_planned", 5, True),
    ):
        conf_in = torch.full(
            (num_reqs, n_steps), 0.78, dtype=torch.float32, device=device
        )
        plan_in = (
            torch.full((num_reqs, n_steps), 0.74, dtype=torch.float32, device=device)
            if has_planned
            else None
        )

        for sync_arm in (
            "nccl_torch_dist_broadcast",
            "roce_oneshot_broadcast",
            "zero_broadcast_tp_invariant",
        ):
            mgr = make_live_manager(
                max_num_reqs=num_reqs,
                num_steps=n_steps,
                planned_confidences=has_planned,
                tp_invariant=(sync_arm == "zero_broadcast_tp_invariant"),
            )

            def do_bcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
                if sync_arm == "nccl_torch_dist_broadcast":
                    dist.broadcast(tensor, src=src)
                    return tensor
                flat = tensor.view(-1)
                gathered = rt.all_gather(flat, dim=0)
                if rank != src:
                    flat.copy_(gathered.view(world, -1)[src])
                return tensor

            av_mod.get_tp_group = lambda: SimpleNamespace(
                world_size=world,
                broadcast=do_bcast,
                device_communicator=SimpleNamespace(rank_order_reduce_scatter=True),
            )

            for _ in range(warmup_iters):
                mgr.record_confidences(conf_in, batch_obj, planned=plan_in)
            torch.cuda.synchronize()
            dist.barrier()

            samples_us: list[float] = []
            lat_t = torch.empty((1,), dtype=torch.float32, device=device)
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

            for _ in range(bench_iters):
                dist.barrier()
                start_ev.record()
                mgr.record_confidences(conf_in, batch_obj, planned=plan_in)
                end_ev.record()
                torch.cuda.synchronize()
                lat_t[0] = start_ev.elapsed_time(end_ev) * 1000.0
                dist.all_reduce(lat_t, op=dist.ReduceOp.MAX)
                samples_us.append(float(lat_t.item()))

            mgr._copy_stream.synchronize()
            torch.cuda.synchronize()
            samples_sorted = sorted(samples_us)
            emit(
                kind="record_confidences_latency",
                payload=payload_name,
                payload_bytes=mgr._confidence_buffer.numel() * 4,
                sync_arm=sync_arm,
                median_us=round(statistics.median(samples_sorted), 3),
                mean_us=round(statistics.mean(samples_sorted), 3),
                p90_us=round(_pct(samples_sorted, 90), 3),
                p99_us=round(_pct(samples_sorted, 99), 3),
                min_us=round(samples_sorted[0], 3),
            )

    # Also benchmark OnlineAcceptanceEstimator.predict (control alloc vs candidate scratch vs predict_from_stats)
    est_bench = OnlineAcceptanceEstimator(num_reqs, num_steps, device)
    logits_bench = torch.randn(
        (num_reqs, 129280), dtype=torch.bfloat16, device=device, generator=gen_shared
    )
    conf_bench = torch.empty((num_reqs, num_steps), dtype=torch.float32, device=device)
    temp_bench = torch.ones((num_reqs,), dtype=torch.float32, device=device)
    step_zero = torch.tensor(0, dtype=torch.int32, device=device)
    stats_max = torch.zeros((num_reqs, world), dtype=torch.float32, device=device)
    stats_sum = torch.full((num_reqs, world), 10.0, dtype=torch.float32, device=device)

    from vllm.triton_utils import triton
    from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
        _MAX_LOG_ODDS,
        _local_max_sumexp_kernel,
        _predict_kernel,
    )

    def run_est_control():
        num_tokens, v_size = logits_bench.shape
        n_blocks = triton.cdiv(v_size, 4096)
        l_max = torch.empty(num_tokens, n_blocks, dtype=torch.float32, device=device)
        l_sum = torch.empty_like(l_max)
        _local_max_sumexp_kernel[(num_tokens, n_blocks)](
            l_max,
            l_max.stride(0),
            l_sum,
            l_sum.stride(0),
            logits_bench,
            logits_bench.stride(0),
            idx_mapping,
            idx_mapping.stride(0),
            temp_bench,
            num_tokens,
            v_size,
            BLOCK_SIZE=4096,
        )
        _predict_kernel[(num_tokens,)](
            est_bench.features,
            est_bench.features.stride(0),
            est_bench.predictions,
            est_bench.predictions.stride(0),
            conf_bench,
            conf_bench.stride(0),
            est_bench.slope,
            est_bench.intercepts,
            l_max,
            l_max.stride(0),
            l_sum,
            l_sum.stride(0),
            idx_mapping,
            idx_mapping.stride(0),
            step_zero,
            num_tokens,
            n_blocks,
            per_token_step=False,
            NUM_SPECULATIVE_STEPS=num_steps,
            MAX_LOG_ODDS=_MAX_LOG_ODDS,
            PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(n_blocks),
        )

    def run_est_candidate_scratch():
        est_bench.predict(logits_bench, idx_mapping, step_zero, conf_bench, temp_bench)

    def run_est_candidate_from_stats():
        est_bench.predict_from_stats(
            stats_max, stats_sum, idx_mapping, step_zero, conf_bench
        )

    for est_arm, fn in (
        ("estimator_predict_alloc_control", run_est_control),
        ("estimator_predict_scratch_candidate", run_est_candidate_scratch),
        ("estimator_predict_from_stats_candidate", run_est_candidate_from_stats),
    ):
        for _ in range(warmup_iters):
            fn()
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        samples_us = []
        for _ in range(bench_iters):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            samples_us.append(float(start_ev.elapsed_time(end_ev) * 1000.0))
        samples_sorted = sorted(samples_us)
        emit(
            kind="estimator_predict_latency",
            arm=est_arm,
            median_us=round(statistics.median(samples_sorted), 3),
            mean_us=round(statistics.mean(samples_sorted), 3),
            p90_us=round(_pct(samples_sorted, 90), 3),
        )

    dist.barrier()
    rt.close()
    dist.destroy_process_group()


def acquire_cluster_locks(nodes: list[tuple[str, str]]) -> list[subprocess.Popen[str]]:
    locks: list[subprocess.Popen[str]] = []
    try:
        for host, _ in sorted(nodes, key=lambda x: x[0]):
            print(f"Acquiring /tmp/gpu.lock on {host}...", flush=True)
            p = subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    host,
                    'flock /tmp/gpu.lock -c "echo LOCKED; cat"',
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert p.stdout is not None
            line = p.stdout.readline().strip()
            if line != "LOCKED":
                stderr = p.stderr.read() if p.stderr else ""
                raise RuntimeError(f"Failed to lock {host}: {line!r} {stderr}")
            locks.append(p)
        return locks
    except Exception:
        release_cluster_locks(locks)
        raise


def release_cluster_locks(locks: list[subprocess.Popen[str]]) -> None:
    for p in locks:
        try:
            if p.stdin:
                p.stdin.close()
            p.wait(timeout=5)
        except Exception:
            p.kill()


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    # Sync candidate vllm files and this script to sp5..sp8
    files_to_sync = [
        "vllm/v1/worker/gpu/sample/states.py",
        "vllm/v1/worker/gpu/sample/sampler.py",
        "vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py",
        "vllm/v1/worker/gpu/spec_decode/speculator.py",
        "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
        "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py",
    ]
    vllm_src = Path(args.vllm_dir).resolve()
    for host, _ in CANDIDATE_NODES:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, "mkdir -p /tmp/zero-bcast-mount"],
            check=True,
        )
        subprocess.run(
            ["scp", "-q", str(script_path), f"{host}:/tmp/screen_zero_broadcast_sync.py"],
            check=True,
        )
        for rel in files_to_sync:
            flat_name = rel.replace("/", "__")
            subprocess.run(
                [
                    "scp",
                    "-q",
                    str(vllm_src / rel),
                    f"{host}:/tmp/zero-bcast-mount/{flat_name}",
                ],
                check=True,
            )

    locks = acquire_cluster_locks(CANDIDATE_NODES)
    try:
        for boot_idx, boot in enumerate(("boot1", "boot2")):
            out_jsonl = out_dir / f"screen-zero-broadcast-sync-{boot}.jsonl"
            if out_jsonl.exists():
                out_jsonl.unlink()

            tp_orders = [3, 4] if boot_idx == 0 else [4, 3]
            for tp_idx, tp_size in enumerate(tp_orders):
                active_nodes = CANDIDATE_NODES[:tp_size]
                master_addr = active_nodes[0][1]
                port = 29740 + boot_idx * 10 + tp_idx
                print(
                    f"=== [{boot}] Running TP{tp_size} zero-broadcast verification & benchmark on {[n[0] for n in active_nodes]} ===",
                    flush=True,
                )

                env_flags: list[str] = []
                for k, v in BASE_ENV.items():
                    env_flags.extend(["-e", f"{k}={v}"])

                mount_flags: list[str] = [
                    "-v",
                    "/tmp/screen_zero_broadcast_sync.py:/tmp/screen_zero_broadcast_sync.py:ro",
                ]
                for rel in files_to_sync:
                    flat_name = rel.replace("/", "__")
                    mount_flags.extend(
                        [
                            "-v",
                            f"/tmp/zero-bcast-mount/{flat_name}:/opt/spark-ds41f/candidate/vllm/{rel}:ro",
                        ]
                    )

                procs: list[tuple[str, subprocess.Popen[str]]] = []
                try:
                    for rank, (host, _) in enumerate(active_nodes):
                        cname = f"zero-bcast-{boot}-tp{tp_size}-r{rank}"
                        docker_cmd = [
                            "docker",
                            "run",
                            "--rm",
                            "--name",
                            cname,
                            "--network=host",
                            "--ipc=host",
                            "--gpus=all",
                            "--device=/dev/infiniband:/dev/infiniband:rwm",
                            "--cap-add=IPC_LOCK",
                            "--ulimit=memlock=-1:-1",
                            "--entrypoint",
                            "python3",
                            *env_flags,
                            *mount_flags,
                            args.image,
                            "/tmp/screen_zero_broadcast_sync.py",
                            "--worker",
                            "--boot",
                            boot,
                            "--rank",
                            str(rank),
                            "--world-size",
                            str(tp_size),
                            "--master-addr",
                            master_addr,
                            "--master-port",
                            str(port),
                            "--verify-steps",
                            str(args.verify_steps),
                            "--bench-iters",
                            str(args.bench_iters),
                        ]
                        p = subprocess.Popen(
                            ["ssh", "-o", "BatchMode=yes", host, " ".join(docker_cmd)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        procs.append((host, p))

                    rank0_out, _ = procs[0][1].communicate(timeout=300)
                    for line in rank0_out.splitlines():
                        if line.startswith("JSON_RECORD:"):
                            rec_str = line[len("JSON_RECORD:") :]
                            print(rec_str, flush=True)
                            with out_jsonl.open("a") as f:
                                f.write(rec_str + "\n")
                    for host, p in procs:
                        out_other = (
                            rank0_out
                            if host == procs[0][0]
                            else p.communicate(timeout=300)[0]
                        )
                        if p.returncode != 0:
                            raise RuntimeError(
                                f"Worker on {host} failed (rc={p.returncode}):\n{out_other}"
                            )
                finally:
                    for host, p in procs:
                        if p.poll() is None:
                            p.kill()
    finally:
        release_cluster_locks(locks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--boot", type=str, default="boot1")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--master-addr", type=str, default="10.30.0.13")
    parser.add_argument("--master-port", type=int, default=29740)
    parser.add_argument("--verify-steps", type=int, default=1000)
    parser.add_argument("--bench-iters", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="/tmp/zero-bcast-receipts")
    parser.add_argument("--vllm-dir", type=str, default="/tmp/zero-bcast-vllm")
    parser.add_argument("--image", type=str, default="spark-ds41f:1107")
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
