#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-boot GPU verification and microbenchmark for:
1. Dead verification row skipping in Sampler.apply_sampling_params / RejectionSampler._verify.
2. Sparse-support sequential Markov sampling in DSparkSpeculator._sample_sequential_topk.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4ForCausalLM
from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight
from vllm.v1.worker.gpu import tilelang_inputs
from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.states import RequestState

VOCAB_SIZE = 129280
N_SPEC = 5
ROWS_PER_REQ = N_SPEC + 1
MARKOV_RANK = 256
DRAFT_TOPK = 128


def _bench_cuda_us(fn, warmup: int = 25, iters: int = 100) -> dict[str, float]:
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
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    arr = np.array(times_us, dtype=np.float64)
    return {
        "median_us": float(np.median(arr)),
        "mean_us": float(np.mean(arr)),
        "p10_us": float(np.percentile(arr, 10)),
        "p90_us": float(np.percentile(arr, 90)),
    }


def verify_and_bench_dead_row_sampler(device: torch.device) -> dict:
    max_num_reqs = 16
    req_states = RequestState(
        max_num_reqs=max_num_reqs,
        max_model_len=128,
        max_num_batched_tokens=max_num_reqs * ROWS_PER_REQ,
        num_speculative_steps=N_SPEC,
        vocab_size=VOCAB_SIZE,
        device=device,
    )
    sampler = Sampler(
        max_num_reqs=max_num_reqs,
        vocab_size=VOCAB_SIZE,
        device=device,
        req_states=req_states,
    )
    from vllm.sampling_params import SamplingParams

    for s_idx in range(max_num_reqs):
        req_states.add_request(
            req_id=f"req_{s_idx}",
            prompt_len=8,
            all_token_ids=list(range(10, 26)),
            num_computed_tokens=16,
            max_tokens=64,
        )
        slot = req_states.req_id_to_index[f"req_{s_idx}"]
        params = SamplingParams(
            temperature=0.8,
            top_k=64,
            top_p=0.9,
            min_p=0.05,
            repetition_penalty=1.15,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            logit_bias={7: 1.5, 12: -2.0},
            seed=1000 + slot,
        )
        params._bad_words_token_ids = [[15, 16], [25]]
        sampler.add_request(slot, 8, params)
    req_states.apply_staged_writes()
    sampler.apply_staged_writes()

    results: dict[str, dict] = {}
    for batch_size in (1, 4, 8, 16):
        num_rows = batch_size * ROWS_PER_REQ
        idx_mapping_np = np.arange(batch_size, dtype=np.int64)
        idx_mapping = torch.from_numpy(idx_mapping_np).to(device)
        expanded_idx_mapping = (
            torch.arange(batch_size, dtype=torch.int64, device=device)
            .repeat_interleave(ROWS_PER_REQ)
            .contiguous()
        )
        expanded_local_pos = (
            torch.arange(ROWS_PER_REQ, dtype=torch.int32, device=device)
            .repeat(batch_size)
            .contiguous()
        )
        cu_num_logits = torch.arange(
            0, (batch_size + 1) * ROWS_PER_REQ, ROWS_PER_REQ, dtype=torch.int32, device=device
        )
        pos = (
            16
            + torch.arange(ROWS_PER_REQ, dtype=torch.int64, device=device).repeat(
                batch_size
            )
        )
        draft_sampled = torch.randint(
            1, 1000, (num_rows,), dtype=torch.int32, device=device
        )

        torch.manual_seed(42 + batch_size)
        raw_logits = torch.randn(
            (num_rows, VOCAB_SIZE), dtype=torch.bfloat16, device=device
        )

        for cut_val in (1, 2):
            dead_row_cut = torch.full(
                (batch_size,), cut_val, dtype=torch.int32, device=device
            )
            live_mask = expanded_local_pos <= cut_val
            raw_dead_nan = raw_logits.clone()
            raw_dead_nan[~live_mask] = float("nan")

            ref_proc = sampler.apply_sampling_params(
                raw_logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
            )
            cand_proc = sampler.apply_sampling_params(
                raw_dead_nan,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
                dead_row_cut=dead_row_cut,
                cu_num_logits=cu_num_logits,
            )
            assert torch.equal(cand_proc[live_mask], ref_proc[live_mask]), (
                f"Mismatch in live rows at B={batch_size}, cut={cut_val}"
            )

            ref_sampled, ref_num = rejection_sample(
                ref_proc,
                None,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                sampler.sampling_states.temperature.gpu,
                sampler.sampling_states.seeds.gpu,
                N_SPEC,
                None,
                use_fp64=False,
                use_block_verification=True,
                dead_row_cut=dead_row_cut,
                kernel_steps=N_SPEC,
            )
            cand_sampled, cand_num = rejection_sample(
                cand_proc,
                None,
                draft_sampled,
                cu_num_logits,
                pos,
                idx_mapping,
                expanded_idx_mapping,
                expanded_local_pos,
                sampler.sampling_states.temperature.gpu,
                sampler.sampling_states.seeds.gpu,
                N_SPEC,
                None,
                use_fp64=False,
                use_block_verification=True,
                dead_row_cut=dead_row_cut,
                kernel_steps=N_SPEC,
            )
            assert torch.equal(cand_num, ref_num)
            for r in range(batch_size):
                n = int(ref_num[r].item())
                assert torch.equal(cand_sampled[r, :n], ref_sampled[r, :n])

            base_timing = _bench_cuda_us(
                lambda: sampler.apply_sampling_params(
                    raw_logits,
                    expanded_idx_mapping,
                    idx_mapping,
                    idx_mapping_np,
                    pos,
                    draft_sampled,
                    expanded_local_pos,
                )
            )
            cand_timing = _bench_cuda_us(
                lambda: sampler.apply_sampling_params(
                    raw_logits,
                    expanded_idx_mapping,
                    idx_mapping,
                    idx_mapping_np,
                    pos,
                    draft_sampled,
                    expanded_local_pos,
                    dead_row_cut=dead_row_cut,
                    cu_num_logits=cu_num_logits,
                )
            )
            speedup = base_timing["median_us"] / cand_timing["median_us"]
            key = f"normal_B{batch_size}_cut{cut_val}"
            results[key] = {
                "batch_size": batch_size,
                "total_rows": num_rows,
                "live_rows": int(live_mask.sum().item()),
                "cut": cut_val,
                "base_median_us": round(base_timing["median_us"], 2),
                "cand_median_us": round(cand_timing["median_us"], 2),
                "speedup": round(speedup, 3),
                "bit_exact": True,
            }

        # Adversarial top_p only (top_k=0, top_p=0.9) with flat/near-uniform dead rows
        sampler.sampling_states.top_k.np[:batch_size] = VOCAB_SIZE
        sampler.sampling_states.top_k.copy_to_uva()
        adv_logits = raw_logits.clone()
        dead_row_cut = torch.full((batch_size,), 1, dtype=torch.int32, device=device)
        live_mask = expanded_local_pos <= 1
        # Near-uniform Ramp on dead rows forces full binary search iterations in top_p
        ramp = torch.linspace(
            -1.0, 1.0, VOCAB_SIZE, dtype=torch.bfloat16, device=device
        )
        adv_logits[~live_mask] = ramp
        base_adv = _bench_cuda_us(
            lambda: sampler.apply_sampling_params(
                adv_logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
            )
        )
        cand_adv = _bench_cuda_us(
            lambda: sampler.apply_sampling_params(
                adv_logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                draft_sampled,
                expanded_local_pos,
                dead_row_cut=dead_row_cut,
                cu_num_logits=cu_num_logits,
            )
        )
        results[f"top_p_only_flat_dead_B{batch_size}_cut1"] = {
            "batch_size": batch_size,
            "total_rows": num_rows,
            "live_rows": int(live_mask.sum().item()),
            "cut": 1,
            "base_median_us": round(base_adv["median_us"], 2),
            "cand_median_us": round(cand_adv["median_us"], 2),
            "speedup": round(base_adv["median_us"] / cand_adv["median_us"], 3),
            "bit_exact": True,
        }
        sampler.sampling_states.top_k.np[:batch_size] = 64
        sampler.sampling_states.top_k.copy_to_uva()

    return results


def verify_and_bench_sparse_topk_markov(device: torch.device) -> dict:
    torch.manual_seed(99)
    weight_bf16 = (
        torch.randn(VOCAB_SIZE, MARKOV_RANK, device=device) * 0.05
    ).bfloat16()
    packed = pack_bf16_weight(weight_bf16.clone())
    shard = VocabParallelEmbedding._get_indices(
        VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE, VOCAB_SIZE, 0, 1
    )
    w2 = SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=MARKOV_RANK,
        packed_bf16_exponent_base=packed.exponent_base,
        shard_indices=shard,
    )
    w1_table = torch.randn(
        VOCAB_SIZE, MARKOV_RANK, dtype=torch.bfloat16, device=device
    )

    draft = DSparkDeepseekV4ForCausalLM.__new__(DSparkDeepseekV4ForCausalLM)
    object.__setattr__(
        draft,
        "model",
        SimpleNamespace(markov_head=SimpleNamespace(markov_w2=w2)),
    )
    object.__setattr__(draft, "logits_processor", SimpleNamespace(scale=1.0))

    max_reqs = 16
    temperature = torch.linspace(0.5, 1.2, max_reqs, dtype=torch.float32, device=device)
    seeds = torch.arange(500, 500 + max_reqs, dtype=torch.int64, device=device)
    step_cols = torch.arange(N_SPEC, dtype=torch.int32, device=device)

    results: dict[str, dict] = {}
    for batch_size in (1, 2, 4, 8, 16):
        base_logits = torch.randn(
            batch_size, N_SPEC, VOCAB_SIZE, dtype=torch.bfloat16, device=device
        )
        base_values, draft_indices = base_logits.topk(DRAFT_TOPK, dim=-1)
        idx_map = (
            torch.arange(batch_size, dtype=torch.int64, device=device)
            .unsqueeze(1)
            .expand(batch_size, N_SPEC)
            .contiguous()
        )
        sample_pos = (
            20
            + torch.arange(N_SPEC, dtype=torch.int64, device=device)
            .unsqueeze(0)
            .expand(batch_size, N_SPEC)
            .contiguous()
        )
        anchor_ids = torch.randint(
            0, VOCAB_SIZE, (batch_size,), dtype=torch.int64, device=device
        )

        # 1. Greedy mode (draft_logits=None)
        dense_work = base_logits.clone()
        dense_tokens_greedy = torch.empty(
            (batch_size, N_SPEC), dtype=torch.int64, device=device
        )
        sparse_tokens_greedy = torch.empty(
            (batch_size, N_SPEC), dtype=torch.int64, device=device
        )

        def run_dense_greedy():
            dense_work.fill_(float("-inf"))
            prev = anchor_ids
            for i in range(N_SPEC):
                m_embed = w1_table[prev]
                logits_i = draft.apply_markov_bias_gathered(
                    m_embed, dense_work[:, i], base_values[:, i], draft_indices[:, i]
                )
                sampled_i = logits_i.argmax(dim=-1)
                dense_tokens_greedy[:, i] = sampled_i
                prev = sampled_i

        def run_sparse_greedy():
            prev = anchor_ids
            for i in range(N_SPEC):
                m_embed = w1_table[prev]
                cand_i = draft.apply_markov_bias_topk(
                    m_embed, base_values[:, i], draft_indices[:, i]
                )
                sampled_i = tilelang_sampling.topk_greedy_sample(
                    cand_i, draft_indices[:, i], VOCAB_SIZE
                )
                sparse_tokens_greedy[:, i] = sampled_i
                prev = sampled_i

        run_dense_greedy()
        run_sparse_greedy()
        assert torch.equal(sparse_tokens_greedy, dense_tokens_greedy), (
            f"Greedy mismatch at B={batch_size}"
        )

        base_g = _bench_cuda_us(run_dense_greedy)
        cand_g = _bench_cuda_us(run_sparse_greedy)
        results[f"greedy_markov_loop_B{batch_size}"] = {
            "batch_size": batch_size,
            "mode": "greedy",
            "topk": DRAFT_TOPK,
            "base_median_us": round(base_g["median_us"], 2),
            "cand_median_us": round(cand_g["median_us"], 2),
            "speedup": round(base_g["median_us"] / cand_g["median_us"], 3),
            "bit_exact": True,
        }

        # 2. Stochastic mode (draft_logits cached in [max_reqs, N_SPEC, VOCAB_SIZE])
        dense_cache = torch.full(
            (max_reqs, N_SPEC, VOCAB_SIZE),
            float("-inf"),
            dtype=torch.bfloat16,
            device=device,
        )
        sparse_cache = torch.full(
            (max_reqs, N_SPEC, VOCAB_SIZE),
            float("-inf"),
            dtype=torch.bfloat16,
            device=device,
        )
        cached_indices = torch.zeros(
            (max_reqs, N_SPEC, DRAFT_TOPK), dtype=torch.int64, device=device
        )
        dense_tokens_prob = torch.empty(
            (batch_size, N_SPEC), dtype=torch.int64, device=device
        )
        sparse_tokens_prob = torch.empty(
            (batch_size, N_SPEC), dtype=torch.int64, device=device
        )

        def run_dense_prob():
            dense_work.fill_(float("-inf"))
            prev = anchor_ids
            for i in range(N_SPEC):
                m_embed = w1_table[prev]
                logits_i = draft.apply_markov_bias_gathered(
                    m_embed, dense_work[:, i], base_values[:, i], draft_indices[:, i]
                )
                sampled_i = gumbel_sample(
                    logits_i,
                    idx_map[:, i],
                    temperature,
                    seeds,
                    sample_pos[:, i] - 1,
                    apply_temperature=True,
                    is_drafting=True,
                    logits_cache=dense_cache,
                    logits_cache_col=step_cols[i],
                    use_fp64=False,
                )
                dense_tokens_prob[:, i] = sampled_i
                prev = sampled_i

        def run_sparse_prob():
            prev = anchor_ids
            for i in range(N_SPEC):
                m_embed = w1_table[prev]
                cand_i = draft.apply_markov_bias_topk(
                    m_embed, base_values[:, i], draft_indices[:, i]
                )
                sampled_i = tilelang_sampling.topk_gumbel_sample(
                    cand_i,
                    draft_indices[:, i],
                    idx_map[:, i],
                    temperature,
                    seeds,
                    sample_pos[:, i] - 1,
                    True,
                    VOCAB_SIZE,
                    is_drafting=True,
                    logits_cache=sparse_cache,
                    logits_cache_col=step_cols[i],
                    cached_indices=cached_indices,
                )
                sparse_tokens_prob[:, i] = sampled_i
                prev = sampled_i

        run_dense_prob()
        run_sparse_prob()
        assert torch.equal(sparse_tokens_prob, dense_tokens_prob), (
            f"Probabilistic token mismatch at B={batch_size}"
        )
        assert torch.equal(
            sparse_cache[:batch_size].view(torch.int16),
            dense_cache[:batch_size].view(torch.int16),
        ), f"Probabilistic draft_logits cache mismatch at B={batch_size}"

        base_p = _bench_cuda_us(run_dense_prob)
        cand_p = _bench_cuda_us(run_sparse_prob)
        results[f"probabilistic_markov_loop_B{batch_size}"] = {
            "batch_size": batch_size,
            "mode": "probabilistic_cached",
            "topk": DRAFT_TOPK,
            "base_median_us": round(base_p["median_us"], 2),
            "cand_median_us": round(cand_p["median_us"], 2),
            "speedup": round(base_p["median_us"] / cand_p["median_us"], 3),
            "bit_exact": True,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda")
    t0 = time.time()
    dead_row_res = verify_and_bench_dead_row_sampler(device)
    sparse_topk_res = verify_and_bench_sparse_topk_markov(device)
    payload = {
        "boot": args.boot,
        "device": torch.cuda.get_device_name(device),
        "vocab_size": VOCAB_SIZE,
        "n_spec": N_SPEC,
        "draft_topk": DRAFT_TOPK,
        "elapsed_sec": round(time.time() - t0, 2),
        "dead_row_sampler": dead_row_res,
        "sparse_topk_markov": sparse_topk_res,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
