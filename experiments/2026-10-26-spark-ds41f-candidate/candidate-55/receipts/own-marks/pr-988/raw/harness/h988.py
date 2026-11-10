"""#988 dead-row part (0794): Sampler.apply_sampling_params on a verification step.

usage: python3 /h/h988.py --arm base|c55dr --boot K --out FILE

The PR's screen_dead_row_and_sparse_topk.py (cfb7069e), dead-row cells only, with its
sampling parameters (temperature 0.8, top_k 64, top_p 0.9, min_p 0.05, repetition 1.15,
frequency 0.2, presence 0.3, logit_bias {7: 1.5, 12: -2.0}, bad words [[15, 16], [25]]),
V = 129,280 and 6 verification rows a request. Each arm times the call its own
RejectionSampler._verify makes: candidate-54 processes every row (no dead_row_cut
argument exists there); 0794 passes dead_row_cut and cu_num_logits. The candidate arm
also times its own call without the cut (the PR's in-process base) and checks live rows
bit-exact against it with every dead row's input set to NaN, and rejection_sample's
outputs equal. Live-row hashes let the arms be compared with each other.

The sampler runs eagerly in serving (it is outside the CUDA graphs), so the serving path's
time is the eager call's stream time (events around each call, the PR's method); CUPTI
kernel time is recorded beside it.
"""

import argparse
import json

import numpy as np
import torch

from omlib import env_info, gpu_warmup, h, kernel_us

VOCAB_SIZE = 129280
N_SPEC = 5
ROWS = N_SPEC + 1


def bench_events(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    es = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ss[i].record()
        fn()
        es[i].record()
    torch.cuda.synchronize()
    arr = np.array([s.elapsed_time(e) * 1000.0 for s, e in zip(ss, es)])
    return {"median_us": round(float(np.median(arr)), 2), "p10_us": round(float(np.percentile(arr, 10)), 2),
            "p90_us": round(float(np.percentile(arr, 90)), 2), "mean_us": round(float(arr.mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cand = args.arm != "base"
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda", 0)
    warm = gpu_warmup()
    max_num_reqs = 16
    req_states = RequestState(max_num_reqs=max_num_reqs, max_model_len=128,
                              max_num_batched_tokens=max_num_reqs * ROWS, num_speculative_steps=N_SPEC,
                              vocab_size=VOCAB_SIZE, device=device)
    sampler = Sampler(max_num_reqs=max_num_reqs, vocab_size=VOCAB_SIZE, device=device, req_states=req_states)
    for s_idx in range(max_num_reqs):
        req_states.add_request(req_id=f"req_{s_idx}", prompt_len=8, all_token_ids=list(range(10, 26)),
                               num_computed_tokens=16, max_tokens=64)
        slot = req_states.req_id_to_index[f"req_{s_idx}"]
        params = SamplingParams(temperature=0.8, top_k=64, top_p=0.9, min_p=0.05, repetition_penalty=1.15,
                                frequency_penalty=0.2, presence_penalty=0.3, logit_bias={7: 1.5, 12: -2.0},
                                seed=1000 + slot)
        params._bad_words_token_ids = [[15, 16], [25]]
        sampler.add_request(slot, 8, params)
    req_states.apply_staged_writes()
    sampler.apply_staged_writes()

    out = {"arm": args.arm, "boot": args.boot, "env": env_info(), "gpu_warmup_ms": warm, "cells": {}}
    for batch in (1, 4, 8, 16):
        n = batch * ROWS
        idx_np = np.arange(batch, dtype=np.int64)
        idx = torch.from_numpy(idx_np).to(device)
        exp_idx = torch.arange(batch, dtype=torch.int64, device=device).repeat_interleave(ROWS).contiguous()
        exp_pos = torch.arange(ROWS, dtype=torch.int32, device=device).repeat(batch).contiguous()
        cu = torch.arange(0, (batch + 1) * ROWS, ROWS, dtype=torch.int32, device=device)
        pos = 16 + torch.arange(ROWS, dtype=torch.int64, device=device).repeat(batch)
        torch.manual_seed(4242 + batch)
        draft = torch.randint(1, 1000, (n,), dtype=torch.int32, device=device)
        torch.manual_seed(42 + batch)
        raw = torch.randn((n, VOCAB_SIZE), dtype=torch.bfloat16, device=device)
        ramp = torch.linspace(-1.0, 1.0, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
        cases = [("normal", 1), ("normal", 2), ("top_p_only_flat_dead", 1)]
        for kind, cut_val in cases:
            if kind == "top_p_only_flat_dead":
                sampler.sampling_states.top_k.np[:batch] = VOCAB_SIZE
                sampler.sampling_states.top_k.copy_to_uva()
            cut = torch.full((batch,), cut_val, dtype=torch.int32, device=device)
            live = exp_pos <= cut_val
            logits = raw.clone()
            if kind == "top_p_only_flat_dead":
                logits[~live] = ramp

            def base_call():
                return sampler.apply_sampling_params(logits, exp_idx, idx, idx_np, pos, draft, exp_pos)

            def cut_call(inp=logits):
                return sampler.apply_sampling_params(inp, exp_idx, idx, idx_np, pos, draft, exp_pos,
                                                     dead_row_cut=cut, cu_num_logits=cu)

            key = f"{kind}_B{batch}_cut{cut_val}"
            cell = {"batch": batch, "rows": n, "live_rows": int(live.sum().item()), "cut": cut_val}
            ref = base_call()
            cell["live_hash_nocut"] = h(ref[live])
            if cand:
                nan_in = logits.clone()
                nan_in[~live] = float("nan")
                got = cut_call(nan_in)
                cell["live_bit_exact_vs_nocut_nan_dead"] = bool(torch.equal(got[live], ref[live]))
                cell["live_hash_cut"] = h(got[live])
                rs = []
                for proc in (ref, got):
                    rs.append(rejection_sample(proc, None, draft, cu, pos, idx, exp_idx, exp_pos,
                                               sampler.sampling_states.temperature.gpu,
                                               sampler.sampling_states.seeds.gpu, N_SPEC, None, use_fp64=False,
                                               use_block_verification=True, dead_row_cut=cut, kernel_steps=N_SPEC))
                same_num = torch.equal(rs[0][1], rs[1][1])
                same_tok = same_num and all(
                    torch.equal(rs[0][0][r, : int(rs[0][1][r])], rs[1][0][r, : int(rs[0][1][r])]) for r in range(batch))
                cell["rejection_sample_equal"] = bool(same_num and same_tok)
                cell["serving_call"] = "dead_row_cut"
                cell["serving_events"] = bench_events(cut_call)
                cell["serving_kernel"] = kernel_us(cut_call)
                cell["nocut_events"] = bench_events(base_call)
                cell["nocut_kernel"] = kernel_us(base_call)
            else:
                cell["serving_call"] = "all rows (candidate-54)"
                cell["serving_events"] = bench_events(base_call)
                cell["serving_kernel"] = kernel_us(base_call)
            for k in ("serving_kernel", "nocut_kernel"):
                if k in cell:
                    cell[k].pop("kernels", None)
            out["cells"][key] = cell
            print(key, json.dumps({k: v for k, v in cell.items() if k.endswith(("events", "kernel", "exact", "equal"))}),
                  flush=True)
            if kind == "top_p_only_flat_dead":
                sampler.sampling_states.top_k.np[:batch] = 64
                sampler.sampling_states.top_k.copy_to_uva()
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
