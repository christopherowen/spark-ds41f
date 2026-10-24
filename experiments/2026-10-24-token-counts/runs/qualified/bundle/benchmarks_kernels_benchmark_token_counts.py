# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired graph timings for DS41F's post-verification counts, against Triton.

Eight allocations per case, alternating measurement order, and a paired 95%
interval. --gate rejects a slowdown; --threads screens fixed launch layouts.
"""

import argparse
import json
import math
import statistics

import torch

from vllm.v1.worker.gpu import tilelang_token_counts
from vllm.v1.worker.gpu.input_batch import _get_num_sampled_and_rejected_kernel


def graph(call):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(100):
            call()
    capture.replay()
    torch.cuda.synchronize()
    return capture


def elapsed(capture):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    capture.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 10


def measure(requests, dead_rows, threads, seed):
    ours, theirs, differences = [], [], []
    for sample in range(8):
        torch.manual_seed(seed + sample)
        spacer = torch.empty((sample * 513 + 17) * 256, device="cuda")
        capacity = requests + 19
        idx = torch.randperm(capacity, device="cuda")[:requests]
        prefill = torch.randint(
            1, 1048577, (capacity,), device="cuda", dtype=torch.int32
        )
        seq = prefill[idx] + 1
        if dead_rows:
            seq[::3] -= 2
        widths = torch.full((requests,), 6, device="cuda", dtype=torch.int32)
        cu = torch.cat((widths.new_zeros(1), widths.cumsum(0).int()))
        sampled = torch.randint(1, 7, (requests,), device="cuda", dtype=torch.int32)
        rejected = torch.empty_like(sampled)
        cut = torch.randint(0, 6, (requests,), device="cuda", dtype=torch.int32)
        kernel = tilelang_token_counts.counts_kernel(dead_rows, "int64", threads)

        inputs = (sampled, rejected, seq, cu, idx, prefill, cut)

        def original(inputs=inputs):
            _get_num_sampled_and_rejected_kernel[(requests,)](
                *inputs, DEAD_ROWS=dead_rows
            )

        def candidate(kernel=kernel, inputs=inputs):
            kernel(*inputs)

        saved = sampled.clone()
        original()
        expected_sampled, expected_rejected = sampled.clone(), rejected.clone()
        sampled.copy_(saved)
        candidate()
        if not (
            torch.equal(sampled, expected_sampled)
            and torch.equal(rejected, expected_rejected)
        ):
            raise RuntimeError(f"counts differ: requests={requests}, dead={dead_rows}")
        a, b = graph(candidate), graph(original)
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(c): elapsed(c) for c in pair}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
        del a, b, pair, spacer
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    result = {
        "requests": requests,
        "dead_rows": dead_rows,
        "threads": threads,
        "tilelang_us": statistics.median(ours),
        "triton_us": statistics.median(theirs),
        "change_pct": 100 * mean / statistics.fmean(theirs),
        "interval_us": [mean - half, mean + half],
        "slower": mean - half > 0,
        "tilelang_samples_us": ours,
        "triton_samples_us": theirs,
    }
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--threads", default=str(tilelang_token_counts._THREADS))
    parser.add_argument("--seed", type=int, default=4135)
    parser.add_argument("--requests", default="1,8,16,128,257")
    args = parser.parse_args()
    results = [
        measure(requests, dead_rows, threads, args.seed)
        for threads in map(int, args.threads.split(","))
        for dead_rows in (False, True)
        for requests in map(int, args.requests.split(","))
    ]
    slower = [
        [r["requests"], r["dead_rows"], r["threads"]] for r in results if r["slower"]
    ]
    print(
        json.dumps({"summary": "token counts benchmark", "slower": slower}), flush=True
    )
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
