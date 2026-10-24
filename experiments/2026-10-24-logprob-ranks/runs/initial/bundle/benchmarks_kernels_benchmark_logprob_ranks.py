# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired rank timings against Triton: the same inputs and output, alternating order.

Eight fresh allocations per shape, warmed CUDA graphs of 100 calls, and a 95%
interval over paired differences. --gate rejects any supported serving shape
whose interval shows a slowdown. --threads compares fixed kernel configurations.
"""

import argparse
import json
import math
import statistics

import torch

from vllm.v1.worker.gpu.sample import logprob, tilelang_logprob


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
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    capture.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 10


def measure(rows, vocab, dtype, threads):
    ours, theirs, differences = [], [], []
    for sample in range(8):
        torch.manual_seed(4100 + sample)
        # Change allocation placement without changing the values each side reads.
        spacer = torch.empty((sample * 513 + 17) * 256, device="cuda")
        logits = torch.randn(rows, vocab, dtype=dtype, device="cuda")
        ids = torch.randint(vocab, (rows,), device="cuda", dtype=torch.int64)
        output = torch.empty(rows, device="cuda", dtype=torch.int64)
        kernel = tilelang_logprob.ranks_kernel(
            vocab, vocab, tilelang_logprob._DTYPES[dtype], "int64", threads
        )

        def original(output=output, logits=logits, ids=ids):
            logprob._ranks_kernel[(rows,)](
                output, logits, logits.stride(0), ids, vocab, BLOCK_SIZE=8192
            )

        def candidate(kernel=kernel, logits=logits, ids=ids, output=output):
            kernel(logits, ids, output)

        original()
        expected = output.clone()
        candidate()
        if not torch.equal(output, expected):
            raise RuntimeError(f"ranks differ: rows={rows}, dtype={dtype}")
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
        "rows": rows,
        "vocab": vocab,
        "dtype": str(dtype),
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
    parser.add_argument("--threads", default="256")
    parser.add_argument("--rows", default="1,8,16,48,96")
    parser.add_argument("--dtypes", default="float32,bfloat16,float16")
    args = parser.parse_args()
    results = [
        measure(rows, 129280, getattr(torch, dtype), threads)
        for threads in map(int, args.threads.split(","))
        for dtype in args.dtypes.split(",")
        for rows in map(int, args.rows.split(","))
    ]
    slower = [[r["dtype"], r["rows"], r["threads"]] for r in results if r["slower"]]
    print(json.dumps({"summary": "rank benchmark", "slower": slower}), flush=True)
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
