# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired graph timings for DS41F's resampled-token insertion, against Triton.

With --dead-rows, the insertion that caps each count at its dead-row cut plus one,
against what the rejection sampler launched before it took the cap: the uncapped
insertion, then torch's ``dead_row_cut + 1`` and ``torch.minimum``.

Eight allocations per case, alternating measurement order, and a paired 95%
interval. --gate rejects a slowdown; --threads screens fixed launch layouts.
"""

import argparse
import json
import math
import statistics

import torch

from vllm.v1.worker.gpu.spec_decode import tilelang_resampled as port
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _insert_resampled_kernel,
)


def graph(call, reset):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            reset()
            call(0)
    torch.cuda.current_stream().wait_stream(side)
    reset()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for i in range(100):
            call(i)
    capture.replay()
    torch.cuda.synchronize()
    return capture


def elapsed(capture, reset):
    # Every call has its own count buffer. Reset outside the timed graph so
    # insertion always uses a valid slot, without charging either kernel a copy.
    reset()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    capture.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 10


def measure(requests, blocks, threads, seed, mixed, dead_rows=False):
    ours, theirs, differences = [], [], []
    for sample in range(8):
        torch.manual_seed(seed + sample)
        spacer = torch.empty((sample * 513 + 17) * 256, device="cuda")
        sampled = torch.full((requests, 6), -1, device="cuda", dtype=torch.int64)
        initial = torch.randint(0, 6, (requests,), device="cuda", dtype=torch.int32)
        counts = initial.repeat(100, 1)
        count_views = list(counts.unbind())
        ids = torch.randint(0, 129280, (requests, blocks), device="cuda")
        values = torch.randn(requests, blocks, device="cuda")
        cu = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * 6
        mapping = torch.randperm(requests + 17, device="cuda")[
            :requests
        ].repeat_interleave(6)
        temp = torch.ones(requests + 17, device="cuda")
        if mixed:
            temp[mapping[::12]] = 0
        kernel = port.insert_kernel(
            blocks, "float32", "int64", threads, 6, blocks, blocks, dead_rows
        )
        # Cuts from -1 to past the last row.
        cut = torch.randint(-1, 7, (requests,), device="cuda", dtype=torch.int32)
        capped = torch.empty_like(counts)
        capped_views = list(capped.unbind())
        uncapped = port.insert_kernel(
            blocks, "float32", "int64", threads, 6, blocks, blocks, False
        )

        def torch_capped(
            i,
            uncapped=uncapped,
            sampled=sampled,
            count_views=count_views,
            capped_views=capped_views,
            ids=ids,
            values=values,
            cu=cu,
            mapping=mapping,
            temp=temp,
            cut=cut,
        ):
            uncapped(
                sampled, count_views[i], ids, values, cu, mapping, temp, count_views[i]
            )
            torch.minimum(count_views[i], cut + 1, out=capped_views[i])

        def triton_insert(
            i,
            sampled=sampled,
            count_views=count_views,
            ids=ids,
            values=values,
            cu=cu,
            mapping=mapping,
            temp=temp,
        ):
            _insert_resampled_kernel[(requests,)](
                sampled,
                6,
                count_views[i],
                ids,
                blocks,
                values,
                blocks,
                blocks,
                cu,
                mapping,
                temp,
                PADDED_RESAMPLE_NUM_BLOCKS=1 << (blocks - 1).bit_length(),
            )

        def candidate(
            i,
            kernel=kernel,
            sampled=sampled,
            count_views=count_views,
            ids=ids,
            values=values,
            cu=cu,
            mapping=mapping,
            temp=temp,
            cut=cut,
        ):
            kernel(
                sampled,
                count_views[i],
                ids,
                values,
                cu,
                mapping,
                temp,
                cut if dead_rows else count_views[i],
            )

        def reset(counts=counts, initial=initial):
            counts.copy_(initial.expand_as(counts))

        original = torch_capped if dead_rows else triton_insert
        original(0)
        expected = sampled.clone()
        expected_counts = (capped if dead_rows else counts)[0].clone()
        sampled.fill_(-1)
        reset()
        candidate(0)
        if not (
            torch.equal(sampled, expected) and torch.equal(counts[0], expected_counts)
        ):
            raise RuntimeError("resampled tokens or counts differ")
        a, b = graph(candidate, reset), graph(original, reset)
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(c): elapsed(c, reset) for c in pair}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
        del a, b, pair, spacer
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    result = {
        "requests": requests,
        "blocks": blocks,
        "mixed": mixed,
        "dead_rows": dead_rows,
        # The "triton_us" timings are the reference's: with dead_rows, the uncapped
        # insertion and torch's cap.
        "reference": "insertion + torch cap" if dead_rows else "triton",
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
    parser.add_argument("--threads", default="auto")
    parser.add_argument("--seed", type=int, default=5791)
    parser.add_argument("--requests", default="1,8,16,96")
    parser.add_argument("--blocks", default="3,4,127")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--dead-rows", action="store_true")
    args = parser.parse_args()
    layouts = (
        [None] if args.threads == "auto" else list(map(int, args.threads.split(",")))
    )
    results = [
        measure(
            requests,
            blocks,
            (1 if blocks <= 4 else port._THREADS) if threads is None else threads,
            args.seed,
            args.mixed,
            args.dead_rows,
        )
        for threads in layouts
        for blocks in map(int, args.blocks.split(","))
        for requests in map(int, args.requests.split(","))
        if not (threads == 1 and blocks > 4)
    ]
    slower = [
        [r["requests"], r["blocks"], r["threads"]] for r in results if r["slower"]
    ]
    print(
        json.dumps({"summary": "resampled-token benchmark", "slower": slower}),
        flush=True,
    )
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
