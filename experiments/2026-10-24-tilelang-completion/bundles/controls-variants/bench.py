#!/usr/bin/env python3
"""Min-p's free shapes, measured: which tile and block of the tile-form kernel beats Triton.

The generation controls' min-p (#524) took one row a block and loaded each element behind
its own bound check, so ptxas issued a tile's loads in small groups between branches; at
96 rows it was 8% slower than Triton. The tile-form kernel loads a whole tile before using
it (as 16-byte vectors when the rows are aligned) and handles only the last, partial tile
with clamped loads and predicated stores. This sweeps its run (4: vectors, 1: scalars),
threads a block and tile width. Every variant is first compared with Triton bit for bit:
the threshold-boundary seeds 48291-48298 at 96 rows, and rows of -inf, NaN and +inf. Then
it is timed against Triton at 1, 16 and 96 rows of 129280 columns: eight samples, each a
CUDA graph of eight calls on their own fresh logits, replayed as a pair whose order
alternates. One JSON line per variant and shape, then a summary.
"""
import json
import math
import statistics

import torch

from vllm.v1.worker.gpu.sample import tilelang_controls as candidate
from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel

VOCAB = 129280
CALLS = 8
SAMPLES = 8
VARIANTS = [
    (run, threads, tile)
    for run in (4, 1)
    for threads in (128, 256)
    for tile in (1024, 2048, 4096)
    if tile % (threads * run) == 0
]


def ours(variant):
    run, threads, tile = variant

    def apply(logits, mapping, thresholds):
        assert run == 1 or candidate._min_p_run(logits) == 4
        kernel = candidate.min_p_kernel(
            logits.shape[1], logits.stride(0), candidate._DTYPES[mapping.dtype],
            run, threads, tile)
        kernel(logits, mapping, thresholds)

    return apply


def triton(logits, mapping, thresholds):
    _min_p_kernel[(logits.shape[0],)](logits, logits.stride(0), mapping, thresholds,
                                      logits.shape[1], BLOCK_SIZE=1024)


def same_bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def exact(apply):
    """Seeds 48291-48298 at 96 rows, then rows of -inf, NaN and +inf."""
    for seed in range(48291, 48299):
        torch.manual_seed(seed)
        mapping = torch.randperm(105, device="cuda")[:96]
        logits = torch.randn((96, VOCAB), device="cuda")
        thresholds = torch.full((105,), 0.1, device="cuda")
        thresholds[::5] = 0
        expected = logits.clone()
        triton(expected, mapping, thresholds)
        apply(logits, mapping, thresholds)
        if not same_bits(logits, expected):
            return f"seed {seed}"
    torch.manual_seed(18093)
    logits = torch.randn((6, VOCAB), device="cuda")
    mapping = torch.randperm(13, device="cuda")[:6]
    thresholds = torch.full((13,), 0.1, device="cuda")
    logits[0].fill_(-float("inf"))
    logits[1, 0] = float("nan")
    logits[2, -1] = float("inf")
    logits[3].fill_(float("nan"))
    logits[4, :1024] = float("nan")
    expected = logits.clone()
    triton(expected, mapping, thresholds)
    apply(logits, mapping, thresholds)
    return None if same_bits(logits, expected) else "non-finite rows"


def case(rows, seed):
    torch.manual_seed(seed)
    states = rows + 9
    mapping = torch.randperm(states, device="cuda")[:rows]
    source = torch.randn((rows, VOCAB), device="cuda")
    buffers = [source.clone() for _ in range(CALLS)]
    thresholds = torch.full((states,), 0.1, device="cuda")
    thresholds[::5] = 0

    def reset():
        for logits in buffers:
            logits.copy_(source)

    return reset, buffers, mapping, thresholds


def capture(apply, buffers, mapping, thresholds):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        apply(buffers[0], mapping, thresholds)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for logits in buffers:
            apply(logits, mapping, thresholds)
    return graph


def elapsed(graph, reset):
    reset()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / CALLS


def measure(variant, rows):
    apply = ours(variant)
    mine, theirs = [], []
    for sample in range(SAMPLES):
        reset, buffers, mapping, thresholds = case(rows, 48291 + sample)
        reset()
        a = capture(apply, buffers, mapping, thresholds)
        reset()
        b = capture(triton, buffers, mapping, thresholds)
        order = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(graph): elapsed(graph, reset) for graph in order}
        mine.append(times[id(a)])
        theirs.append(times[id(b)])
        del a, b, order, buffers
    paired = [x - y for x, y in zip(mine, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    return dict(run=variant[0], threads=variant[1], tile=variant[2], rows=rows,
                tilelang_us=round(statistics.median(mine), 2),
                triton_us=round(statistics.median(theirs), 2),
                change_pct=round(100 * mean / statistics.fmean(theirs), 2),
                interval_us=[round(mean - half, 2), round(mean + half, 2)],
                slower=mean - half > 0, faster=mean + half < 0)


def main():
    results, inexact = [], []
    for variant in VARIANTS:
        problem = exact(ours(variant))
        print(json.dumps({"variant": variant, "exact": problem is None,
                          "problem": problem}), flush=True)
        if problem:
            inexact.append(variant)
            continue
        for rows in (1, 16, 96):
            result = measure(variant, rows)
            results.append(result)
            print(json.dumps(result), flush=True)
    passing = [v for v in VARIANTS if v not in inexact and not any(
        r["slower"] for r in results if (r["run"], r["threads"], r["tile"]) == v)]
    print(json.dumps({"summary": "min-p variants", "inexact": inexact,
                      "never_slower": passing}), flush=True)


if __name__ == "__main__":
    main()
