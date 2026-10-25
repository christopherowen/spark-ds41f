# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CUDA graph timings of the V2 top-k / top-p filter, TileLang against
Triton, with fresh inputs per graph node.

The filter masks its logits in place, so no call may see an earlier call's output:
that changes both the workload and the branch paths. Each of the graph's calls gets
its own initialized logits. Copies are outside event timing. Both implementations
see the same tensor addresses. Before timing, each sample checks the TileLang
output against Triton's bit for bit, padding included.

Rows come six to a request (five drafts), each request with its own k and p. Row
counts cover each slice count of Triton's split top-p pipeline on 48 SMs (1, 2, 4,
8, 16 and 32 rows: 32, 16, 8, 4, 2 and 1 slices; 6, 12 and 24 rows at the upper
ends) and the monolithic kernel above 64 rows. Modes: top-k alone, top-p alone,
both, and ``mixed`` (requests with top-k alone, top-p alone and both).
"""

import argparse
import json
import math
import statistics

import torch

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
from vllm.v1.worker.gpu.sample import tilelang_topk_topp as candidate

WIDTH = 6  # Rows a request: one target and five drafts.
TOP_K = (20, 50, 64, 100, 40)
TOP_P = (0.9, 0.95, 0.8, 0.92)


def make_controls(mode, rows, vocab):
    """Per-row k and p, the same across a request's rows."""
    request = torch.arange(rows, device="cuda") // WIDTH
    k = torch.tensor(TOP_K, device="cuda", dtype=torch.int32)[request % len(TOP_K)]
    p = torch.tensor(TOP_P, device="cuda")[request % len(TOP_P)]
    if mode == "mixed":
        # Requests in threes: top-k alone, top-p alone, both.
        k = torch.where(request % 3 == 1, vocab, k).to(torch.int32)
        p = torch.where(request % 3 == 0, 1.0, p)
    return (None if mode == "p" else k), (None if mode == "k" else p)


def make_logits(rows, vocab, generator):
    """Model-like rows: a Gaussian body with a head of a few dozen tokens well
    above it, of varied height."""
    logits = torch.randn((rows, vocab), device="cuda", generator=generator) * 1.5
    head = torch.randint(vocab, (rows, 32), device="cuda", generator=generator)
    lift = torch.rand((rows, 32), device="cuda", generator=generator) * 10 + 4
    return logits.scatter_add_(1, head, lift)


def make_case(mode, rows, vocab, stride, seed, count):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = make_logits(rows, vocab, generator)
    storage = [
        torch.empty((rows, stride), device="cuda").fill_(-7.0) for _ in range(count)
    ]
    buffers = [buffer[:, :vocab] for buffer in storage]
    k, p = make_controls(mode, rows, vocab)

    def reset():
        for logits in buffers:
            logits.copy_(source)

    def invoke(index, ours):
        if ours:
            candidate.apply_top_k_top_p(buffers[index], k, p)
        else:
            apply_top_k_top_p_triton(buffers[index], k, p)

    return reset, invoke, lambda i: storage[i]


def capture(invoke, ours, count):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        invoke(0, ours)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(count):
            invoke(index, ours)
    return graph


def elapsed(graph, reset, count):
    reset()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / count


def measure(mode, rows, vocab, stride, seed):
    count = 8
    ours, theirs = [], []
    for sample in range(8):
        reset, invoke, output = make_case(
            mode, rows, vocab, stride, seed + sample, count
        )
        reset()
        invoke(0, False)
        expected = output(0).clone()
        reset()
        invoke(0, True)
        actual = output(0)
        if not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
            indices = (actual.view(torch.int32) != expected.view(torch.int32)).nonzero()
            print(
                json.dumps(
                    {
                        "sample": sample,
                        "count": indices.shape[0],
                        "differences": [
                            [
                                index.tolist(),
                                actual[tuple(index)].item(),
                                expected[tuple(index)].item(),
                            ]
                            for index in indices[:8]
                        ],
                    }
                ),
                flush=True,
            )
            raise RuntimeError(
                f"outputs differ: {mode} rows={rows} vocab={vocab} stride={stride}"
            )
        reset()
        a = capture(invoke, True, count)
        reset()
        b = capture(invoke, False, count)
        order = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(graph): elapsed(graph, reset, count) for graph in order}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
        del a, b, order, reset, invoke, output, expected
    paired = [a - b for a, b in zip(ours, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    result = dict(
        mode=mode,
        rows=rows,
        vocab=vocab,
        stride=stride,
        tilelang_us=statistics.median(ours),
        triton_us=statistics.median(theirs),
        interval_us=[mean - half, mean + half],
        slower=mean - half > 0,
        tilelang_samples_us=ours,
        triton_samples_us=theirs,
    )
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--seed", type=int, default=48291)
    parser.add_argument("--modes", default="k,p,kp,mixed")
    parser.add_argument("--rows", default="1,2,4,6,8,12,16,24,32,64,96")
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument(
        "--stride", type=int, default=0, help="row stride (default: the vocabulary)"
    )
    args = parser.parse_args()
    stride = args.stride or args.vocab
    cases = [
        (mode, int(rows), args.vocab, stride)
        for mode in args.modes.split(",")
        for rows in args.rows.split(",")
    ]
    results = [measure(*case, args.seed) for case in cases]
    failed = [(r["mode"], r["rows"]) for r in results if r["slower"]]
    print(json.dumps({"summary": "top-k / top-p", "slower": failed}), flush=True)
    if args.gate and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
