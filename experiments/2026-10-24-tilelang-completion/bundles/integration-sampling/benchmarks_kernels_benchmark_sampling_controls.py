# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired CUDA graph timings of V2 controls, with fresh inputs per graph node.

In-place filtering and penalties must not feed their own outputs into later
iterations: that changes both the workload and the branch paths. Each of the
graph's calls gets its own initialized logits/output buffers. Copies and resets
are outside event timing. Both implementations see the same tensor addresses.
"""

import argparse
import json
import math
import statistics

import torch

from vllm.v1.worker.gpu.sample import tilelang_controls as candidate
from vllm.v1.worker.gpu.sample.min_p import _min_p_kernel
from vllm.v1.worker.gpu.sample.penalties import _bincount_kernel, _penalties_kernel


def make_case(kind, rows, width, seed, count):
    torch.manual_seed(seed)
    states = rows + 9
    mapping = torch.randperm(states, device="cuda")[:rows]
    if kind.startswith("bincount"):
        tokens = torch.randint(
            129280, (states, width), device="cuda", dtype=torch.int32
        )
        if kind == "bincount_repeated":
            tokens[:, ::3] = 31
        prompt = torch.full((states,), width // 2, device="cuda", dtype=torch.int32)
        prefill = torch.full_like(prompt, width)
        buffers = [
            (
                torch.zeros((states, 4040), device="cuda", dtype=torch.int32),
                torch.zeros((states, 129280), device="cuda", dtype=torch.int32),
            )
            for _ in range(count)
        ]

        if kind == "bincount_clear":
            # The runner's int32 request indices (the torch clear widened them).
            mapping = mapping.int()

        def reset():
            for mask, counts in buffers:
                if kind == "bincount_clear":
                    mask.fill_(-1)
                    counts.fill_(5)
                else:
                    mask.zero_()
                    counts.zero_()

        def invoke(index, ours):
            mask, counts = buffers[index]
            if kind == "bincount_clear":
                if ours:
                    candidate.clear_bincount_rows(mapping, mask, counts)
                else:
                    idx_long = mapping.long()
                    mask.index_fill_(0, idx_long, 0)
                    counts.index_fill_(0, idx_long, 0)
            elif ours:
                candidate.accumulate_bincount(
                    mapping, tokens, prompt, prefill, mask, counts, width
                )
            else:
                _bincount_kernel[(rows, (width + 1023) // 1024)](
                    mapping,
                    tokens,
                    tokens.stride(0),
                    prompt,
                    prefill,
                    mask,
                    mask.stride(0),
                    counts,
                    counts.stride(0),
                    BLOCK_SIZE=1024,
                )

        return reset, invoke, lambda i: buffers[i]

    source = torch.randn((rows, width), device="cuda")
    buffers = [source.clone() for _ in range(count)]
    thresholds = torch.full((states,), 0.1, device="cuda")
    thresholds[::5] = 0
    positions = (torch.arange(rows, device="cuda") % 6).int()
    if kind == "penalties":
        mapping = (torch.arange(rows, device="cuda") // 6 + 2).long()
    tokens = torch.randint(width, (rows,), device="cuda", dtype=torch.int32)
    rep = torch.full((states,), 1.3, device="cuda")
    freq = torch.full_like(rep, 0.25)
    pres = torch.full_like(rep, -0.1)
    mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (states, (width + 31) // 32),
        device="cuda",
        dtype=torch.int32,
    )
    counts = torch.randint(0, 8, (states, width), device="cuda", dtype=torch.int32)

    def reset():
        for logits in buffers:
            logits.copy_(source)

    def invoke(index, ours):
        logits = buffers[index]
        if kind == "min_p":
            if ours:
                candidate.apply_min_p(logits, mapping, thresholds)
            else:
                return _min_p_kernel[(rows,)](
                    logits,
                    logits.stride(0),
                    mapping,
                    thresholds,
                    width,
                    BLOCK_SIZE=1024,
                )
        elif ours:
            candidate.apply_penalties(
                logits, mapping, tokens, positions, rep, freq, pres, mask, counts
            )
        else:
            _penalties_kernel[(rows, (width + 8191) // 8192)](
                logits,
                logits.stride(0),
                mapping,
                tokens,
                positions,
                rep,
                freq,
                pres,
                mask,
                mask.stride(0),
                counts,
                counts.stride(0),
                width,
                BLOCK_SIZE=8192,
            )

    return reset, invoke, lambda i: (buffers[i],)


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


def measure(kind, rows, width, seed):
    count = 8
    ours, theirs = [], []
    for sample in range(8):
        reset, invoke, output = make_case(kind, rows, width, seed + sample, count)
        reset()
        compiled = invoke(0, False)
        expected = [value.clone() for value in output(0)]
        reset()
        invoke(0, True)
        if any(not torch.equal(a, b) for a, b in zip(output(0), expected)):
            if compiled is not None:
                ptx = compiled.asm["ptx"]
                print(
                    json.dumps(
                        {
                            "ptx_arithmetic": [
                                line.strip()
                                for line in ptx.splitlines()
                                if any(
                                    op in line
                                    for op in (
                                        "lg2.",
                                        "fma.",
                                        "max.",
                                        "mul.f32",
                                        "add.f32",
                                    )
                                )
                            ]
                        }
                    ),
                    flush=True,
                )
            for actual, target in zip(output(0), expected):
                indices = (actual != target).nonzero()[:8]
                print(
                    json.dumps(
                        {
                            "sample": sample,
                            "differences": [
                                [
                                    index.tolist(),
                                    actual[tuple(index)].item(),
                                    target[tuple(index)].item(),
                                ]
                                for index in indices
                            ],
                        }
                    ),
                    flush=True,
                )
            raise RuntimeError(f"outputs differ: {kind} rows={rows} width={width}")
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
        kernel=kind,
        rows=rows,
        width=width,
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
    parser.add_argument(
        "--kernels",
        default="min_p,penalties,bincount,bincount_repeated,bincount_clear",
    )
    args = parser.parse_args()
    cases = [
        (kind, rows, width)
        for kind in args.kernels.split(",")
        if kind in ("min_p", "penalties")
        for rows in (1, 16, 96)
        for width in (129280,)
    ]
    for kind in args.kernels.split(","):
        if not kind.startswith("bincount"):
            continue
        # The clear's work is the vocabulary's, whatever the token history's length.
        widths = (128,) if kind == "bincount_clear" else (128, 8192, 131072)
        cases += [(kind, rows, width) for rows in (1, 16) for width in widths]
    results = [measure(*case, args.seed) for case in cases]
    failed = [(r["kernel"], r["rows"], r["width"]) for r in results if r["slower"]]
    print(json.dumps({"summary": "sampling controls", "slower": failed}), flush=True)
    if args.gate and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
