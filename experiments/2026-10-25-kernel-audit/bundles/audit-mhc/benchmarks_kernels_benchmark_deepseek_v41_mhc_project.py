# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of DeepSeek-V4.1's mHC sublayer entry projection
(``tilelang.mhc.project_streams``), whose fn tiles now load before they are needed,
against the kernel as it was at vLLM 26382f5cf
(``tests/kernels/deepseek_v41_baselines.py``).

At a decode step each CTA used to wait for an L2 round trip of its fn tile before
each of the four streams' GEMMs. Shapes: decode steps of one, eight and 16 requests
(1 + 5 drafted rows each) and 8K and 16K prefill chunks, for every entry: the
fused ``post_pre`` (most layers), ``pre`` (after a standalone post), ``broadcast``
(the first layer) and ``post_pre`` of four rank parts (prefill sequence
parallelism).

Each shape is measured on eight fresh inputs. On each, both kernels' outputs are
first checked bit for bit. Then CUDA graphs of 100 calls of both run back to back,
alternating which goes first, and the shape's 95% interval comes from the eight
per-input differences. The gate (``--gate``) is that the new kernel is slower at
no shape (the interval's lower bound above zero).
"""

import argparse
import json
import math
import statistics

import torch

from tests.kernels.deepseek_v41_baselines import project_streams_26382f5cf
from vllm.models.deepseek_v4_1.tilelang.mhc import (
    BLOCK_H,
    BLOCK_M,
    MIXES,
    project_streams,
)

DEVICE = "cuda"
CALLS = 100
HIDDEN = 5120
TOKENS = {
    "decode c1": 6,
    "decode c8": 48,
    "decode c16": 96,
    "prefill 8K": 8192,
    "prefill 16K": 16384,
}
ENTRIES = {
    "post_pre": ("post_pre", 1),
    "pre": ("pre", 1),
    "broadcast": ("broadcast", 1),
    "post_pre of 4 parts": ("post_pre", 4),
}


def case(mode: str, world: int, tokens: int, seed: int):
    """Both kernels on one fresh entry, each with its outputs."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=DEVICE) * scale

    streams = 1 if mode == "broadcast" else 4
    x = randn(world, tokens, HIDDEN).bfloat16()
    residual = randn(tokens, 4 * HIDDEN).bfloat16()
    post = 2 * torch.sigmoid(randn(tokens, 4))
    comb = torch.softmax(randn(tokens, 4, 4), dim=-1)
    incoming = torch.softmax(randn(tokens, 4), dim=-1) + 1e-6
    fn = randn(MIXES, streams * HIDDEN, scale=0.02)
    splits = HIDDEN // BLOCK_H

    def outputs():
        return (
            torch.empty(tokens, 4 * HIDDEN, dtype=torch.bfloat16, device=DEVICE),
            torch.empty(splits, tokens, 1, MIXES, device=DEVICE),
            torch.empty(splits, tokens, 1, device=DEVICE),
            torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE),
            torch.empty(splits, tokens, device=DEVICE),
        )

    ours_out, theirs_out = outputs(), outputs()
    ours_kernel = project_streams(HIDDEN, mode, block_M=BLOCK_M, world=world)
    theirs_kernel = project_streams_26382f5cf(
        HIDDEN, mode, block_M=BLOCK_M, world=world
    )

    def call(kernel, out):
        source = out[0] if mode == "broadcast" else residual
        return lambda: kernel(x, source, post, comb, incoming, fn, *out)

    def check():
        written = ours_out[1:] if mode == "pre" else ours_out
        baseline = theirs_out[1:] if mode == "pre" else theirs_out
        for a, b in zip(written, baseline):
            bits = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            if not torch.equal(a.view(bits), b.view(bits)):
                raise SystemExit(f"{mode} x{world}, {tokens} tokens: bits differ")

    return call(ours_kernel, ours_out), call(theirs_kernel, theirs_out), check


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS  # microseconds a call


def measure(mode: str, world: int, tokens: int, samples: int = 8) -> dict:
    ours_us, theirs_us, differences = [], [], []
    for sample in range(samples):
        ours_call, theirs_call, check = case(mode, world, tokens, sample)
        ours_call()
        theirs_call()
        torch.cuda.synchronize()
        check()
        ours, theirs = graph(ours_call), graph(theirs_call)
        pair = (ours, theirs) if sample % 2 == 0 else (theirs, ours)
        times = {id(capture): elapsed(capture) for capture in pair}
        ours_us.append(times[id(ours)])
        theirs_us.append(times[id(theirs)])
        differences.append(times[id(ours)] - times[id(theirs)])
        del ours, theirs, pair
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tokens": tokens,
        "prefetched_us": statistics.median(ours_us),
        "26382f5cf_us": statistics.median(theirs_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    results = {
        f"{entry} {shape}": measure(*ENTRIES[entry], tokens)
        for entry in ENTRIES
        for shape, tokens in TOKENS.items()
        # The rank parts arrive only at sequence-parallel prefill.
        if entry != "post_pre of 4 parts" or shape.startswith("prefill")
    }
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [name for name, r in results.items() if r["interval_us"][0] > 0]
        if slower:
            raise SystemExit(f"project_streams slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
