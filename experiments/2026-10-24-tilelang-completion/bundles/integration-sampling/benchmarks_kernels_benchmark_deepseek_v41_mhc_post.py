# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of DeepSeek-V4.1's TileLang mHC post and stream collapse
against the TileKernels CUDA kernels they replace (``mhc_post_fwd``,
``mhc_pre_apply_mix_fwd``), at serving shapes: four BF16 streams of 5,120.

Each shape is measured on eight fresh inputs; on each, both kernels' outputs are
first checked bit for bit against each other, then CUDA graphs of 100 calls of both
run back to back, alternating which goes first, and the case's 95% interval comes
from the eight per-input differences. The gate (``--gate``) is that neither port is
slower at a decode step (16 requests of 1 + 5 drafted tokens: 96 tokens) or a
long-prefill chunk (16,384 tokens).
"""

import argparse
import json
import math
import statistics

import tile_kernels
import torch

# Registers the ops.
from vllm.models.deepseek_v4_1.tilelang import mhc  # noqa: F401

STREAMS, HIDDEN = 4, 5120
SHAPES = {"decode c1": 6, "decode c16": 96, "prefill 8K": 8192, "prefill 16K": 16384}
GATED = ("decode c16", "prefill 16K")


def post_case(tokens: int, seed: int):
    """The TileLang post and TileKernels' on one fresh input, each with its output."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    x = randn(tokens, HIDDEN)
    residual = randn(tokens, STREAMS, HIDDEN)
    post = 2 * torch.sigmoid(
        torch.randn(tokens, STREAMS, generator=generator, device="cuda")
    )
    comb = torch.softmax(
        torch.randn(tokens, STREAMS, STREAMS, generator=generator, device="cuda"), -1
    )
    ours_out, theirs_out = torch.empty_like(residual), torch.empty_like(residual)

    def ours():
        torch.ops.vllm.dsv41_tilelang_mhc_post(x, residual, post, comb, ours_out)

    def theirs():
        tile_kernels.mhc.mhc_post_fwd(
            x.view(1, tokens, HIDDEN),
            residual.view(1, tokens, STREAMS, HIDDEN),
            post.view(1, tokens, STREAMS, 1),
            comb.view(1, tokens, STREAMS, STREAMS),
            out=theirs_out.view(1, tokens, STREAMS, HIDDEN),
        )

    return ours, theirs, ours_out, theirs_out


def collapse_case(tokens: int, seed: int):
    """The TileLang stream collapse and TileKernels' on one fresh input."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(tokens, STREAMS, HIDDEN, generator=generator, device="cuda")
    state = state.bfloat16()
    mix = torch.randn(tokens, STREAMS, generator=generator, device="cuda").sigmoid()
    ours_out = torch.empty(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    theirs_out = torch.empty_like(ours_out)

    def ours():
        torch.ops.vllm.dsv41_tilelang_mhc_collapse(state, mix, ours_out)

    def theirs():
        tile_kernels.mhc.mhc_pre_apply_mix_fwd(state, mix, theirs_out)

    return ours, theirs, ours_out, theirs_out


CASES = {"post": post_case, "collapse": collapse_case}


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
        for _ in range(100):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # ms / 100 calls to microseconds


def measure(name: str, tokens: int, samples: int = 8) -> dict:
    ours_us, theirs_us, differences = [], [], []
    for sample in range(samples):
        ours_call, theirs_call, ours_out, theirs_out = CASES[name](tokens, sample)
        ours_call()
        theirs_call()
        torch.cuda.synchronize()
        if not torch.equal(ours_out.view(torch.int16), theirs_out.view(torch.int16)):
            raise SystemExit(f"{name}, {tokens} tokens: bits differ from TileKernels'")
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
        "tilelang_us": statistics.median(ours_us),
        "tile_kernels_us": statistics.median(theirs_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    results = {
        name: {shape: measure(name, tokens) for shape, tokens in SHAPES.items()}
        for name in CASES
    }
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [
            f"{name} {shape}"
            for name in CASES
            for shape in GATED
            if results[name][shape]["interval_us"][0] > 0
        ]
        if slower:
            raise SystemExit(f"TileLang port slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
