# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of DeepSeek-V4.1's mHC finalize writing the sublayer input's MXFP8
cast (``linear.MXFP8Rows``) against the release's finalize (``--baseline``, its
``mhc.py``) followed by the separate casts the cast replaces (TileKernels'
``per_token_cast``), at serving token counts. Four BF16 streams of 5,120.

The sequences, as a TP4 forward runs them:

- decode attention entry: the finalize, then one cast (attention's fused Q-A/KV);
- decode FFN entry: the finalize, then two casts of the same rows (the shared
  expert's gate/up and the routed experts);
- prefill SP layer: both entries on one rank's quarter of a 16K chunk, and the casts
  of the gathered chunk the layer still needs: attention's fused Q-A/KV of the
  gathered rows (a compressor layer) and the FFN's (once, where the release cast the
  same rows twice).

Each case is measured on eight fresh inputs. On each, the outputs are first checked:
the new finalize's ``y`` equals the release's bit for bit, and its cast equals
``per_token_cast`` of that ``y`` byte for byte. Then CUDA graphs of 100 sequences of
each run back to back, alternating which goes first; the case's 95% interval comes
from the eight per-input differences. The gate (``--gate``) is that the new
sequences are not slower in any case.
"""

import argparse
import importlib.util
import json
import math
import statistics

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import mhc
from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words
from vllm.models.deepseek_v4_1.tilelang.linear import mxfp8_rows

HIDDEN, STREAMS = 5120, 4
RMS_EPS, HC_EPS, ITERATIONS = 1e-6, 1e-6, 20
CASES = {
    "decode c1 attention": ("attention", 4),
    "decode c1 FFN": ("ffn", 4),
    "decode c4 attention": ("attention", 24),
    "decode c4 FFN": ("ffn", 24),
    "decode c16 attention": ("attention", 96),
    "decode c16 FFN": ("ffn", 96),
    "prefill SP layer": ("sp", 4096),
}
SP_WORLD = 4


def baseline_module(path: str, name: str):
    """The release's copy of a TileLang module, under ``name`` in the same package:
    its relative imports resolve to the package's other modules, and the custom ops
    it defines stay the package's (their names are registered already)."""

    def unregistered(*args, **kwargs):
        def define(function):
            function.register_fake = lambda fake: fake
            return function

        return define

    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    original, torch.library.custom_op = torch.library.custom_op, unregistered
    try:
        spec.loader.exec_module(module)
    finally:
        torch.library.custom_op = original
    return module


class Entry:
    """One sublayer entry's finalize inputs (the projection partials of real streams),
    with both builds' outputs."""

    def __init__(self, release, tokens: int, generator):
        def randn(*shape, scale=1.0):
            return torch.randn(*shape, generator=generator, device="cuda") * scale

        residual = randn(tokens, STREAMS * HIDDEN).bfloat16()
        fn = randn(mhc.MIXES, STREAMS * HIDDEN, scale=0.02)
        incoming = torch.softmax(randn(tokens, STREAMS), dim=-1) + HC_EPS
        splits = HIDDEN // mhc.BLOCK_H
        self.P = torch.empty((splits, tokens, 1, mhc.MIXES), device="cuda")
        self.S = torch.empty((splits, tokens, 1), device="cuda")
        self.C = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        self.Q = torch.empty((splits, tokens), device="cuda")
        comb = torch.empty((tokens, STREAMS, STREAMS), device="cuda")
        mhc.project_streams(HIDDEN, "pre", block_M=mhc.BLOCK_M)(
            self.C.unsqueeze(0), residual, incoming, comb, incoming, fn, residual,
            self.P, self.S, self.C, self.Q,
        )  # fmt: skip
        self.scale = randn(3, scale=0.1) + 1.0
        self.base = randn(mhc.MIXES, scale=0.1)
        self.weight = (randn(HIDDEN, scale=0.1) + 1.0).bfloat16()
        self.mixes = {
            name: (
                torch.empty((tokens, STREAMS), device="cuda"),
                torch.empty((tokens, STREAMS, STREAMS), device="cuda"),
                torch.empty((tokens, STREAMS), device="cuda"),
                torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda"),
            )
            for name in ("new", "release")
        }
        rows = mxfp8_rows(tokens)
        self.yq = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
        self.ysf = torch.empty(
            (rows, 4 * scale_words(HIDDEN)), dtype=torch.uint8, device="cuda"
        )
        key = (HIDDEN, STREAMS * HIDDEN, RMS_EPS, HC_EPS, ITERATIONS, RMS_EPS)
        self.new_kernel = mhc.finalize(*key)
        self.release_kernel = release.finalize(*key)
        self.tokens = tokens

    def new(self) -> None:
        post, comb, pre_out, y = self.mixes["new"]
        self.new_kernel(
            self.P, self.S, self.C, self.Q, self.scale, self.base, self.weight,
            post, comb, pre_out, y, self.yq, self.ysf,
        )  # fmt: skip

    def release(self) -> None:
        post, comb, pre_out, y = self.mixes["release"]
        self.release_kernel(
            self.P, self.S, self.C, self.Q, self.scale, self.base, self.weight,
            post, comb, pre_out, y,
        )  # fmt: skip

    def check(self) -> None:
        self.new()
        self.release()
        torch.cuda.synchronize()
        for new, old in zip(self.mixes["new"], self.mixes["release"]):
            if not torch.equal(new, old):
                raise SystemExit(f"{self.tokens} tokens: finalize outputs differ")
        values, scales = Cast(self.mixes["release"][3]).run_once()
        live = self.tokens
        if not (
            torch.equal(self.yq[:live].view(torch.uint8), values.view(torch.uint8))
            and torch.equal(self.ysf[:live], scales)
        ):
            raise SystemExit(f"{self.tokens} tokens: cast differs from per_token_cast")


class Cast:
    """TileKernels' ``per_token_cast`` of BF16 rows into a projection's buffers."""

    def __init__(self, rows: torch.Tensor):
        tokens = rows.shape[0]
        self.rows = rows
        self.values = torch.empty(
            (tokens, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda"
        )
        self.scales = torch.empty(
            (tokens, 4 * scale_words(HIDDEN)), dtype=torch.uint8, device="cuda"
        )

    def __call__(self) -> None:
        tile_kernels.quant.per_token_cast(
            self.rows, "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
            out=(self.values, self.scales),
        )  # fmt: skip

    def run_once(self):
        self()
        torch.cuda.synchronize()
        return self.values, self.scales


def sequences(release, kind: str, tokens: int, seed: int):
    """The new and the release sequence of one case, after checking their outputs."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if kind != "sp":
        entry = Entry(release, tokens, generator)
        entry.check()
        casts = [
            Cast(entry.mixes["release"][3])
            for _ in range(1 if kind == "attention" else 2)
        ]

        def new():
            entry.new()

        def old():
            entry.release()
            for cast in casts:
                cast()

        return new, old
    # One rank's rows of each entry; the layer's gathered rows (the other ranks'
    # entries ran elsewhere).
    entries = [Entry(release, tokens, generator) for _ in ("attention", "ffn")]
    for entry in entries:
        entry.check()
    gathered = torch.randn(
        (SP_WORLD * tokens, HIDDEN), generator=generator, device="cuda"
    ).bfloat16()
    attention, ffn = Cast(gathered), Cast(gathered)
    shared, routed = Cast(gathered), Cast(gathered)

    def new():
        entries[0].new()
        attention()
        entries[1].new()
        ffn()

    def old():
        entries[0].release()
        attention()
        entries[1].release()
        shared()
        routed()

    return new, old


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
    return start.elapsed_time(end) * 10  # ms / 100 sequences to microseconds


def measure(release, kind: str, tokens: int, samples: int = 8) -> dict:
    new_us, old_us, differences = [], [], []
    for sample in range(samples):
        new_call, old_call = sequences(release, kind, tokens, sample)
        new, old = graph(new_call), graph(old_call)
        pair = (new, old) if sample % 2 == 0 else (old, new)
        times = {id(capture): elapsed(capture) for capture in pair}
        new_us.append(times[id(new)])
        old_us.append(times[id(old)])
        differences.append(times[id(new)] - times[id(old)])
        del new, old, pair
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tokens": tokens,
        "finalize_cast_us": statistics.median(new_us),
        "release_us": statistics.median(old_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="the release's vllm/models/deepseek_v4_1/tilelang/mhc.py",
    )
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    release = baseline_module(args.baseline, "mhc_release")
    results = {
        name: measure(release, kind, tokens) for name, (kind, tokens) in CASES.items()
    }
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [name for name, case in results.items() if case["interval_us"][0] > 0]
        if slower:
            raise SystemExit(f"finalize with the cast slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
