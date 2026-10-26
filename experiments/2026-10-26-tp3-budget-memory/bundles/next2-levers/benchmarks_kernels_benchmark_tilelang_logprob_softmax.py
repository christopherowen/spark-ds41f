# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of the TileLang log-softmax of selected tokens
(``tilelang_logprob.topk_log_softmax``, Barron et al.'s fix) against a baseline at
DeepSeek V4.1's logprob shapes: one request, sixteen and 96 rows of FP32 and bf16
logits (129280 columns of the LM head's 131072-wide rows), one requested token (the
sampled one) and twenty. The baseline is the Triton kernel the TileLang port
replaced (``logprob._topk_log_softmax_kernel``, launched as production launched it),
or with ``--baseline DIR`` the release's TileLang kernel: ``DIR`` holds the
release's ``tilelang_logprob.py`` and ``tilelang_rejection.py`` (whose header the
former reads), loaded beside the installed modules.

Each case is measured on eight fresh inputs; on each, CUDA graphs of 100 calls of
both run back to back, alternating which goes first, and the case's 95% interval
comes from the eight per-input differences. The gate (``--gate``) is that the
TileLang kernel is never slower (the interval's lower bound above zero). Before
timing, each pair's outputs are compared: the baseline's arithmetic is not the fix's,
so within 1e-5 relative and 1e-6 absolute (the collapsed log probabilities of the
baseline are 0 where the fix's are about -1e-8), NaN where the other is.
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.v1.worker.gpu.sample import logprob, tilelang_logprob

DEVICE = torch.device("cuda")
CALLS = 100
VOCAB = 129280
PADDED = 131072
SHAPES = {"1 request": 1, "16 rows": 16, "96 rows": 96}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
TOPKS = (1, 20)
REJECTION = "vllm.v1.worker.gpu.spec_decode.tilelang_rejection"


def load_release(directory: str):
    """The release's ``tilelang_logprob`` from ``directory``, importing the release's
    ``tilelang_rejection`` from there too (both under names of their own; the
    installed modules are left as they were)."""

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    root = Path(directory)
    rejection = load("release_tilelang_rejection", root / "tilelang_rejection.py")
    installed = sys.modules[REJECTION]
    sys.modules[REJECTION] = rejection
    try:
        return load("release_tilelang_logprob", root / "tilelang_logprob.py")
    finally:
        sys.modules[REJECTION] = installed


def graph(function: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """A CUDA graph of ``CALLS`` calls of ``function``, after one warm-up call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        function()
    torch.cuda.current_stream().wait_stream(side)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture: torch.cuda.CUDAGraph) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    capture.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / CALLS


def case(dtype: torch.dtype, rows: int, topk: int, seed: int, release):
    """(baseline call, TileLang call, outputs to compare) on fresh inputs."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    # The LM head's padded rows, the real vocabulary in view.
    backing = torch.randn(rows, PADDED, generator=gen, device=DEVICE) * 4
    logits = backing.to(dtype)[:, :VOCAB]
    ids = torch.randint(VOCAB, (rows, topk), generator=gen, device=DEVICE)
    expected = torch.empty(rows, topk, device=DEVICE)
    actual = torch.empty(rows, topk, device=DEVICE)

    def reference():
        if release is not None:
            release.topk_log_softmax(logits, ids, expected)
            return
        logprob._topk_log_softmax_kernel[(rows,)](
            expected,
            logits,
            logits.stride(0),
            ids,
            topk,
            VOCAB,
            BLOCK_SIZE=1024,
            TOPK_BLOCK_SIZE=logprob._MAX_TOPK_BLOCK,
        )

    def candidate():
        tilelang_logprob.topk_log_softmax(logits, ids, actual)

    return reference, candidate, lambda: (expected, actual)


def measure(dtype: torch.dtype, rows: int, topk: int, samples: int, release) -> dict:
    ours, theirs = [], []
    for sample in range(samples):
        reference, candidate, outputs = case(dtype, rows, topk, 7919 + sample, release)
        reference()
        candidate()
        expected, actual = outputs()
        if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True):
            raise SystemExit(
                f"{dtype} at {rows} rows, {topk} tokens: outputs differ from the "
                "baseline's beyond its rounding"
            )
        a, b = graph(candidate), graph(reference)
        order = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(g): elapsed(g) for g in order}
        ours.append(times[id(a)])
        theirs.append(times[id(b)])
    paired = [x - y for x, y in zip(ours, theirs)]
    mean = statistics.fmean(paired)
    half = 2.365 * statistics.stdev(paired) / math.sqrt(len(paired))
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "rows": rows,
        "topk": topk,
        "baseline": "release" if release is not None else "triton",
        "tilelang_us": statistics.median(ours),
        "baseline_us": statistics.median(theirs),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
        "slower": mean - half > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--baseline",
        metavar="DIR",
        help="the release's tilelang_logprob.py and tilelang_rejection.py",
    )
    args = parser.parse_args()
    release = load_release(args.baseline) if args.baseline else None
    results = []
    for dtype in DTYPES.values():
        for name, rows in SHAPES.items():
            for topk in TOPKS:
                result = measure(dtype, rows, topk, args.samples, release)
                result["shape"] = name
                results.append(result)
                print(json.dumps(result), flush=True)
    slower = [
        f"{r['dtype']} at {r['shape']}, {r['topk']} tokens"
        for r in results
        if r["slower"]
    ]
    print(json.dumps({"summary": "logprob log-softmax", "slower": slower}), flush=True)
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
