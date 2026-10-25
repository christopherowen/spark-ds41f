# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of ``tilelang_topk_ids.logprob_token_ids`` against what it
replaced in ``compute_topk_scores``' fast path, ``torch.cat((sampled[:, None],
torch.topk(logits, k).indices), 1)``: the DeepSeek V4.1 vocabulary (129280 columns,
dense rows), FP32 and bf16 logits, one, sixteen and 96 rows, k 1, 5 and 20.

Each case is measured on eight fresh inputs; on each, CUDA graphs of 100 calls of
both run back to back, alternating which goes first, and the case's 95% interval
comes from the eight per-input differences. The gate (``--gate``) is that the
TileLang launch is nowhere slower (the interval's lower bound above zero). Before
timing, each pair's outputs are compared index for index (the logits are rounded to
bf16, so ties reach the top k).
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch

from vllm.v1.worker.gpu.sample import tilelang_topk_ids

DEVICE = torch.device("cuda")
CALLS = 100
VOCAB = 129280
SHAPES = {"1 row": 1, "16 rows": 16, "96 rows": 96}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
KS = (1, 5, 20)


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


def case(dtype: torch.dtype, rows: int, k: int, seed: int):
    """(torch call, TileLang call, their outputs) on fresh logits: normal, sd 4,
    rounded to bf16."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = (torch.randn(rows, VOCAB, generator=gen, device=DEVICE) * 4).to(
        torch.bfloat16
    )
    logits = logits.to(dtype)
    sampled = torch.randint(VOCAB, (rows,), generator=gen, device=DEVICE)
    out = {}

    def reference():
        topk = torch.topk(logits, k, dim=-1).indices
        out["ref"] = torch.cat((sampled.unsqueeze(-1), topk), dim=1)

    def candidate():
        out["new"] = tilelang_topk_ids.logprob_token_ids(logits, k, sampled)

    return reference, candidate, lambda: (out["ref"], out["new"])


def measure(dtype: torch.dtype, rows: int, k: int, samples: int) -> dict:
    ours, theirs = [], []
    for sample in range(samples):
        reference, candidate, outputs = case(dtype, rows, k, 7919 + sample)
        reference()
        candidate()
        ref, new = outputs()
        if not torch.equal(ref, new):
            raise SystemExit(
                f"{dtype} at {rows} rows, k {k}: ids differ from torch.topk's"
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
        "rows": rows,
        "k": k,
        "tilelang_us": statistics.median(ours),
        "reference_us": statistics.median(theirs),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
        "slower": mean - half > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    results = []
    for dtype_name, dtype in DTYPES.items():
        for shape, rows in SHAPES.items():
            for k in KS:
                result = measure(dtype, rows, k, args.samples)
                result["dtype"] = dtype_name
                result["shape"] = shape
                results.append(result)
                print(json.dumps(result), flush=True)
    slower = [
        f"{r['dtype']} at {r['shape']}, k {r['k']}" for r in results if r["slower"]
    ]
    print(
        json.dumps({"summary": "top-k token ids vs torch.topk", "slower": slower}),
        flush=True,
    )
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
