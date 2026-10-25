# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of the sink detector's TileLang kernels against the torch
chains they replace, and of the TileLang grammar bitmask against its Triton kernel,
at decode shapes: one request, sixteen and 96 logits rows (1 + 5 drafted rows a
request), the flags over four residual streams and the greedy alternates over a
4-way vocabulary shard (32320 columns).

Each kernel and shape is measured on eight fresh inputs; on each, CUDA graphs of 100
calls of both run back to back, alternating which goes first, and the case's 95%
interval comes from the eight per-input differences. The gate (``--gate``) is that no
TileLang kernel is slower (the interval's lower bound above zero). Before timing,
each pair's outputs are compared bit for bit (the shard candidates on rows without
ties, where torch.topk's order is defined).
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch

from vllm.v1.worker.gpu import tilelang_structured
from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

DEVICE = torch.device("cuda")
CALLS = 100
SHARD = 32320
THRESHOLD = 85000.0
ALTERNATES = 3
SHAPES = {"1 request": 1, "16 rows": 16, "96 rows": 96}


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


def _cu(rows: int) -> torch.Tensor:
    counts = [min(6, rows - 6 * i) for i in range(-(-rows // 6))]
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )


# --- the torch chains the kernels replace (sink.py before the port) -------------------


def torch_flags(columns, logits_indices, cu, num_reqs, dead_cut):
    rows = columns[logits_indices].abs()
    values = rows.amax(dim=1).nan_to_num(0.0)
    flags = (rows > THRESHOLD).any(dim=1)
    index = torch.arange(flags.shape[0], dtype=cu.dtype, device=DEVICE)
    req = torch.searchsorted(cu[1 : num_reqs + 1], index, right=True)
    flags &= index - cu[req] <= dead_cut[req]
    return flags.to(torch.int8), values


def torch_cut(logits, flags, drafts, cu, num_reqs):
    num_logits, num_cols = logits.shape
    starts = torch.zeros(num_logits + 1, dtype=torch.bool, device=DEVICE)
    starts.index_fill_(0, cu[:num_reqs].long(), True)
    cols = drafts[1:num_logits].long()
    cut = flags[1:num_logits].bool() & ~starts[1:num_logits] & (cols >= 0)
    cut &= cols < num_cols
    cols = cols.where(cut, 0)[:, None]
    rows = logits[: num_logits - 1]
    saved = rows.gather(1, cols)
    rows.scatter_(1, cols, saved.masked_fill(cut[:, None], float("-inf")))
    rows.scatter_(1, cols, saved)  # the restore


def torch_alternates(local, cu, num_sampled, sampled):
    num_reqs = num_sampled.shape[0]
    rows = cu[:num_reqs].to(torch.int64) + (num_sampled.to(torch.int64) - 1).clamp(
        min=0
    )
    values, indices = local[rows, :SHARD].float().topk(ALTERNATES + 1, dim=-1)
    values = values.masked_fill(values.isnan(), float("-inf"))
    last = (num_sampled.long() - 1).clamp(min=0)[:, None]
    excluded = (indices == sampled.gather(1, last)) | values.isnan()
    values = values.masked_fill(excluded, float("-inf"))
    values, order = values.sort(dim=1, descending=True, stable=True)
    alternates = indices.gather(1, order[:, :ALTERNATES])
    return alternates.masked_fill(values[:, :ALTERNATES] == float("-inf"), -1)


def tilelang_alternates(local, cu, num_sampled, sampled):
    packed = tilelang_sampling.sink_shard_top(
        local, cu, num_sampled, 0, SHARD, ALTERNATES + 1
    )
    return tilelang_sampling.sink_merge(
        packed, None, sampled, num_sampled, ALTERNATES, ALTERNATES + 1
    )


# --- the cases -------------------------------------------------------------------


def case(kind: str, rows: int, seed: int):
    """(torch or Triton call, TileLang call, outputs to compare) on fresh inputs."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    cu = _cu(rows)
    num_reqs = cu.shape[0] - 1
    if kind == "flags":
        columns = torch.randn(16384, 4, generator=gen, device=DEVICE) * 60000.0
        indices = torch.randperm(16384, generator=gen, device=DEVICE)[:rows]
        dead_cut = torch.randint(0, 6, (num_reqs,), generator=gen, device=DEVICE).int()
        out = {}

        def reference():
            out["ref"] = torch_flags(columns, indices, cu, num_reqs, dead_cut)

        def candidate():
            out["new"] = tilelang_sampling.sink_flags(
                columns, indices, cu, num_reqs, dead_cut, THRESHOLD
            )

        return reference, candidate, lambda: (out["ref"], out["new"])
    if kind == "cut":
        logits = torch.randn(rows, SHARD, generator=gen, device=DEVICE)
        flags = torch.randint(0, 2, (rows,), generator=gen, device=DEVICE).to(
            torch.int8
        )
        drafts = torch.randint(0, SHARD, (rows,), generator=gen, device=DEVICE).int()
        before = logits.clone()

        def reference():
            torch_cut(logits, flags, drafts, cu, num_reqs)

        def candidate():
            saved, cols = tilelang_sampling.sink_cut(
                logits, flags, drafts, cu, num_reqs, 0, SHARD
            )
            tilelang_sampling.sink_restore(logits, saved, cols, flags, drafts, cu)

        return reference, candidate, lambda: ((before,), (logits,))
    if kind == "alternates":
        # A permutation a row: no ties, so torch.topk's order is defined.
        local = torch.stack(
            [torch.randperm(SHARD, generator=gen, device=DEVICE) for _ in range(rows)]
        ).float()
        num_sampled = torch.tensor(
            [min(6, rows - 6 * i) for i in range(num_reqs)],
            dtype=torch.int32,
            device=DEVICE,
        )
        sampled = torch.randint(0, SHARD, (num_reqs, 6), generator=gen, device=DEVICE)
        out = {}

        def reference():
            out["ref"] = torch_alternates(local, cu, num_sampled, sampled)

        def candidate():
            out["new"] = tilelang_alternates(local, cu, num_sampled, sampled)

        return reference, candidate, lambda: ((out["ref"],), (out["new"],))
    # The grammar bitmask: every row masked by its request at position 0..5.
    vocab = 129280
    logits = torch.randn(rows, vocab, generator=gen, device=DEVICE)
    keys = torch.tensor(
        [r * 8 + p for r in range(num_reqs) for p in range(int(cu[r + 1] - cu[r]))],
        dtype=torch.int32,
        device=DEVICE,
    )
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (rows, -(-vocab // 32)), generator=gen, device=DEVICE
    ).int()
    expected, actual = logits.clone(), logits.clone()

    def reference():
        _apply_grammar_bitmask_kernel[(rows, -(-vocab // 8192))](
            expected,
            expected.stride(0),
            keys,
            cu,
            None,
            None,
            bitmask,
            bitmask.stride(0),
            vocab,
            MASK_STRIDE=8,
            BLOCK_SIZE=8192,
        )

    def candidate():
        tilelang_structured.apply_grammar_bitmask(
            actual, keys, cu, None, None, bitmask, 8
        )

    return reference, candidate, lambda: ((expected,), (actual,))


def same(a, b) -> bool:
    return all(
        torch.equal(
            x.view(torch.int32) if x.dtype == torch.float32 else x,
            y.view(torch.int32) if y.dtype == torch.float32 else y,
        )
        for x, y in zip(a, b)
    )


def measure(kind: str, rows: int, samples: int) -> dict:
    ours, theirs = [], []
    for sample in range(samples):
        reference, candidate, outputs = case(kind, rows, 7919 + sample)
        reference()
        candidate()
        ref, new = outputs()
        if not same(ref, new):
            raise SystemExit(
                f"{kind} at {rows} rows: outputs differ from the reference"
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
        "kernel": kind,
        "rows": rows,
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
    for kind in ("flags", "cut", "alternates", "bitmask"):
        for name, rows in SHAPES.items():
            result = measure(kind, rows, args.samples)
            result["shape"] = name
            results.append(result)
            print(json.dumps(result), flush=True)
    slower = [f"{r['kernel']} at {r['shape']}" for r in results if r["slower"]]
    print(
        json.dumps({"summary": "sink glue and grammar bitmask", "slower": slower}),
        flush=True,
    )
    if args.gate and slower:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
