# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness and paired per-call timing of each CED port at serving shapes."""

# Callbacks are consumed synchronously by paired(), before the loop advances.
# ruff: noqa: B023

import json
import math
import statistics

import torch

from vllm.models.deepseek_v4_1.ced import (
    _copy_rows,
    _decoder_indices,
    _decoder_requests,
)
from vllm.models.deepseek_v4_1.tilelang import ced
from vllm.triton_utils import triton


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


def elapsed(capture):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # ms / 100 calls to microseconds


def paired(kernel, shape, reference, candidate, equal):
    captures = [graph(reference), graph(candidate)]
    assert equal(), (kernel, shape)
    samples = [[], []]
    for repeat in range(30):
        for i in (0, 1) if repeat % 2 == 0 else (1, 0):
            samples[i].append(elapsed(captures[i]))
    differences = [b / a - 1 for a, b in zip(*samples)]
    row = dict(
        kernel=kernel,
        **shape,
        reference_us=statistics.mean(samples[0]),
        candidate_us=statistics.mean(samples[1]),
        change_pct=100 * statistics.mean(differences),
        ci95_pct=100 * 2.045 * statistics.stdev(differences) / math.sqrt(30),
        samples_us=samples,
    )
    print(json.dumps(row), flush=True)
    return row


def requests_cases():
    for maximum, requests in (
        (1, 1),
        (8, 1),
        (8, 8),
        (16, 1),
        (16, 8),
        (16, 16),
        (32, 32),
    ):
        for length in (1, 6, 129, 4096):
            starts = (
                torch.arange(requests + 1, dtype=torch.int32, device="cuda") * length
            )
            seq = torch.full((maximum,), 100000, dtype=torch.int32, device="cuda")
            full = torch.arange(maximum, device="cuda") % 2 == 1
            limits = torch.where(full, 2**31 - 1, 128).to(torch.int32)
            prefix = torch.zeros(maximum, dtype=torch.int64, device="cuda")
            outputs = [
                (
                    torch.empty(maximum + 1, dtype=torch.int32, device="cuda"),
                    torch.empty(maximum, dtype=torch.int64, device="cuda"),
                    torch.empty(maximum, dtype=torch.int64, device="cuda"),
                    torch.empty(2, dtype=torch.int32, device="cuda"),
                )
                for _ in range(2)
            ]
            yield paired(
                "requests",
                dict(max_requests=maximum, requests=requests, query_rows=length),
                lambda: _decoder_requests[(1,)](
                    starts,
                    seq,
                    full,
                    prefix,
                    *outputs[0],
                    requests,
                    maximum,
                    triton.next_power_of_2(maximum + 1),
                    128,
                ),
                lambda: ced.requests(
                    starts,
                    seq,
                    limits,
                    prefix,
                    *outputs[1],
                    requests,
                    compact=length > 128,
                ),
                lambda: all(torch.equal(a, b) for a, b in zip(*outputs)),
            )


def indices_cases():
    for requests in (1, 8, 16):
        for length in (1, 6, 128):
            starts = torch.arange(requests + 1, dtype=torch.int32, device="cuda") * 4096
            compact = (
                torch.arange(requests + 1, dtype=torch.int32, device="cuda") * length
            )
            nt = requests * length
            capacity = max(128, triton.next_power_of_2(nt))
            outputs = [
                torch.empty(capacity, dtype=torch.int64, device="cuda")
                for _ in range(2)
            ]
            yield paired(
                "indices",
                dict(requests=requests, kept_rows=length, capacity=capacity),
                lambda: _decoder_indices[(triton.cdiv(capacity, 128),)](
                    starts, compact, outputs[0], requests, nt, capacity, 128
                ),
                lambda: ced.indices(starts, compact, outputs[1], requests, nt),
                lambda: torch.equal(*outputs),
            )


def copy_cases():
    for length, retained in ((1, 1), (48, 48), (4048, 128)):
        for dtype, tail in (
            (torch.int64, ()),
            (torch.bool, ()),
            (torch.float32, (4,)),
            (torch.bfloat16, (5120,)),
            (torch.bfloat16, (4, 5120)),
        ):
            for scatter in (False, True):
                source_rows = retained if scatter else length
                output_rows = length if scatter else retained
                source = torch.ones((source_rows, *tail), dtype=dtype, device="cuda")
                indices = torch.arange(
                    length - retained, length, dtype=torch.int64, device="cuda"
                )
                outputs = [
                    torch.zeros((output_rows, *tail), dtype=dtype, device="cuda")
                    for _ in range(2)
                ]
                width = math.prod(tail)
                yield paired(
                    "copy",
                    dict(
                        rows=length,
                        retained=retained,
                        dtype=str(dtype),
                        tail=tail,
                        scatter=scatter,
                    ),
                    lambda: _copy_rows[
                        (
                            triton.cdiv(
                                (source_rows if scatter else output_rows) * width, 512
                            ),
                        )
                    ](
                        source,
                        indices,
                        outputs[0],
                        source_rows,
                        output_rows,
                        tail,
                        source.stride(),
                        width,
                        scatter,
                        512,
                    ),
                    lambda: ced.copy_rows(source, indices, outputs[1], scatter),
                    lambda: torch.equal(*outputs),
                )


def main():
    rows = [*requests_cases(), *indices_cases(), *copy_cases()]
    print(json.dumps({"summary": "CED ports per call", "rows": rows}))
    print(json.dumps({"bits": "CED metadata and packing", "differs_at": []}))
    slower = [r for r in rows if r["change_pct"] - r["ci95_pct"] > 0.0]
    if slower:
        raise RuntimeError(f"CED ports are slower (95% paired intervals): {slower}")


if __name__ == "__main__":
    main()
