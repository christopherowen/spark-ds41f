# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness and paired per-call timing of each CED port at serving shapes.

Each case checks the port's outputs against Triton's on separate buffers, then
times both kernels writing the same buffers, so neither gains from where its
output happens to land. Two graphs of the same kernel still differ by up to
about one percent (Triton against itself, 2026-10-19), more than one pair of
graphs' samples can show, and a large copy's time depends on where its inputs
land. So each case is measured on eight fresh inputs, placed apart, and graph
pairs, alternating which comes first, and its 95% interval comes from those
eight differences; Triton against itself on the gated copies checks the
interval. With dozens of gated cases some read slower by chance, so a case that
reads slower is measured again on fresh placements and fails only if it reads
slower twice. The gate is serving shapes: the metadata kernels run on every
step, decode included; the row copies run only under a CED plan, which keeps at
least 128 rows, so their smaller cases are reported but do not gate.
"""

import functools
import json
import math
import random
import statistics

import torch

from benchmarks.kernels.ced_reference import (
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


INSTANCES = 8
PAIRS = 16
T_975 = 2.365  # Student t, 7 degrees of freedom
PLACEMENT = random.Random(1019)


def instance(build, order, candidate_is_reference=False):
    """One fresh input set and pair of graphs: the candidate's mean per-call
    change against the reference. Both graphs replay untimed first: a graph's
    first replay is slow, and one cold sample in sixteen pairs moves the mean by
    about a percent. ``order`` picks which graph is captured and replayed first,
    alternating between instances. A spacer of random size first moves where
    the inputs land: a large copy's time depends on placement (9.7 to 24 us for
    one Triton copy), and successive instances otherwise reuse the same blocks."""
    spacer = torch.empty(
        PLACEMENT.randrange(1, 64 << 20), dtype=torch.uint8, device="cuda"
    )
    reference, candidate, outputs, equal = build()
    if candidate_is_reference:
        candidate = reference
    theirs, ours = outputs(), outputs()
    for capture in (graph(reference(theirs)), graph(candidate(ours))):
        capture.replay()
    torch.cuda.synchronize()
    if not equal(theirs, ours):
        raise AssertionError("outputs differ")
    functions = [reference(theirs), candidate(theirs)]
    first = (0, 1) if order == 0 else (1, 0)
    captures = [None, None]
    for i in first:
        captures[i] = graph(functions[i])
    for _ in range(3):
        for i in first:
            elapsed(captures[i])
    samples = [[], []]
    for repeat in range(PAIRS):
        for i in first if repeat % 2 == 0 else first[::-1]:
            samples[i].append(elapsed(captures[i]))
    del spacer
    return samples, statistics.mean(b / a - 1 for a, b in zip(*samples))


def paired(kernel, shape, build, candidate_is_reference=False, attempt=1):
    """``build()`` makes fresh inputs and returns (reference, candidate, outputs,
    equal): the kernels as functions of an output set, a factory for output
    sets, and their comparison."""
    runs = [instance(build, i % 2, candidate_is_reference) for i in range(INSTANCES)]
    changes = [change for _, change in runs]
    row = dict(
        kernel=kernel,
        **shape,
        attempt=attempt,
        reference_us=statistics.mean(t for s, _ in runs for t in s[0]),
        candidate_us=statistics.mean(t for s, _ in runs for t in s[1]),
        change_pct=100 * statistics.mean(changes),
        ci95_pct=100 * T_975 * statistics.stdev(changes) / math.sqrt(INSTANCES),
        instance_changes_pct=[100 * c for c in changes],
    )
    print(json.dumps(row), flush=True)
    return row


def slower(row):
    return row["change_pct"] - row["ci95_pct"] > 0.0


def requests_case(maximum, requests, length):
    starts = torch.arange(requests + 1, dtype=torch.int32, device="cuda") * length
    seq = torch.full((maximum,), 100000, dtype=torch.int32, device="cuda")
    full = torch.arange(maximum, device="cuda") % 2 == 1
    limits = torch.where(full, 2**31 - 1, 128).to(torch.int32)
    prefix = torch.zeros(maximum, dtype=torch.int64, device="cuda")
    # Final chunks and decode rows: replay_rows' floors give the frozen kernel's
    # metadata (a chunk over the window replays from the prompt end less it).
    floors = torch.where(~full & (length > 128), seq.long() - 128, prefix)

    def outputs():
        return (
            torch.empty(maximum + 1, dtype=torch.int32, device="cuda"),
            torch.empty(maximum, dtype=torch.int64, device="cuda"),
            torch.empty(maximum, dtype=torch.int64, device="cuda"),
            torch.empty(2, dtype=torch.int32, device="cuda"),
        )

    return (
        lambda out: lambda: _decoder_requests[(1,)](
            starts,
            seq,
            full,
            prefix,
            *out,
            requests,
            maximum,
            triton.next_power_of_2(maximum + 1),
            128,
        ),
        lambda out: lambda: ced.requests(
            starts, seq, limits, floors, *out, requests, compact=length > 128
        ),
        outputs,
        lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b)),
    )


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
            yield (
                "requests",
                dict(max_requests=maximum, requests=requests, query_rows=length),
                functools.partial(requests_case, maximum, requests, length),
            )


def indices_case(requests, length):
    starts = torch.arange(requests + 1, dtype=torch.int32, device="cuda") * 4096
    compact = torch.arange(requests + 1, dtype=torch.int32, device="cuda") * length
    nt = requests * length
    capacity = max(128, triton.next_power_of_2(nt))
    return (
        lambda out: lambda: _decoder_indices[(triton.cdiv(capacity, 128),)](
            starts, compact, out, requests, nt, capacity, 128
        ),
        lambda out: lambda: ced.indices(starts, compact, out, requests, nt),
        lambda: torch.empty(capacity, dtype=torch.int64, device="cuda"),
        torch.equal,
    )


def indices_cases():
    for requests in (1, 8, 16):
        for length in (1, 6, 128):
            capacity = max(128, triton.next_power_of_2(requests * length))
            yield (
                "indices",
                dict(requests=requests, kept_rows=length, capacity=capacity),
                functools.partial(indices_case, requests, length),
            )


def copy_case(length, retained, padding, dtype, tail, scatter):
    source_rows = retained + padding if scatter else length
    output_rows = length if scatter else retained + padding
    source = torch.ones((source_rows, *tail), dtype=dtype, device="cuda")
    indices = torch.cat(
        [
            torch.arange(length - retained, length, device="cuda"),
            torch.full((padding,), -1, device="cuda"),
        ]
    )
    width = math.prod(tail)
    return (
        lambda out: lambda: _copy_rows[
            (triton.cdiv((source_rows if scatter else output_rows) * width, 512),)
        ](
            source,
            indices,
            out,
            source_rows,
            output_rows,
            tail,
            source.stride(),
            width,
            scatter,
            512,
        ),
        lambda out: lambda: ced.copy_rows(source, indices, out, scatter),
        lambda: torch.zeros((output_rows, *tail), dtype=dtype, device="cuda"),
        torch.equal,
    )


def copy_cases():
    # The serving layouts (CEDState.warmup), and a plan padded with -1 rows.
    for length, retained, padding in (
        (1, 1, 0),
        (48, 48, 0),
        (4048, 128, 0),
        (4048, 128, 64),
    ):
        serving = retained >= 128
        for dtype, tail in (
            (torch.int32, ()),
            (torch.int64, ()),
            (torch.bool, ()),
            (torch.float32, (4,)),
            (torch.float32, (4, 4)),
            (torch.bfloat16, (5120,)),
            (torch.bfloat16, (4, 5120)),
        ):
            for scatter in (False, True):
                yield (
                    "copy",
                    dict(
                        rows=length,
                        retained=retained,
                        padding=padding,
                        dtype=str(dtype),
                        tail=tail,
                        scatter=scatter,
                        serving=serving,
                    ),
                    functools.partial(
                        copy_case, length, retained, padding, dtype, tail, scatter
                    ),
                )


def confirmed(cases, rows, candidate_is_reference=False):
    """Rows slower on fresh placements again, of those that read slower."""
    again = [
        paired(*case, candidate_is_reference, attempt=2)
        for case, row in zip(cases, rows)
        if slower(row)
    ]
    return again, [row for row in again if slower(row)]


def main():
    cases = [*requests_cases(), *indices_cases(), *copy_cases()]
    rows = [paired(*case) for case in cases]
    gated = [(c, r) for c, r in zip(cases, rows) if r.get("serving", True)]
    again, failed = confirmed(*zip(*gated))
    checks = [
        (kernel, {**shape, "calibration": True}, build)
        for kernel, shape, build in copy_cases()
        if shape["serving"]
    ]
    calibration = [paired(*case, True) for case in checks]
    calibration_again, calibration_failed = confirmed(checks, calibration, True)
    print(json.dumps({"summary": "CED ports per call", "rows": rows + again}))
    print(json.dumps({"bits": "CED metadata and packing", "differs_at": []}))
    print(
        json.dumps(
            {
                "calibration": "Triton against itself, gated copies",
                "cases": len(calibration),
                "slower": sum(map(slower, calibration)),
                "slower_twice": len(calibration_failed),
                "faster": sum(r["change_pct"] + r["ci95_pct"] < 0 for r in calibration),
            }
        )
    )
    if failed:
        raise RuntimeError(f"CED ports are slower twice (95% intervals): {failed}")


if __name__ == "__main__":
    main()
