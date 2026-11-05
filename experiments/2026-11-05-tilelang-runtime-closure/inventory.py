#!/usr/bin/env python3
"""Account for every CUDA kernel, including graph nodes, in a full-model trace."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("trace", type=Path)
parser.add_argument("compiled", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
compiled = json.loads(args.compiled.read_text())
known = {name for row in compiled for name in row["kernels"]}
assert known, "no generated CUDA symbols: cannot establish TileLang provenance"
db = sqlite3.connect(args.trace)
launches = db.execute(
    "SELECT k.start, k.end, s.value, k.streamId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
).fetchall()
assert launches, "empty CUDA trace"
windows = {}
for marker, phase in (("iket_step_marker", "request"), ("iket_grid_marker", "grid")):
    boundaries = [(begin, end) for begin, end, name, _ in launches if name.removesuffix("_kernel") == marker]
    assert len(boundaries) == 2, (marker, boundaries)
    windows[phase] = (boundaries[0][1], boundaries[1][0])


def category(name):
    if name in known:
        return "tilelang"
    if "at::native" in name or name.startswith("_ZN2at6native"):
        return "aten"
    if "nccl" in name.lower():
        return "nccl"
    return "unknown"


phases = {}
for phase, (low, high) in {"whole_process": (0, float("inf")), **windows}.items():
    kernels = defaultdict(lambda: dict(count=0, duration_ns=0, streams=set()))
    totals = defaultdict(lambda: dict(count=0, duration_ns=0))
    count = 0
    for begin, end, name, stream in launches:
        if low <= begin and end <= high:
            row = kernels[name]
            row["count"] += 1
            row["duration_ns"] += end - begin
            row["streams"].add(stream)
            group = totals[category(name)]
            group["count"] += 1
            group["duration_ns"] += end - begin
            count += 1
    assert sum(row["count"] for row in totals.values()) == count
    phases[phase] = dict(launches=count, totals=dict(totals), kernels=[
        dict(name=name, category=category(name), **{**row, "streams": sorted(row["streams"])})
        for name, row in sorted(kernels.items())])

result = dict(trace=str(args.trace), trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              compiled_sha256=hashlib.sha256(args.compiled.read_bytes()).hexdigest(),
              known_symbols=len(known), phases=phases,
              scope="all CUDA compute launches in the recorded workload; startup separated by markers; DMA/memsets are not compute kernels")
args.out.write_text(json.dumps(result, indent=2) + "\n")
for name, phase in phases.items():
    print(name, phase["launches"], phase["totals"])
