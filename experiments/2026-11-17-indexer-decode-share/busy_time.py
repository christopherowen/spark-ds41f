#!/usr/bin/env python3
"""GPU busy and idle time of every profiler window of one rank, with kernel overlap removed.

usage: busy_time.py OUT.json TRACE_DIR

Kernel durations overlap (a kernel launched with programmatic dependent launch starts
before its predecessor ends, and the prefetch kernel runs beside the others), so the sum
of durations exceeds the window. For every trace of TRACE_DIR, in window order, this
writes per family (FAMILIES, by kernel-name prefix):

- sum_ms: the durations added up;
- owned_ms: each instant counted once, for the running kernel that started first;
- alone_ms: the time no kernel of another family is running;

and for the window busy_ms (some kernel running), span_ms (first kernel start to last
kernel end), idle_ms, launches and steps (launches of STEP_KERNEL, which runs once a step).
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

FAMILIES = (("indexer_score", "indexer score"), ("indexer_topk", "indexer top-k"), ("indexer_", "indexer other"),
            ("moe_", "MoE"), ("l2_prefetch", "prefetch"), ("oneshot_", "collectives"))
STEP_KERNEL = "mhc_post_collapse_kernel"


def family(name):
    return next((label for prefix, label in FAMILIES if name.startswith(prefix)), "other")


def union(intervals):
    """Total length of the union of (start, end) intervals sorted by start."""
    total, cursor = 0, None
    for start, end in intervals:
        if cursor is None or start > cursor:
            total += end - start
            cursor = end
        elif end > cursor:
            total += end - cursor
            cursor = end
    return total


out, directory = sys.argv[1], Path(sys.argv[2])
windows = []
for path in sorted(directory.glob("*.json*")):
    with (gzip.open if path.name.endswith(".gz") else open)(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in events
                     if e.get("cat") == "kernel" and e.get("ph") == "X")
    del events
    total = defaultdict(float)
    owned = defaultdict(float)
    cursor = kernels[0][0]
    for start, end, name in kernels:
        total[family(name)] += end - start
        if end > cursor:
            owned[family(name)] += end - max(start, cursor)
            cursor = end
    busy = union((start, end) for start, end, _ in kernels)
    alone = {label: busy - union((s, e) for s, e, name in kernels if family(name) != label) for label in total}
    span = max(end for _, end, _ in kernels) - kernels[0][0]
    windows.append({
        "trace": path.name, "first_kernel_ts": kernels[0][0], "launches": len(kernels),
        "steps": sum(name == STEP_KERNEL for _, _, name in kernels),
        "span_ms": round(span / 1000, 3), "busy_ms": round(busy / 1000, 3), "idle_ms": round((span - busy) / 1000, 3),
        "families": {label: {"sum_ms": round(total[label] / 1000, 3), "owned_ms": round(owned[label] / 1000, 3),
                             "alone_ms": round(alone[label] / 1000, 3)} for label in sorted(total)}})
    del kernels
windows.sort(key=lambda w: w["first_kernel_ts"])
json.dump({"windows": windows}, open(out, "w"), indent=1)
for w in windows:
    print(w["trace"], w["steps"], w["span_ms"], w["busy_ms"], w["idle_ms"])
