#!/usr/bin/env python3
"""Attribute each measured prefill chunk's GPU time from a capture_depth.py trace.

usage: analyze_trace.py TRACE.json.gz [--top N]

The profiler window holds the four measured requests, in depth order, one
second apart. GPU kernels are grouped into requests at gaps longer than
300 ms. For each request: wall time from first kernel start to last kernel
end, the union of kernel-busy time, and kernel time per category.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

DEPTHS = (8192, 65536, 131072, 200704)
CATEGORIES = (
    # Score, select-prepare, top-k and sort kernels all live in b12x dsa_indexer.
    ("indexer", re.compile(r"dsa_indexer|^_pages$")),
    ("sparse_mla", re.compile(r"sparse_mla|compressed_sparse|mla|Attention", re.I)),
    ("moe", re.compile(r"moe|expert|w4a8", re.I)),
    ("collective", re.compile(r"nccl|all_?reduce|all_?gather|reduce_?scatter|roce", re.I)),
    ("dense_gemm", re.compile(r"gemm|dense|linear|matmul|mhc|quant", re.I)),
)


def category(name: str) -> str:
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as handle:
    events = json.load(handle)["traceEvents"]
kernels = sorted(
    (e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"),
    key=lambda e: e["ts"],
)
groups, current, last_end = [], [], None
for event in kernels:
    if last_end is not None and event["ts"] - last_end > 300_000:
        groups.append(current)
        current = []
    current.append(event)
    last_end = max(last_end or 0, event["ts"] + event["dur"])
if current:
    groups.append(current)
print(f"{len(kernels)} kernels in {len(groups)} groups")
if len(groups) != len(DEPTHS):
    print("warning: expected one group per depth", DEPTHS)
labels = [label for label, _ in CATEGORIES] + ["other"]
print("| depth | wall ms | busy ms | " + " | ".join(labels) + " |")
print("|---" * (3 + len(labels)) + "|")
per_name_last = None
for depth, group in zip(DEPTHS + (None,) * len(groups), groups):
    wall = (max(e["ts"] + e["dur"] for e in group) - group[0]["ts"]) / 1000
    busy = union((e["ts"], e["ts"] + e["dur"]) for e in group) / 1000
    by_category = defaultdict(float)
    by_name = defaultdict(lambda: [0.0, 0])
    for event in group:
        by_category[category(event["name"])] += event["dur"] / 1000
        by_name[event["name"]][0] += event["dur"] / 1000
        by_name[event["name"]][1] += 1
    cells = " | ".join(f"{by_category[label]:.1f}" for label in labels)
    print(f"| {depth} | {wall:.1f} | {busy:.1f} | {cells} |")
    per_name_last = (depth, by_name)
depth, by_name = per_name_last
print(f"\nTop kernels at depth {depth} (ms, count, category):")
for name, (ms, count) in sorted(by_name.items(), key=lambda item: -item[1][0])[:top]:
    print(f"{ms:9.2f} {count:6d} {category(name):11s} {name[:150]}")
