#!/usr/bin/env python3
"""Compare GPU kernel time between two decode traces.

usage: analyze_decode.py BASE_TRACE.json.gz CANDIDATE_TRACE.json.gz [--top N]

Sums kernel time by name in each trace (the whole profiler window, one request
each), prints totals and the kernels whose time differs most.
"""
import gzip
import json
import sys
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    by_name = defaultdict(lambda: [0.0, 0])
    for event in kernels:
        by_name[event["name"]][0] += event["dur"] / 1000
        by_name[event["name"]][1] += 1
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000
    return by_name, span


top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
base, base_span = load(sys.argv[1])
cand, cand_span = load(sys.argv[2])
total = lambda d: sum(v[0] for v in d.values())
print(f"base: {total(base):.1f} ms kernel time, {sum(v[1] for v in base.values())} kernels, span {base_span:.1f} ms")
print(f"cand: {total(cand):.1f} ms kernel time, {sum(v[1] for v in cand.values())} kernels, span {cand_span:.1f} ms")
names = set(base) | set(cand)
rows = sorted(names, key=lambda n: -abs(cand.get(n, [0, 0])[0] - base.get(n, [0, 0])[0]))
print(f"\n{'delta ms':>9} {'base ms':>9} {'n':>6} {'cand ms':>9} {'n':>6}  kernel")
for name in rows[:top]:
    b, c = base.get(name, [0.0, 0]), cand.get(name, [0.0, 0])
    print(f"{c[0] - b[0]:9.2f} {b[0]:9.2f} {b[1]:6d} {c[0]:9.2f} {c[1]:6d}  {name[:140]}")
