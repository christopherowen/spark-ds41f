#!/usr/bin/env python3
"""Where a traced prefill's time goes, and how much of its communication is exposed.

usage: prefill_shares.py TRACE [--json OUT]

TRACE is one rank's Kineto trace of a prefill (the lab's profile job, prefill workload).
Over the window from the first to the last kernel it splits wall time by what runs:

- exposed communication: only collective kernels (sequence-parallel exchange, all-gathers,
  all-reduces) run, no compute;
- compute under communication: both run (the exchange hidden under compute);
- compute only;
- idle: nothing runs (host gaps, the Engram wait).

It also gives each kernel family's busy time (the union of its kernels), so the routed MoE,
attention and communication shares of a prefill chunk can be tracked per release. The SP
exchange share the brief asks for is the exposed-communication line; a change that moves
the exchange under compute (vLLM 0505's combine overlap, #593) moves time from it into
"compute under communication". Lab-only, standard library only.
"""
import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

COMM = ("ncclDevKernel", "oneshot_allreduce", "oneshot_allgather", "collectives_rank_order_sum")
FAMILIES = (
    ("routed MoE", ("moe_grouped", "moe_combine", "moe_dispatch", "moe_swiglu")),
    ("attention", ("sparse_mla", "indexer", "metadata", "cache_writer", "rope", "compress")),
    ("dense GEMMs", ("gemm_",)),
    ("mHC", ("mhc_",)),
    ("communication", COMM),
    ("L2 prefetch", ("l2_prefetch",)),
    ("copies", ("Memcpy", "Memset")),
)


def family(name: str) -> str:
    for label, prefixes in FAMILIES:
        if name.startswith(prefixes):
            return label
    return "other"


def union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def split(kernels: list[tuple[float, float, str]]) -> dict:
    start = min(k[0] for k in kernels)
    stop = max(k[0] + k[1] for k in kernels)
    edges = []
    for ts, dur, name in kernels:
        comm = name.startswith(COMM)
        edges += [(ts, 1, comm), (ts + dur, -1, comm)]
    edges.sort(key=lambda e: (e[0], e[1]))
    active = {True: 0, False: 0}
    last, out = start, defaultdict(float)
    for t, kind, comm in edges:
        if t > last:
            c, x = active[True] > 0, active[False] > 0
            out["exposed communication" if c and not x else "compute under communication" if c and x
                else "compute only" if x else "idle"] += t - last
            last = t
        active[comm] += kind
    span = stop - start
    by_family = defaultdict(list)
    for ts, dur, name in kernels:
        by_family[family(name)].append((ts, ts + dur))
    return {"span_ms": span / 1e3, "split_ms": {k: v / 1e3 for k, v in out.items()},
            "split_share": {k: v / span for k, v in out.items()},
            "family_busy_ms": {k: union(v) / 1e3 for k, v in sorted(by_family.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = [(e["ts"], e["dur"], e["name"]) for e in events
               if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    out = split(kernels)
    print(f"window {out['span_ms']:.1f} ms")
    for key in ("compute only", "compute under communication", "exposed communication", "idle"):
        print(f"  {key:28s} {out['split_ms'].get(key, 0):8.1f} ms  {100 * out['split_share'].get(key, 0):5.1f}%")
    print("busy time by family (union of its kernels):")
    for key, value in sorted(out["family_busy_ms"].items(), key=lambda kv: -kv[1]):
        print(f"  {key:28s} {value:8.1f} ms  {100 * value / out['span_ms']:5.1f}%")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
