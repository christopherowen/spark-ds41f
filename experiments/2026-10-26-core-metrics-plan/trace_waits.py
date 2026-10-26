#!/usr/bin/env python3
"""Host gaps, collective waits and late peers, from the lab's torch-profiler traces.

usage: trace_waits.py TRACE [TRACE ...] [--json OUT]

Each TRACE is one rank's Kineto trace (*.pt.trace.json.gz) of the same window. With one
rank it reports what that rank sees; with every rank of the window it also aligns the
ranks' collectives and names the late peer.

- Steps run from one target step's embedding kernel to the next (the drafter's embedding
  kernel in between is part of the step); the first and last steps of a trace are dropped.
- Host gaps: time inside a step when no kernel, copy or set runs on any stream. Reported
  per step with the share of the step and the largest single gap.
- Collective waits: a one-shot all-reduce or all-gather ends at about the same moment on
  every rank (when the last peer's data lands), so each rank's kernel is its own staging
  and reduce plus the wait for the slowest peer. With every rank's trace, the collectives
  are matched by order (the index offset whose consecutive end-to-end intervals agree
  best), each rank's clock offset is the median difference of matched end times, and a
  rank's wait at a collective is the latest aligned start less its own. The late peer is
  the rank that started last. With one rank, the wait is estimated as each kernel's time
  above the 5th percentile of its kind.

Lab-only: it reads profiler traces, which need a profiling window. Standard library only.
"""
import argparse
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

MARKS = ("vocab_embedding_kernel", "void vllm::vocab_embedding::vocab_parallel_embedding_kernel")
COLLECTIVES = ("oneshot_allreduce", "oneshot_allgather", "ncclDev",
               "kernel_cutlass_kernel_b12xcommroce_oneshot", "kernel_cutlass_kernel_b12xcommroce_allgather")
MOE = ("moe_grouped", "kernel_cutlass_kernel_b12xmoe")


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    base_us = trace.get("baseTimeNanoseconds", 0) / 1000
    kernels = sorted((e["ts"] + base_us, e["dur"], e["name"]) for e in trace["traceEvents"]
                     if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    return {"path": str(path), "host": trace.get("host_name"), "kernels": kernels}


def steps(kernels: list) -> list[tuple[float, float]]:
    """(start, end) of each whole decode step: a target segment (the 40 layers' launches)
    and the drafter segment after it (3 layers', so far fewer)."""
    marks = [k[0] for k in kernels if k[2].startswith(MARKS)]
    counts, j = [], 0
    for a, b in zip(marks, marks[1:]):
        while j < len(kernels) and kernels[j][0] < a:
            j += 1
        n = 0
        while j + n < len(kernels) and kernels[j + n][0] < b:
            n += 1
        counts.append(n)
    out, i = [], 0
    while i + 2 < len(marks):
        if counts[i] > 3 * counts[i + 1] and counts[i + 1] > 0:
            out.append((marks[i], marks[i + 2]))
            i += 2
        else:
            i += 1
    return out[1:-1]


def union_gaps(kernels: list, start: float, end: float) -> tuple[float, float, int]:
    """Idle time in [start, end) with no kernel on any stream, the largest gap, and how many
    gaps exceed 5 us."""
    idle, largest, count, edge = 0.0, 0.0, 0, start
    for ts, dur, _ in kernels:
        if ts >= end:
            break
        if ts + dur <= start:
            continue
        a = max(ts, start)
        if a > edge:
            gap = a - edge
            idle += gap
            largest = max(largest, gap)
            count += gap > 5.0
        edge = max(edge, min(ts + dur, end))
    if end > edge:
        idle += end - edge
        largest = max(largest, end - edge)
        count += end - edge > 5.0
    return idle, largest, count


def collectives(kernels: list) -> list[tuple[float, float, str]]:
    return [k for k in kernels if k[2].startswith(COLLECTIVES)]


def per_rank(trace: dict) -> dict:
    ks = trace["kernels"]
    spans = steps(ks)
    if not spans:
        return {"host": trace["host"], "steps": 0}
    coll = collectives(ks)
    floors = {}
    for name in {c[2] for c in coll}:
        durations = sorted(c[1] for c in coll if c[2] == name)
        floors[name] = durations[max(0, math.ceil(0.05 * len(durations)) - 1)]
    rows = []
    for a, b in spans:
        idle, largest, count = union_gaps(ks, a, b)
        inside = [c for c in coll if a <= c[0] < b]
        rows.append({"step_us": b - a, "idle_us": idle, "largest_gap_us": largest, "gaps_over_5us": count,
                     "collectives": len(inside), "collective_us": sum(c[1] for c in inside),
                     "wait_estimate_us": sum(c[1] - floors[c[2]] for c in inside)})
    mean = lambda key: statistics.fmean(r[key] for r in rows)
    return {"host": trace["host"], "steps": len(rows), "step_ms": mean("step_us") / 1e3,
            "host_gap_ms": mean("idle_us") / 1e3, "host_gap_share": mean("idle_us") / mean("step_us"),
            "largest_gap_ms_p50": statistics.median(r["largest_gap_us"] for r in rows) / 1e3,
            "gaps_over_5us": mean("gaps_over_5us"), "collectives_per_step": mean("collectives"),
            "collective_ms": mean("collective_us") / 1e3, "wait_estimate_ms": mean("wait_estimate_us") / 1e3,
            "collective_floor_us": {name: round(v, 2) for name, v in sorted(floors.items())}}


def align(reference: list, other: list, window: int = 400) -> tuple[int, float] | None:
    """Index offset and clock offset that map ``other``'s collectives onto ``reference``'s:
    the offset whose consecutive end-to-end intervals agree best."""
    ends_r = [c[0] + c[1] for c in reference]
    ends_o = [c[0] + c[1] for c in other]
    dr = [b - a for a, b in zip(ends_r, ends_r[1:])]
    do = [b - a for a, b in zip(ends_o, ends_o[1:])]
    best = None
    for shift in range(-window, window + 1):
        pairs = [(dr[i], do[i + shift]) for i in range(len(dr)) if 0 <= i + shift < len(do)]
        if len(pairs) < 50:
            continue
        score = statistics.median(abs(x - y) for x, y in pairs)
        if best is None or score < best[0]:
            best = (score, shift)
    if best is None:
        return None
    shift = best[1]
    offset = statistics.median(ends_o[i + shift] - ends_r[i] for i in range(len(ends_r)) if 0 <= i + shift < len(ends_o))
    return shift, offset


def late_peers(traces: list[dict]) -> dict:
    """Waits and late peers over the collectives every rank's trace holds."""
    coll = [collectives(t["kernels"]) for t in traces]
    maps = [(0, 0.0)]
    for other in coll[1:]:
        found = align(coll[0], other)
        if found is None:
            return {"error": "could not align the ranks' collectives"}
        maps.append(found)
    spans = steps(traces[0]["kernels"])
    lo, hi = (spans[0][0], spans[-1][1]) if spans else (-math.inf, math.inf)
    ranks = len(traces)
    waits = [0.0] * ranks
    last = [0] * ranks
    skews, matched, mismatched = [], 0, 0
    for i, c0 in enumerate(coll[0]):
        if not lo <= c0[0] < hi:
            continue
        group = []
        for r, (shift, offset) in enumerate(maps):
            j = i + shift
            if not 0 <= j < len(coll[r]):
                group = None
                break
            c = coll[r][j]
            group.append((c[0] - offset, c[1], c[2]))
        if group is None:
            continue
        if len({g[2] for g in group}) != 1:
            mismatched += 1
            continue
        matched += 1
        starts = [g[0] for g in group]
        latest = max(starts)
        for r in range(ranks):
            waits[r] += latest - starts[r]
        last[starts.index(latest)] += 1
        skews.append(latest - min(starts))
    if not matched:
        return {"error": "no collectives matched"}
    n_steps = len(spans) or 1
    skews.sort()
    pick = lambda p: skews[min(len(skews) - 1, max(0, math.ceil(p / 100 * len(skews)) - 1))]
    return {"ranks": [t["host"] for t in traces], "matched": matched, "mismatched_names": mismatched,
            "clock_offset_us": [round(m[1], 1) for m in maps], "index_shift": [m[0] for m in maps],
            "wait_ms_per_step": [round(w / n_steps / 1e3, 3) for w in waits],
            "last_share": [round(x / matched, 3) for x in last],
            "skew_us": {"p50": round(pick(50), 2), "p90": round(pick(90), 2), "p99": round(pick(99), 2),
                        "mean": round(statistics.fmean(skews), 2)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    traces = [load(p) for p in args.traces]
    out = {"ranks": [per_rank(t) for t in traces]}
    print(f"{'rank host':10s} {'steps':>5} {'step ms':>8} {'gap ms':>7} {'gap %':>6} {'max gap':>8} {'gaps>5us':>8} "
          f"{'coll/stp':>8} {'coll ms':>8} {'wait~ms':>8}")
    for r in out["ranks"]:
        if not r["steps"]:
            print(f"{r['host'] or '?':10s} {0:5d}  no whole decode steps")
            continue
        print(f"{r['host'] or '?':10s} {r['steps']:5d} {r['step_ms']:8.2f} {r['host_gap_ms']:7.2f} "
              f"{100 * r['host_gap_share']:5.1f}% {r['largest_gap_ms_p50']:8.3f} {r['gaps_over_5us']:8.1f} "
              f"{r['collectives_per_step']:8.1f} {r['collective_ms']:8.2f} {r['wait_estimate_ms']:8.2f}")
    hosts = [t["host"] for t in traces]
    if len(traces) > 1 and len(set(hosts)) < len(hosts):
        print(f"\nlate peers: not computed, the traces are not one window's ranks (hosts {hosts})")
    elif len(traces) > 1:
        out["late_peers"] = late_peers(traces)
        lp = out["late_peers"]
        if "error" in lp:
            print(f"late peers: {lp['error']}")
        else:
            print(f"\nlate peers over {lp['matched']} matched collectives ({lp['mismatched_names']} name mismatches);"
                  f" clock offsets {lp['clock_offset_us']} us, index shifts {lp['index_shift']}")
            print(f"  skew (last start - first start) p50 {lp['skew_us']['p50']} us, p90 {lp['skew_us']['p90']}, "
                  f"p99 {lp['skew_us']['p99']}, mean {lp['skew_us']['mean']}")
            for host, wait, share in zip(lp["ranks"], lp["wait_ms_per_step"], lp["last_share"]):
                print(f"  {host}: waits {wait:.3f} ms a step for the last peer; arrives last at {100 * share:.1f}% of collectives")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
