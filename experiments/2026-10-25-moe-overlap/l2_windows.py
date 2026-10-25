#!/usr/bin/env python3
"""How the L2 prefetch windows' consumers fare in a decode trace: for each projection a
window prefetches, its time when the window's prefetch kernel had finished before it
started and when it had not, and the excess over its L2-speed floor in each case.

usage: l2_windows.py TRACE.pt.trace.json.gz   (a rank's torch profiler trace of decode
                                               steps, e.g. request-inventory-5's)

A consumer is matched to the latest prefetch issued before it on the prefetch stream:
WO-A and WO-B to the WO window, the router and the shared expert to the FFN window, the
next layer's mHC projection, fused Q-A/KV and Q-B to the NEXT window. "While the prefetch
runs" is time a cache policy cannot recover (the lines are not there yet); "after it
finished" bounds what a policy that keeps prefetched lines could recover. The floor is
the consumer's 10th-percentile time among calls whose prefetch had finished.
"""
import bisect
import collections
import gzip
import json
import statistics
import sys

PREFETCH = "l2_prefetch_kernel"
# (kernel name, grid, block) at TP4 -> label. Grids over N tiles do not depend on rows.
CONSUMERS = {
    ("gemm_mxfp8_kernel", (28, 1, 1), (128, 1, 1)): "fused Q-A/KV",
    ("gemm_mxfp8_kernel", (64, 1, 1), (128, 1, 1)): "Q-B (and the indexer's)",
    ("gemm_fp8w_bf16_partials_kernel", (32, 4, 1), (128, 1, 1)): "WO-A",
    ("gemm_mxfp8_kernel", (40, 1, 1), (128, 1, 1)): "WO-B",
    ("gemm_mxfp8_kernel", (36, 1, 1), (128, 1, 1)): "shared gate/up",
    ("gemm_mxfp8_kernel", (40, 1, 1), (64, 1, 1)): "shared down",
    ("gemm_bf16_partials_kernel", (6, 8, 1), (256, 1, 1)): "router",
    ("mhc_project_streams_kernel", (40, 1, 1), (128, 1, 1)): "mHC projection (one row tile)",
}


def kernels(path: str) -> list[tuple]:
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    out = [
        (e["ts"], e["dur"], e.get("tid"), e["name"],
         tuple(e.get("args", {}).get("grid", ())), tuple(e.get("args", {}).get("block", ())))
        for e in events if e.get("cat") == "kernel" and "dur" in e
    ]  # fmt: skip
    return sorted(out)


def main(path: str) -> None:
    ks = kernels(path)
    steps = sum(1 for k in ks if k[3].startswith("combine_sampled_and_draft_tokens"))
    prefetch_stream = collections.Counter(k[2] for k in ks if k[3] == PREFETCH)
    stream = prefetch_stream.most_common(1)[0][0]
    busy, current = 0.0, None
    for start, end in sorted((ts, ts + dur) for ts, dur, tid, *_ in ks if tid != stream):
        if current is None or start > current[1]:
            busy += current[1] - current[0] if current else 0.0
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    busy += current[1] - current[0]
    print(f"{steps} steps; GPU busy outside the prefetch stream {busy / steps / 1000:.2f} ms a step")
    last = None
    calls = collections.defaultdict(lambda: {"finished": [], "running": []})
    for ts, dur, tid, name, grid, block in ks:
        if name == PREFETCH and tid == stream:
            last = ts + dur
            continue
        label = CONSUMERS.get((name, grid, block))
        if label is not None and last is not None:
            calls[label]["finished" if last <= ts else "running"].append(dur)
    totals = [0.0, 0.0]
    for label, split in calls.items():
        finished = sorted(split["finished"])
        floor = finished[len(finished) // 10] if finished else 0.0
        typical = statistics.median(finished) if finished else 0.0
        running = sum(max(0.0, d - typical) for d in split["running"])
        after = sum(max(0.0, d - floor) for d in finished)
        totals[0] += running
        totals[1] += after
        print(f"  {label:30} floor {floor:5.1f} us; finished {len(finished):4d} calls, p50 "
              f"{typical:5.1f}; running {len(split['running']):4d} calls; excess a step: "
              f"{running / steps:6.1f} us while the prefetch runs, {after / steps:6.1f} us after")
    print(f"excess a step: {totals[0] / steps / 1000:.2f} ms while the prefetch runs "
          f"({100 * totals[0] / busy:.1f}% of GPU busy), {totals[1] / steps / 1000:.2f} ms after "
          f"it finished ({100 * totals[1] / busy:.1f}%)")
    # The shared expert's down projection beside the routed FC1, and the indexer's Q-B.
    fc1 = [ts for ts, dur, tid, name, grid, block in ks
           if name.startswith("moe_grouped_fp8_fp4_gemm") and grid[:1] == (9,)]
    beside, before = [], []
    for ts, dur, tid, name, grid, block in ks:
        if CONSUMERS.get((name, grid, block)) == "shared down":
            i = bisect.bisect_left(fc1, ts)
            (beside if i < len(fc1) and fc1[i] < ts + dur else before).append(dur)
    print(f"shared down: p50 {statistics.median(beside):.1f} us beside FC1 ({len(beside)}), "
          f"{statistics.median(before):.1f} us before it ({len(before)})")
    qb = [(ts, dur) for ts, dur, tid, name, grid, block in ks
          if CONSUMERS.get((name, grid, block)) == "Q-B (and the indexer's)"]
    indexer = [dur for i, (ts, dur) in enumerate(qb) if i and ts - qb[i - 1][0] < 80]
    if indexer:
        print(f"indexer Q-B (outside the windows): {len(indexer) / steps:.1f} a step, p50 "
              f"{statistics.median(indexer):.1f} us")


if __name__ == "__main__":
    main(sys.argv[1])
