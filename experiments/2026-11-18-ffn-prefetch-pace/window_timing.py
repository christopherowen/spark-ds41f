#!/usr/bin/env python3
"""Where each L2 prefetch window of a target forward ends against the kernels that read it.

usage: window_timing.py TRACE.json.gz

Reads one rank's torch-profiler trace of a decode window (the single-stream floor
experiment's, experiments/2026-11-17-c1-floor-now) and takes every target forward (a
CUDA graph with 80 routed-expert GEMMs). In each of its 40 layers it names, by order:

  WO-B        the first MXFP8 GEMM after WO-A's partials
  AR1, AR2    the all-reduces after attention and after the routed experts
  shared up   the first MXFP8 GEMM after AR1 starts
  P1, P2, P3  the layer's prefetch kernels: the one that starts before WO-B ends (WO or
              COMP), the next before the routed FC1 (FFN), the first after FC2 (NEXT)

and prints the percentiles of their relative times in microseconds. Then, a layer at a
time, it splits the forwards at the median of "P2 ends after the shared up-projection
starts" and prints that projection's duration and its end after AR1's in the two halves:
what a late FFN fill costs the kernel it fills for. The same split for P3 and the next
layer's first two MXFP8 GEMMs (the fused Q-A/KV and Q-B).
"""
import gzip
import json
import statistics as st
import sys
from collections import defaultdict


def kernels(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    launched = {e["args"]["correlation"]: e["name"] for e in events
                if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    rows = [(e["ts"], e["dur"], e["name"], e["args"].get("stream"), e["args"].get("correlation"),
             tuple(e["args"].get("grid") or ()))
            for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    rows.sort()
    graphs = defaultdict(list)
    for row in rows:
        if launched.get(row[4]) == "cudaGraphLaunch":
            graphs[row[4]].append(row)
    return [g for g in graphs.values()
            if sum(r[2] == "moe_grouped_fp8_fp4_gemm_kernel" for r in g) == 80]


def end(row):
    return row[0] + row[1]


def main(path):
    stats, split = defaultdict(list), defaultdict(list)
    forwards = sorted(kernels(path), key=lambda g: g[0][0])
    for g in forwards:
        main_stream = max({r[3] for r in g}, key=lambda s: sum(r[3] == s for r in g))
        main = [r for r in g if r[3] == main_stream]
        fills = [r for r in g if r[2] == "l2_prefetch_kernel"]
        mla = [r for r in main if r[2] == "sparse_mla_kernel"]
        fc = [r for r in g if r[2] == "moe_grouped_fp8_fp4_gemm_kernel"]
        if len(mla) != 40:
            continue
        for layer in range(40):
            m, fc1, fc2 = mla[layer], fc[2 * layer], fc[2 * layer + 1]
            nxt = mla[layer + 1][0] if layer < 39 else end(g[-1])
            seg = [r for r in main if m[0] <= r[0] < nxt]
            wo_a = next(r for r in seg if r[2] == "gemm_fp8w_bf16_partials_kernel")
            wo_b = next(r for r in seg if r[2] == "gemm_mxfp8_kernel" and r[0] > wo_a[0])
            ars = [r for r in seg if r[2] == "oneshot_allreduce_kernel"]
            ar1 = ars[0]
            ar2 = next((r for r in ars if r[0] > fc2[0]), None)
            shared = next((r for r in seg if r[2] == "gemm_mxfp8_kernel" and r[0] > ar1[0]), None)
            mine = [r for r in fills if m[0] - 3 <= r[0] < nxt]
            p1 = next((r for r in mine if r[0] < end(wo_b)), None)
            p2 = next((r for r in mine if p1 and p1[0] < r[0] < fc1[0] + 50), None)
            p3 = next((r for r in mine if r[0] > fc2[0]), None)
            stats["layer period"].append(nxt - m[0])
            stats["WO-B duration"].append(wo_b[1])
            stats["AR1 ends after WO-B ends"].append(end(ar1) - end(wo_b))
            if p1:
                stats["P1 (WO) duration"].append(p1[1])
                stats["P1 ends after WO-B ends"].append(end(p1) - end(wo_b))
            if p1 and p2 and shared:
                stats["P2 (FFN) duration"].append(p2[1])
                stats["P2 starts after WO-B ends"].append(p2[0] - end(wo_b))
                stats["P2 waits behind P1"].append(max(0.0, end(p1) - end(wo_b)))
                stats["P2 ends after shared up starts"].append(end(p2) - shared[0])
                stats["P2 ends after routed FC1 starts"].append(end(p2) - fc1[0])
                stats["shared up duration"].append(shared[1])
                split["shared up", layer].append((end(p2) - shared[0], shared[1], end(shared) - end(ar1)))
            if p3 and ar2:
                stats["P3 (NEXT) duration"].append(p3[1])
                stats["P3 ends after AR2 ends"].append(end(p3) - end(ar2))
                if layer < 39:
                    after = [r for r in main if ar2[0] < r[0] < mla[layer + 1][0]
                             and r[2] == "gemm_mxfp8_kernel"]
                    for name, row in zip(("next Q-A/KV", "next Q-B"), after):
                        split[name, layer].append((end(p3) - row[0], row[1], end(row) - end(ar2)))
    print(f"{len(forwards)} target forwards")
    print(f"{'microseconds':34} {'n':>6} {'p10':>8} {'p50':>8} {'p90':>8} {'mean':>8}")
    for name, values in stats.items():
        values = sorted(values)
        pick = lambda p: values[min(len(values) - 1, int(p * len(values)))]
        print(f"{name:34} {len(values):6d} {pick(.1):8.1f} {pick(.5):8.1f} {pick(.9):8.1f} "
              f"{st.mean(values):8.1f}")
    print("\nA layer's forwards split at the median of 'the fill ends after the consumer starts':")
    for name in ("shared up", "next Q-A/KV", "next Q-B"):
        early, late = defaultdict(list), defaultdict(list)
        for (kind, _), rows in split.items():
            if kind != name or len(rows) < 50:
                continue
            rows = sorted(rows)
            half = len(rows) // 2
            for column in range(3):
                early[column].append(st.mean(r[column] for r in rows[:half]))
                late[column].append(st.mean(r[column] for r in rows[half:]))
        print(f"  {name:12} {len(early[0])} layers: fill ends {st.mean(early[0]):+.1f} / "
              f"{st.mean(late[0]):+.1f} us after its start; duration {st.mean(early[1]):.1f} / "
              f"{st.mean(late[1]):.1f} us; ends {st.mean(early[2]):.1f} / {st.mean(late[2]):.1f} us "
              "after the all-reduce")


if __name__ == "__main__":
    main(sys.argv[1])
