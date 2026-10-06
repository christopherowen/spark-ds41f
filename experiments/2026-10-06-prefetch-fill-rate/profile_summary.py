#!/usr/bin/env python3
"""Hot all-reduce and kernel-group timings from the lab's decode profiles, per arm and rank.

usage: profile_summary.py ROOT ARM [ARM...] > summary.json
       (ROOT/<arm>/decode-trace/<node>/*.json.gz, as lab.py's profile job writes them)

Per rank: the profiled window's span, the 5th/50th/95th percentile of the graph-replayed
4-block one-shot all-reduce split by the kernel before it on the same stream (the MoE
all-reduce follows an elementwise kernel, the attention all-reduce the dense GEMM), and
total kernel time per group. TileLang names its kernels main_kernel, so those are grouped
by launch shape.
"""
import collections
import glob
import gzip
import json
import sys


def pct(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 1) if values else None


def group(event):
    name, args = event["name"], event.get("args", {})
    if name.startswith("main_kernel"):
        return (f"tilelang grid {tuple(args.get('grid', []))} block {tuple(args.get('block', []))} "
                f"regs {args.get('registers per thread')} smem {args.get('shared memory')}")
    lowered = name.lower()
    for needle, label in (("prefetch", "l2 prefetch"), ("oneshot", "one-shot collective"),
                          ("dense_gemm", "dense gemm"), ("gemv", "gemv"), ("mla", "attention"),
                          ("attention", "attention"), ("mhc", "mhc"), ("nccl", "nccl")):
        if needle in lowered:
            return label
    return "other"


def rank_summary(path):
    events = json.load(gzip.open(path, "rt"))["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
    streams = collections.defaultdict(list)
    for event in kernels:
        streams[event["args"].get("stream")].append(event)
    moe, attention = [], []
    for ordered in streams.values():
        for before, event in zip(ordered, ordered[1:]):
            name = event["name"].lower()
            if "oneshot" in name and "gather" not in name and event["args"].get("grid", [0])[0] == 4:
                if "elementwise" in before["name"]:
                    moe.append(event["dur"])
                elif "dense_gemm" in before["name"]:
                    attention.append(event["dur"])
    groups = collections.Counter()
    for event in kernels:
        groups[group(event)] += event["dur"] / 1000
    return {"span_ms": round((max(e["ts"] + e["dur"] for e in kernels) - kernels[0]["ts"]) / 1000, 1),
            "moe_allreduce_us_p5_p50_p95": [pct(moe, q) for q in (0.05, 0.5, 0.95)],
            "attention_allreduce_us_p5_p50_p95": [pct(attention, q) for q in (0.05, 0.5, 0.95)],
            "kernel_ms_by_group": {k: round(v, 1) for k, v in groups.most_common()}}


def main():
    root, arms = sys.argv[1], sys.argv[2:]
    out = {"what": __doc__.split("\n\n")[0].strip(), "arms": {}}
    for arm in arms:
        out["arms"][arm] = {path.split("/")[-2]: rank_summary(path)
                            for path in sorted(glob.glob(f"{root}/{arm}/decode-trace/*/*.json.gz"))}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
