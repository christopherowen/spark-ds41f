#!/usr/bin/env python3
"""Per-step GPU time of a profile_mixed4.py trace (one rank's torch profiler trace).

usage: analyze_profile.py TRACE.json.gz

Kernels are attributed to engine steps through their launch calls: a kernel belongs to the
step whose `execute_context_*` annotation was the latest to start before its launch. Per step:
the main-stream GPU span, busy time (union of kernel intervals) and idle time, then kernel time
by category for the step carrying the prompt against the next plain step.
"""
import bisect
import collections
import gzip
import json
import sys

CATEGORIES = (
    ("l2_prefetch", "l2 prefetch (side stream)"), ("moe", "routed moe"), ("sparse_mla", "sparse mla"),
    ("indexer", "indexer"), ("compress", "compressor"), ("gemm", "dense gemm"),
    ("allreduce", "collectives"), ("allgather", "collectives"), ("nccl", "collectives"),
    ("mhc", "mhc"), ("engram", "engram"), ("metadata", "metadata"), ("cache_writer", "kv write"),
    ("vocab", "vocab head"), ("router", "router"), ("norm", "norm"), ("rope", "rope"),
    ("cast", "quant/cast"), ("swiglu", "activation"), ("argmax", "sampler"), ("reject", "sampler"),
)


def category(name: str) -> str:
    low = name.lower()
    return next((label for key, label in CATEGORIES if key in low), "other")


def union_ms(kernels) -> float:
    total, start, end = 0.0, None, None
    for s, e in sorted((k["ts"], k["ts"] + k["dur"]) for k in kernels):
        if start is None:
            start, end = s, e
        elif s <= end:
            end = max(end, e)
        else:
            total, start, end = total + end - start, s, e
    return (total + (end - start if start is not None else 0)) / 1000


def main():
    events = json.load(gzip.open(sys.argv[1]))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
    starts = [e["ts"] for e in steps]
    by_step = collections.defaultdict(list)
    for k in kernels:
        launch = launches.get(k.get("args", {}).get("correlation"))
        if launch is not None and (i := bisect.bisect_right(starts, launch["ts"]) - 1) >= 0:
            by_step[i].append(k)
    prompt = None
    for i, step in enumerate(steps):
        main_stream = [k for k in by_step[i] if "l2_prefetch" not in k["name"]]
        if not main_stream:
            continue
        span = (max(k["ts"] + k["dur"] for k in main_stream) - min(k["ts"] for k in main_stream)) / 1000
        busy = union_ms(main_stream)
        print(f"{i} {step['name']:40s} host {step['dur'] / 1000:6.1f}  gpu span {span:6.1f} "
              f"busy {busy:6.1f} idle {span - busy:5.1f} ms")
        if prompt is None and not step["name"].startswith("execute_context_0("):
            prompt = i
    if prompt is None or prompt + 1 not in by_step:
        return
    tables = []
    for i in (prompt, prompt + 1):
        table = collections.Counter()
        for k in by_step[i]:
            table[category(k["name"])] += k["dur"] / 1000
        tables.append(table)
    print(f"\n{'category':26s} {'prompt step':>11s} {'next step':>10s}")
    for name in sorted(set(tables[0]) | set(tables[1]), key=lambda n: -tables[0][n]):
        print(f"{name:26s} {tables[0][name]:11.1f} {tables[1][name]:10.1f}")


if __name__ == "__main__":
    main()
