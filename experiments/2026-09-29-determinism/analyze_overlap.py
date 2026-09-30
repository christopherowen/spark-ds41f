#!/usr/bin/env python3
"""Which kernels run beside the shared-expert kernels on the side stream? (torch trace)

usage: analyze_overlap.py TRACE.json.gz

The side stream is the one running vllm::act_and_mul (the shared-expert
activation). For every side-stream kernel it lists the other streams' kernels
whose execution overlaps it, and totals overlap by kernel name.
"""
import gzip
import json
import sys
from collections import Counter, defaultdict

events = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
by_stream = defaultdict(list)
for e in kernels:
    by_stream[e.get("tid")].append(e)
side = Counter(e.get("tid") for e in kernels if "act_and_mul" in e["name"]).most_common(1)[0][0]
print("side stream", side, "kernels", len(by_stream[side]))
side_names = Counter(e["name"][:70] for e in by_stream[side])
print("side-stream kernel kinds:", side_names.most_common(6))
others = sorted((e for e in kernels if e.get("tid") != side), key=lambda e: e["ts"])
starts = [e["ts"] for e in others]
import bisect
overlap = defaultdict(float)
count = Counter()
for s in by_stream[side]:
    s0, s1 = s["ts"], s["ts"] + s["dur"]
    i = bisect.bisect_left(starts, s0 - 5000)
    kind = "down/gate GEMM" if "DenseGemm" in s["name"] else s["name"][:40]
    while i < len(others) and others[i]["ts"] < s1:
        o = others[i]
        o0, o1 = o["ts"], o["ts"] + o["dur"]
        if o1 > s0 and o0 < s1:
            key = (kind, o["name"][:60], o.get("tid"))
            overlap[key] += min(o1, s1) - max(o0, s0)
            count[key] += 1
        i += 1
for key, us in sorted(overlap.items(), key=lambda kv: -kv[1])[:25]:
    print(f"{us / 1000:8.2f} ms {count[key]:5d}x  side {key[0]:22s} vs tid {key[2]}: {key[1]}")
