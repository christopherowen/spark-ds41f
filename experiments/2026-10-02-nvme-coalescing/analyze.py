#!/usr/bin/env python3
"""Summarize the NVMe coalescing A/B: bench points per arm, latency, interrupt and read deltas."""
import json
import os
import sys
from collections import defaultdict

repo = os.path.expanduser("~/projects/spark3-vllm-ds41f/results/private/bench")
res = sys.argv[1]
arms = ["1-on", "2-off", "3-on"]
m = lambda x: (x.get("mean") if isinstance(x, dict) else x)  # noqa: E731


def load(name):
    p = os.path.join(repo, name, "bench.json")
    return json.load(open(p)) if os.path.exists(p) else None


rows = defaultdict(dict)
for arm in arms:
    b = load(f"nvme-{arm}")
    if b:
        for k, p in b["suites"]["decode"]["points"].items():
            rows[f"{k} tps"][arm] = m(p["tps"])
            if m(p.get("step_ms")) is not None:
                rows[f"{k} step_ms"][arm] = m(p["step_ms"])
            rows[f"{k} ttft_ms"][arm] = m(p["ttft_s"]) * 1000
            if m(p.get("accepted_per_draft")) is not None:
                rows[f"{k} accepted/draft"][arm] = m(p["accepted_per_draft"])
        for k, p in b["suites"]["prefill"]["points"].items():
            rows[f"source prefill {k} tok/s"][arm] = m(p["prefill_tps"])
    n = load(f"nvme-{arm}-novel")
    if n:
        for k, p in n["suites"]["prefill"]["points"].items():
            rows[f"novel prefill {k} tok/s"][arm] = m(p["prefill_tps"])
print(f"{'metric':38} " + " ".join(f"{a:>10}" for a in arms) + "   off vs mean(on)")
for k in sorted(rows):
    v = rows[k]
    cells = " ".join(f"{v.get(a, float('nan')):10.2f}" for a in arms)
    on = [v[a] for a in ("1-on", "3-on") if a in v]
    d = f"{100 * (v['2-off'] / (sum(on) / len(on)) - 1):+6.1f}%" if on and "2-off" in v else ""
    print(f"{k:38} {cells}   {d}")

print("\nlatency (us): qd1 p50/p99, burst32 p50/p99")
for line in open(os.path.join(res, "latency.jsonl")):
    arm, js = line.split(" ", 1)
    j = json.loads(js)
    print(f"  {arm:5} {j['host']}: qd1 {j['qd1_us']['p50']:7.1f} / {j['qd1_us']['p99']:7.1f}   burst {j['burst_us']['p50']:8.1f} / {j['burst_us']['p99']:8.1f}")

print("\ncounters per arm (bench + novel): nvme interrupts, reads, read MiB, duration")
c = defaultdict(dict)
for line in open(os.path.join(res, "counters.txt")):
    label, host, irq, reads, sectors, writes, wsectors, t = line.split()
    c[(label.rsplit("-", 1)[0], host)][label.rsplit("-", 1)[1]] = (int(irq), int(reads), int(sectors), float(t))
for (arm, host), d in sorted(c.items()):
    if "start" in d and "end" in d:
        s, e = d["start"], d["end"]
        dt = e[3] - s[3]
        print(f"  {arm:5} {host}: irq {e[0]-s[0]:9d} ({(e[0]-s[0])/dt:7.0f}/s)  reads {e[1]-s[1]:9d} ({(e[1]-s[1])/dt:7.0f}/s)  "
              f"{(e[2]-s[2])*512/2**20:8.0f} MiB  irq/read {(e[0]-s[0])/max(1, e[1]-s[1]):.2f}  {dt:5.0f} s")
