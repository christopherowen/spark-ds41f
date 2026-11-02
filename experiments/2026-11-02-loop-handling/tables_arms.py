#!/usr/bin/env python3
"""The loop soak's receipts for each arm: the known-trigger probes and the concurrent load.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/bench/RUN-<arm>/ and results/private/determinism/RUN/)

Per arm: each probe reply's verdict (probes.py), the probe summary, the 16-stream distinct
load's summary, and one-stream prose decode from the lean bench. Speeds are against the arm
labelled c37 when there is one, else the first.
"""
import json
import os
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
B = f"results/private/bench/{RUN}-{{}}/bench.json"
D = f"results/private/determinism/{RUN}/{{}}-{{}}.jsonl"
REFERENCE = "c37" if "c37" in ARMS else ARMS[0]


def lines(kind, arm):
    path = D.format(kind, arm)
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip().startswith("{")]


def prose_tps(arm):
    path = B.format(arm)
    if not os.path.exists(path):
        return None
    point = json.load(open(path))["suites"].get("decode", {}).get("points", {}).get("prose-c1")
    return point["tps"]["mean"] if point else None


def change(value, base):
    return f" {100 * (value - base) / base:+5.1f}%" if value and base else ""


print("known-trigger probes (ok, finish, longest run, completion tokens)")
for arm in ARMS:
    rows = lines("probes", arm)
    summary = next((r["summary"] for r in rows if "summary" in r), None)
    print(f"  {arm}: " + (f"{summary['replies']} replies, {summary['failed']} failed"
                          if summary else "no summary (the probe did not finish)"))
    for r in rows:
        if "id" in r:
            print(f"    {r['id']:20s} seed {r['seed']:<3} {'ok  ' if r['ok'] else 'FAIL'} "
                  f"{r['finish_reason']:8s} run {r['longest_run']:5d} tokens {r['completion_tokens']}"
                  + (f"  {r['why']}" if r["why"] else ""))
print("16 distinct streams, 1,024 tokens (tok/s ±95%)")
base = next((r["tps_mean"] for r in lines("c16-distinct", REFERENCE) if "tps_mean" in r), None)
for arm in ARMS:
    s = [r for r in lines("c16-distinct", arm) if "tps_mean" in r]
    print(f"  {arm:12s} {s[-1]['tps_mean']:7.2f} ±{s[-1]['ci95_pct']:.1f}%{change(s[-1]['tps_mean'], base)}"
          if s else f"  {arm:12s} (no summary)")
print("one-stream prose decode (tok/s)")
base = prose_tps(REFERENCE)
for arm in ARMS:
    tps = prose_tps(arm)
    print(f"  {arm:12s} {tps:7.2f}{change(tps, base)}" if tps else f"  {arm:12s} (no bench)")
