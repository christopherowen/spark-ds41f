#!/usr/bin/env python3
"""The limit-sync runs' receipts for each arm: copy probe, single-stream suite, census
of non-finite MoE gate rows, and distinct concurrent streams.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/determinism/RUN/)
"""
import json
import os
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
D = f"results/private/determinism/{RUN}/{{}}-{{}}.jsonl"


def lines(kind, arm):
    path = D.format(kind, arm)
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip().startswith("{")]


for arm in ARMS:
    print(f"== {arm}")
    for record in lines("copy-probe", arm):
        print(f"  copy probe: {json.dumps(record)[:240]}")
    for record in lines("single-stream", arm):
        if record.get("summary") == "servebench closed":
            print(f"  single-stream suite: {record['tps_mean']:.2f} tok/s, "
                  f"{record['tokens_per_step']:.3f} tokens/step, {record['failed']} failed")
        elif "exit" in record:
            print(f"  single-stream suite: exit {record['exit']} ({record['output']})")
    for stem in ("census", "census-c8"):
        for record in lines(stem, arm):
            print(f"  {stem} {record['node']}: {record['nonfinite']} NONFINITE lines, "
                  f"{len(record['trouble'])} assert/traceback lines")
    for stem in ("c8-distinct", "c16-distinct"):
        s = [r for r in lines(stem, arm) if "tps_mean" in r]
        if s:
            print(f"  {stem}: {s[-1]['tps_mean']:.2f} tok/s ±{s[-1]['ci95_pct']:.1f}%")
