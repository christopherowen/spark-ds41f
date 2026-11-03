#!/usr/bin/env python3
"""The device-limit runs' receipts for each arm: copy probe, single-stream suite, the
host limit reads of every rank (overlay/limit-reads), and distinct concurrent streams.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/determinism/RUN/)

A limit-reads line per node: its last COPY-LIMITS counts, and whether the ranks' CRCs of
consequential reads agree window by window.
"""
import json
import os
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
D = f"results/private/determinism/{RUN}/{{}}-{{}}.jsonl"
COUNTS = ("reads", "queued", "stale", "consequential", "wide_slots", "placeholder_rows",
          "narrow_slots", "dropped_drafts", "unresolved")


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
    for stem in ("reads", "reads-c8", "reads-c16"):
        records = lines(stem, arm)
        for record in records:
            last = record["last"]
            counts = " ".join(f"{k}={last[k]}" for k in COUNTS if k in last)
            print(f"  {stem} {record['node']}: {record['windows']} windows, {counts or '-'}; "
                  f"{len(record['trouble'])} assert/traceback lines")
        if len(records) > 1:
            crcs = [dict(map(tuple, r["crcs"])) for r in records]
            shared = set.intersection(*(set(c) for c in crcs))
            differ = sorted(int(w) for w in shared if len({c[w] for c in crcs}) > 1)
            print(f"  {stem} ranks: {len(shared)} shared windows, CRCs differ in "
                  f"{len(differ)}{' (first at reads=' + str(differ[0]) + ')' if differ else ''}")
    for stem in ("c8-distinct", "c16-distinct"):
        s = [r for r in lines(stem, arm) if "tps_mean" in r]
        if s:
            print(f"  {stem}: {s[-1]['tps_mean']:.2f} tok/s ±{s[-1]['ci95_pct']:.1f}%")
