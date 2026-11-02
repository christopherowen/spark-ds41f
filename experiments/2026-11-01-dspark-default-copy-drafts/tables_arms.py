#!/usr/bin/env python3
"""The #842 crash repro's receipts for each arm: the census, the copy probe and the loads.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/determinism/RUN/)

Per arm, in run order: the copy probe's tokens a step per category (copy_probe.py), the
single-stream suite's exit (suite_single_stream.py), then each census collection's
NONFINITE and assert/traceback lines per node (census_logs.py: ``census`` after the
suite, ``census-c8`` after the 8-stream load), and the distinct loads' summaries
(scripts/distinct_streams.py).
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
        print(f"  single-stream suite: exit {record['exit']} ({record['output']})")
    for stem in ("census", "census-c8"):
        for record in lines(stem, arm):
            print(f"  {stem} {record['node']}: {record['nonfinite']} NONFINITE lines, "
                  f"{len(record['trouble'])} assert/traceback lines")
            for line in record["lines"][:12]:
                print(f"    {line[-260:]}")
    for stem in ("c8-distinct", "c16-distinct"):
        for record in lines(stem, arm)[-1:]:
            print(f"  {stem}: {json.dumps(record)[:240]}")
