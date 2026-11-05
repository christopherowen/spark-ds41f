#!/usr/bin/env python3
"""The quality sweep's receipts: for each arm, the summary line of every workload it ran.

usage: tables_arms.py RUN ARM [ARM...]   (from the deployment checkout; RUN names
       results/private/determinism/RUN/)
"""
import glob
import json
import os
import sys

RUN, ARMS = sys.argv[1], sys.argv[2:]
for arm in ARMS:
    print(f"== {arm}")
    for path in sorted(glob.glob(f"results/private/determinism/{RUN}/*-{arm}.jsonl")):
        stem = os.path.basename(path)[: -len(f"-{arm}.jsonl")]
        records = [json.loads(x) for x in open(path) if x.strip().startswith("{")]
        summary = next((r for r in reversed(records) if "summary" in r), records[-1] if records else None)
        print(f"  {stem:24s} {json.dumps(summary)[:300] if summary else '(no output)'}")
