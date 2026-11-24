#!/usr/bin/env python3
"""From a step trace: how did decode steps of long-context requests run?
Counts steps whose rows sit at position >= MINPOS by (graph mode, rows), and marks the row
counts where the last rank's index slice would be empty if the step is split.
usage: long_steps.py TRACE.jsonl TP [MINPOS=16384]"""
import collections, json, sys
tp = int(sys.argv[2]); minpos = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
c = collections.Counter()
for l in open(sys.argv[1]):
    if not l.strip(): continue
    s = json.loads(l)
    if s.get("big") or not s["positions"]: continue
    p = s["positions"]; p = [x[0] if isinstance(x, list) else x for x in p]
    if max(p) < minpos: continue
    c[(s["cg_mode"].split(".")[-1], s["n"], s["num_reqs"])] += 1
tot = collections.Counter()
for (mode, n, r), k in c.items(): tot[mode] += k
print(f"decode steps at position >= {minpos}: by graph mode {dict(tot)}")
for (mode, n, r), k in sorted(c.items()):
    empty = n >= tp and (tp - 1) * (-(-n // tp)) >= n
    print(f"  {mode:5s} rows {n:3d} reqs {r}: {k:4d} steps{'   <- last-rank slice empty if split' if empty else ''}")
