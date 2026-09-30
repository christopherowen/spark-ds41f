#!/usr/bin/env python3
"""Did the shared expert's down projection see changing weights, or change afterwards?

usage: analyze_down_state.py NODE_DIR

Reads one rank's latest tags dump (SPARK3_SHARED_RECOMPUTE=1). Per decode
record of the shared-expert MLP the tags are: 2 act, 3 down, 4-6 packed
weight bytes (values, MMA scales, tiled values), 7 an immediate recompute of
down, 8 act re-read, 9 down re-read, 10 the shared unit alpha. Reports, per
layer, whether the weights and alpha stayed constant over the whole log, and
how many records had a recompute (7), input (8) or output (9) that differs
from the first reading.
"""
import glob
import os
import sys
from collections import defaultdict

import torch

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-tags-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
last = torch.load(files[-1])
rows, count, cap, names = last["rows"], last["count"], last["capacity"], last["names"]
if count > cap:
    k = count % cap
    rows = torch.cat([rows[k:], rows[:k]])
print(f"{os.path.basename(files[-1])}: {rows.shape[0]} tag records")

constant = defaultdict(dict)   # (slot, tag) -> first values
changed = defaultdict(int)
seen = defaultdict(int)
mismatch = defaultdict(lambda: defaultdict(int))
records = defaultdict(int)
examples = []
current = {}                   # slot -> {tag: values} for the record in progress
for r in rows:
    slot, n, tag = int(r[0]), int(r[1]), int(r[2])
    values = r[3:3 + n].clone()
    if tag in (4, 5, 6, 10):
        key = (slot, tag)
        seen[key] += 1
        if key not in constant:
            constant[key] = values
        elif not torch.equal(constant[key], values):
            changed[key] += 1
        continue
    if tag == 0:
        current[slot] = {}
    state = current.setdefault(slot, {})
    state[tag] = values
    if tag == 9 and all(t in state for t in (2, 3, 7, 8)):
        records[slot] += 1
        for probe, ref in ((7, 3), (8, 2), (9, 3)):
            if not torch.equal(state[probe], state[ref]):
                mismatch[slot][probe] += 1
                if len(examples) < 8:
                    diff = (state[probe] - state[ref]).abs().max().item()
                    examples.append(f"{names[slot]} rows {n}: tag {probe} vs {ref} max row-sum diff {diff:.4g}")

labels = {4: "values", 5: "scale_mma", 6: "values_tiled", 10: "alpha"}
bad = False
for slot in sorted(records):
    weights = ", ".join(
        f"{labels[t]} {'changed ' + str(changed[(slot, t)]) + 'x' if changed[(slot, t)] else 'constant'}"
        for t in (4, 5, 6, 10) if (slot, t) in seen)
    m = mismatch[slot]
    line = (f"{names[slot]}: {records[slot]} records; recompute differs {m[7]}, "
            f"act changed {m[8]}, down changed {m[9]}; {weights}")
    if m or any(changed[(slot, t)] for t in (4, 5, 6, 10)):
        bad = True
        print("! " + line)
if not bad:
    print("every layer: recompute equal, act and down unchanged, weights and alpha constant")
    print(f"layers {len(records)}, records {sum(records.values())}")
for e in examples:
    print("  " + e)
