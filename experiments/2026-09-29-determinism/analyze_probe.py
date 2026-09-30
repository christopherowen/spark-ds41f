#!/usr/bin/env python3
"""Does serving's wq_b agree with its own recomputations? (runs from an attn-trace arm with
SPARK3_DEBUG_RECOMPUTE=1)

usage: analyze_probe.py OUT_DIR

For every run, rank and step, the first four layers' served query projection
(tag 24) is compared row by row with the same call repeated (25), the same
rows with a private scratch (26) and the rows shifted one position down in a
batch one row larger (27). A 24/25 mismatch means the served call is not
repeatable at a fixed batch (a race); 24/26 alone points at the shared
workspace; 24/27 alone means the result depends on the row's position or the
batch size. Prints, per comparison, the steps and rows that differ and the
batch sizes where they occur.
"""
import glob
import os
import sys
from collections import Counter, defaultdict

import torch

OUT = sys.argv[1]
LABELS = {25: "repeat", 26: "private scratch", 27: "shifted batch"}


def latest(node_dir, kind):
    files = sorted(glob.glob(os.path.join(node_dir, f"rank*-{kind}-*.pt")),
                   key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
    return torch.load(files[-1]) if files else None


def chronological(log):
    rows, count, capacity = log["rows"], log["count"], log["capacity"]
    if count > capacity:
        k = count % capacity
        rows = torch.cat([rows[k:], rows[:k]])
    return rows


totals = {tag: Counter() for tag in LABELS}
where = {tag: defaultdict(Counter) for tag in LABELS}
checked = Counter()
for run in sorted(glob.glob(os.path.join(OUT, "*-m*-r*"))):
    for node in ("dgx1", "dgx2", "dgx3"):
        log = latest(os.path.join(run, node), "attn")
        if log is None:
            continue
        names = log["names"]
        step = None
        current = {}
        for r in list(chronological(log)) + [None]:
            if r is None or r[0].item() == -1.0:
                for (slot, _), records in current.items():
                    served = records.get(24)
                    if served is None:
                        continue
                    for tag in LABELS:
                        other = records.get(tag)
                        if other is None:
                            continue
                        checked[tag] += 1
                        bad = (served != other).nonzero().flatten().tolist()
                        if bad:
                            totals[tag][node] += 1
                            where[tag][names[slot].split(".attn")[0].split("model.")[-1]][len(served)] += 1
                current = {}
                continue
            slot, n, tag = int(r[0].item()), int(r[1].item()), int(r[2].item())
            if tag in (24, *LABELS):
                current.setdefault((slot, 0), {})[tag] = r[3:3 + n].clone()
for tag, label in LABELS.items():
    n_bad = sum(totals[tag].values())
    print(f"served vs {label}: {n_bad} of {checked[tag]} layer-steps differ "
          f"(per rank {dict(totals[tag])})")
    for layer, sizes in sorted(where[tag].items()):
        print(f"    {layer}: batch rows where it differs {dict(sorted(sizes.items()))}")
