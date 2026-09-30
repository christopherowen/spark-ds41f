#!/usr/bin/env python3
"""Where does the immediate recompute of the shared down projection disagree?

usage: analyze_recompute.py NODE_DIR

Splits the latest tags dump into startup records and the two requests
(boundary: the earlier dump's count) and counts, per region and per row count,
records whose recompute (tag 7) differs from the first result (tag 3), with
the fraction of rows and the largest row-sum difference.
"""
import glob
import os
import sys
from collections import defaultdict

import torch

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-tags-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
first, last = torch.load(files[0]), torch.load(files[-1])
rows, c2, cap, names = last["rows"], last["count"], last["capacity"], last["names"]
if c2 > cap:
    k = c2 % cap
    rows = torch.cat([rows[k:], rows[:k]])
base = max(0, c2 - cap)
c1 = first["count"]
per_request = c2 - c1
regions = [("startup", base, c1 - per_request), ("request 1", c1 - per_request, c1), ("request 2", c1, c2)]
for label, lo, hi in regions:
    stats = defaultdict(lambda: [0, 0, 0, 0, 0.0])   # rows -> records, differing, rows total, rows differing, max diff
    state = {}
    for idx in range(max(lo, base), hi):
        r = rows[idx - base]
        slot, n, tag = int(r[0]), int(r[1]), int(r[2])
        if tag == 3:
            state[slot] = r[3:3 + n].clone()
        elif tag == 7 and slot in state:
            ref, val = state.pop(slot), r[3:3 + n]
            s = stats[n]
            s[0] += 1
            s[2] += n
            bad = ref != val
            if bool(bad.any()):
                s[1] += 1
                s[3] += int(bad.sum())
                s[4] = max(s[4], (ref - val).abs().max().item())
    line = ", ".join(f"rows {n}: {s[1]}/{s[0]} records ({s[3]}/{s[2]} rows, max {s[4]:.3g})"
                     for n, s in sorted(stats.items()))
    print(f"{label:9s} {line}")
