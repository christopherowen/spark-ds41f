#!/usr/bin/env python3
"""Are the wrong values in the first run exact values from another tile?

usage: analyze_capture_source.py NODE_DIR

For each latched layer and each wrong 64-column tile T of the first run, looks
for a tile T' whose correct output (second run) equals the wrong values at the
same in-tile positions (epilogue-buffer reuse), and separately checks whether
the wrong values equal the correct output shifted by a whole number of
columns within the tile (warp-slice mixing). Reports matches and their offset.
"""
import glob
import os
import sys
from collections import Counter

import torch

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
offsets, unmatched, total = Counter(), 0, 0
examples = []
for name, e in data.items():
    if not e["latched"]:
        continue
    b = e["buffers"]
    first, second = b["first"], b["second"]
    rows, n = first.shape
    f3, s3 = first.view(rows, -1, 64), second.view(rows, -1, 64)
    wrong = f3 != s3
    for t in wrong.any(2).any(0).nonzero().flatten().tolist():
        mask = wrong[:, t]
        total += 1
        hits = [u for u in range(f3.shape[1]) if u != t and torch.equal(f3[:, t][mask], s3[:, u][mask])]
        if hits:
            off = hits[0] - t
            offsets[off] += 1
            if len(examples) < 12:
                examples.append(f"{name.split('layers.')[1].split('.')[0]} tile {t}: equals tile {hits} "
                                f"({int(mask.sum())} values)")
        else:
            unmatched += 1
            # zero? stale accumulator? report a sample
            if len(examples) < 12:
                vals = f3[:, t][mask][:4].float().tolist()
                ref = s3[:, t][mask][:4].float().tolist()
                examples.append(f"{name.split('layers.')[1].split('.')[0]} tile {t}: no source tile; "
                                f"wrong {[round(v, 4) for v in vals]} right {[round(v, 4) for v in ref]}")
print(f"wrong tiles {total}: matched another tile {total - unmatched}, unmatched {unmatched}")
print("tile offsets (T' - T):", dict(offsets.most_common(12)))
for x in examples:
    print("  " + x)
