#!/usr/bin/env python3
"""Compare captured index selections (SPARK3_DEBUG_INDEX_CAPTURE) between two runs, row by row.

usage: compare_index_capture.py RUN_A_NODE_DIR RUN_B_NODE_DIR   (logs-*/<node> directories)

Rows are matched by (layer, position). For each: the selected lengths, how many
positions only one run selected, and the first few of them; plus each run's
index score width per layer and step (tag 23, raw in attn-exact4).
"""
import glob
import os
import sys

import torch


def captures(node_dir):
    files = sorted(glob.glob(os.path.join(node_dir, "rank*-index-capture-*.pt")),
                   key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
    rows = {}
    for entry in (torch.load(files[-1]) if files else []):
        for p, idx, n in zip(entry["positions"].tolist(), entry["indices"], entry["lengths"].tolist()):
            rows.setdefault((entry["layer"], p), (entry["step"], set(int(v) for v in idx[:n].tolist() if v >= 0), n))
    return rows


a, b = captures(sys.argv[1]), captures(sys.argv[2])
print(f"captured rows: {len(a)} and {len(b)}; common {len(set(a) & set(b))}")
for key in sorted(set(a) & set(b))[:40]:
    (sa, xa, na), (sb, xb, nb) = a[key], b[key]
    only_a, only_b = sorted(xa - xb), sorted(xb - xa)
    print(f"layer {key[0]} position {key[1]}: lengths {na}/{nb}, only A {len(only_a)} {only_a[:8]}, "
          f"only B {len(only_b)} {only_b[:8]}")
