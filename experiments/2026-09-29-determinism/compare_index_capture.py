#!/usr/bin/env python3
"""Compare captured index selections (SPARK3_DEBUG_INDEX_CAPTURE) between two runs, row by row.

usage: compare_index_capture.py RUN_A_LOG_DIR RUN_B_LOG_DIR   (logs-* directories with one subdirectory per node)

Each rank captures the rows it scored ("select": selection, top-k scores, the
quantized index query and scales, index weights) and every rank captures the
gathered selection ("indices"). Rows are matched by (layer, position) across
all nodes of each run. For each: which positions only one run selected, with
their scores; whether the index query bytes, scales and weights are equal; and
the largest score difference over positions both runs selected.
"""
import glob
import os
import sys

import torch


def captures(run_dir):
    select, gathered = {}, {}
    for node_dir in sorted(glob.glob(os.path.join(run_dir, "dgx*"))):
        files = sorted(glob.glob(os.path.join(node_dir, "rank*-index-capture-*.pt")),
                       key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
        for entry in (torch.load(files[-1]) if files else []):
            for i, p in enumerate(entry["positions"].tolist()):
                key = (entry["layer"], p)
                if entry.get("kind") == "select":
                    select.setdefault(key, {k: entry[k][i] for k in
                                            ("indices", "scores", "q_data", "q_scales", "weights", "lengths")}
                                      | {"node": os.path.basename(node_dir), "step": entry["step"]})
                else:
                    gathered.setdefault(key, entry["indices"][i])
    return select, gathered


(sa, ga), (sb, gb) = captures(sys.argv[1]), captures(sys.argv[2])
print(f"select rows {len(sa)}/{len(sb)}, gathered rows {len(ga)}/{len(gb)}")
for key in sorted(set(sa) & set(sb))[:30]:
    a, b = sa[key], sb[key]
    n = int(a["lengths"])
    ia, ib = a["indices"][:512].tolist(), b["indices"][:512].tolist()
    score_a = {int(i): float(v) for i, v in zip(ia, a["scores"][:512].tolist()) if i >= 0}
    score_b = {int(i): float(v) for i, v in zip(ib, b["scores"][:512].tolist()) if i >= 0}
    only_a = sorted(set(score_a) - set(score_b))
    only_b = sorted(set(score_b) - set(score_a))
    common = set(score_a) & set(score_b)
    max_diff = max((abs(score_a[i] - score_b[i]) for i in common), default=0.0)
    threshold_a, threshold_b = min(score_a.values()), min(score_b.values())
    print(f"layer {key[0]} position {key[1]} ({a['node']} step {a['step']} vs {b['node']} step {b['step']}): "
          f"q equal {torch.equal(a['q_data'], b['q_data'])}, scales equal {torch.equal(a['q_scales'], b['q_scales'])}, "
          f"weights equal {torch.equal(a['weights'], b['weights'])}, length {n}/{int(b['lengths'])}, "
          f"max score diff on common {max_diff:.3g}, thresholds {threshold_a:.6g}/{threshold_b:.6g}, "
          f"only A {[(i, score_a[i]) for i in only_a[:4]]}, only B {[(i, score_b[i]) for i in only_b[:4]]}")
