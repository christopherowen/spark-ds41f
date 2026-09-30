#!/usr/bin/env python3
"""Per wrong column: is first = second * constant (a wrong scale), or noise?

usage: analyze_capture_ratio.py NODE_DIR
"""
import glob
import os
import sys

import torch

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
shown = 0
fits = []
for name, e in data.items():
    if not e["latched"]:
        continue
    b = e["buffers"]
    first, second = b["first"].double(), b["second"].double()
    full = ((first != second).sum(0) >= first.shape[0] - 1).nonzero().flatten()  # columns wrong in (nearly) all rows
    for c in full.tolist():
        f, s = first[:, c], second[:, c]
        scale = float((f @ s) / (s @ s))
        resid = float((f - scale * s).norm() / (f - s).norm())
        fits.append((scale, resid))
        if shown < 12:
            shown += 1
            print(f"{name.split('layers.')[1].split('.')[0]} col {c}: best scale {scale:.4f}, residual after "
                  f"scaling {resid:.3f}; first {[round(x, 4) for x in f.tolist()]} second "
                  f"{[round(x, 4) for x in s.tolist()]}")
t = torch.tensor(fits)
print(f"full-row columns {len(fits)}: residual<0.1 {int((t[:, 1] < 0.1).sum())}; scale quantiles "
      f"{[round(float(x), 3) for x in t[:, 0].quantile(torch.tensor([0.1, 0.5, 0.9], dtype=t.dtype))]}")
