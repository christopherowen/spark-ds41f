#!/usr/bin/env python3
"""Row/column structure of the wrong tiles in the first down-projection run.

usage: analyze_capture_pattern.py NODE_DIR

For every latched layer and every 64-column tile where the two runs differ,
prints which rows differ and how many columns differ in each; for layers with
dumped weights, tests whether each wrong row equals the right row with one
128-wide K block of the input replaced (a stale-input signature): the
difference restricted to that tile is fitted by one K block's weights.
"""
import glob
import os
import sys

import torch

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
summary = {"rows_full": 0, "rows_partial": 0}
for name, e in data.items():
    if not e["latched"]:
        continue
    b = e["buffers"]
    first, second = b["first"].float(), b["second"].float()
    diff = (first != second).view(first.shape[0], -1, 64)
    tiles = diff.any(2).any(0).nonzero().flatten().tolist()
    parts = []
    for t in tiles:
        per_row = diff[:, t].sum(1).tolist()
        parts.append(f"t{t}:" + "/".join(str(int(c)) for c in per_row))
        for c in per_row:
            if c >= 60:
                summary["rows_full"] += 1
            elif c > 0:
                summary["rows_partial"] += 1
    short = name.split("layers.")[1].split(".")[0]
    print(f"layer {short:>2} ({e['mismatches']}/{e['calls']}): " + " ".join(parts[:10]))
    w = e["weights"]
    if not w:
        continue
    values = w["values"]
    values = values.view(torch.float8_e4m3fn) if values.dtype != torch.float8_e4m3fn else values
    vals = values.float().reshape(values.shape[0], -1)
    scales = w["scale_rows"].view(torch.uint8).reshape(-1, vals.shape[0], vals.shape[1] // 32)[0]
    deq = (vals.view(vals.shape[0], -1, 32) * torch.exp2(scales.float() - 127).unsqueeze(-1)).reshape(vals.shape).double()
    delta = (first - second).double()          # [rows, N]
    for r in range(first.shape[0]):
        cols = (first[r] != second[r]).nonzero().flatten()
        if cols.numel() == 0:
            continue
        bad_tiles = sorted({int(c) // 64 for c in cols})
        colset = torch.cat([torch.arange(t * 64, t * 64 + 64) for t in bad_tiles])
        y = delta[r, colset]
        fits = []
        for kb in range(vals.shape[1] // 128):
            wk = deq[colset, kb * 128:(kb + 1) * 128]          # [cols, 128]
            sol = torch.linalg.lstsq(wk, y.unsqueeze(1)).solution
            resid = (wk @ sol).squeeze(1) - y
            fits.append(float(resid.norm() / max(y.norm(), 1e-30)))
        print(f"   row {r}: tiles {bad_tiles}, one-K-block residual per block "
              f"{[round(f, 3) for f in fits]}")
print(summary)
