#!/usr/bin/env python3
"""Is a wrong column missing, doubling or borrowing one K sub-block product?

usage: analyze_capture_blocks.py NODE_DIR [BLOCK]   (BLOCK: 32 or 128, default 32)

For columns wrong in every row (layers with dumped weights), e = first - second
is compared with, per K sub-block q: -P[c, q] (missing), +P[c, q] (counted
twice), and +P[c', q'] for c' at the same in-tile column of any tile (an extra
product), where P[c, q] = A[:, q] @ W[c, q] over the rows. Reports residual
quality and the most common explanations.
"""
import glob
import os
import sys
from collections import Counter

import torch

node = sys.argv[1]
blk = int(sys.argv[2]) if len(sys.argv) > 2 else 32
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
best_kind, quality = Counter(), []
for name, e in data.items():
    if not (e["latched"] and e["weights"]):
        continue
    b, w = e["buffers"], e["weights"]
    first, second, act = b["first"].double(), b["second"].double(), b["act"].double()
    values = w["values"]
    values = values.view(torch.float8_e4m3fn) if values.dtype != torch.float8_e4m3fn else values
    vals = values.float().reshape(values.shape[0], -1)
    n, k = vals.shape
    scales = w["scale_rows"].view(torch.uint8).reshape(-1, n, k // 32)[0]
    W = (vals.view(n, -1, 32) * torch.exp2(scales.float() - 127).unsqueeze(-1)).reshape(n, k).double()
    q = k // blk
    P = torch.einsum("rqe,cqe->rcq", act.view(-1, q, blk), W.view(n, q, blk))   # [rows, n, q]
    cols = ((first != second).sum(0) >= first.shape[0] - 1).nonzero().flatten().tolist()
    for c in cols:
        err = first[:, c] - second[:, c]
        norm = max(err.norm().item(), 1e-12)
        cands = {}
        for qq in range(q):
            cands[("missing", qq)] = (err + P[:, c, qq]).norm().item() / norm
            cands[("twice", qq)] = (err - P[:, c, qq]).norm().item() / norm
        same = torch.arange(c % 64, n, 64)
        extra = (err.view(-1, 1, 1) - P[:, same, :]).norm(dim=0) / norm       # [tiles, q]
        val, idx = extra.flatten().min(0)
        u, qq = divmod(int(idx), q)
        cands[("extra", int(same[u]) // 64 - c // 64, qq)] = float(val)
        key = min(cands, key=cands.get)
        quality.append(cands[key])
        if cands[key] < 0.1:
            best_kind[key[0]] += 1
t = torch.tensor(quality)
print(f"block {blk}: columns {len(quality)}; residual <0.02 {int((t < 0.02).sum())}, <0.1 {int((t < 0.1).sum())}, "
      f"median {t.median():.3f}; explanations under 0.1: {dict(best_kind)}")
