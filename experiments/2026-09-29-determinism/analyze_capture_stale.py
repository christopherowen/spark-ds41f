#!/usr/bin/env python3
"""Which stale operand explains a wrong column? (captures with weights)

usage: analyze_capture_stale.py NODE_DIR

For each wrong output column c of the first run (layers with dumped weights),
the error e = first - second over the rows is compared with candidate stale
B-operand stages: K block kb (128 wide) of column c computed with the weights
of column c' at K block kb' instead (c' in the same in-tile position of any
64-column tile, kb' any block), i.e. e ~ A[:, kb] @ (W[c', kb'] - W[c, kb]).
A is the captured BF16 input (activation quantization is ignored, so matches
are near, not exact). Reports the best candidate per column and a histogram of
(tile offset, kb, kb').
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
hist, quality, shown = Counter(), [], 0
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
    kbs = k // 128
    # partial[r, c, kb] = A[r, kb block] @ W[c, kb block]
    partial = torch.einsum("rqe,cqe->rcq", act.view(-1, kbs, 128), W.view(n, kbs, 128))
    wrong_cols = (first != second).any(0).nonzero().flatten().tolist()
    layer = name.split("layers.")[1].split(".")[0]
    for c in wrong_cols:
        err = first[:, c] - second[:, c]
        t, j = divmod(c, 64)
        cands = torch.arange(j, n, 64)                           # same in-tile column, every tile
        best = None
        for kb in range(kbs):
            # candidate prediction for every (c', kb'): partial[:, c', kb'] - partial[:, c, kb]
            pred = partial[:, cands, :] - partial[:, c, kb].view(-1, 1, 1)   # [r, tiles, kb']
            res = (pred - err.view(-1, 1, 1)).norm(dim=0) / max(err.norm().item(), 1e-12)
            val, idx = res.flatten().min(0)
            if best is None or val < best[0]:
                u, kb2 = divmod(int(idx), kbs)
                best = (float(val), int(cands[u]) // 64 - t, kb, kb2)
        quality.append(best[0])
        if best[0] < 0.2:
            hist[(best[1], best[2], best[3])] += 1
        if shown < 14:
            shown += 1
            print(f"layer {layer} col {c} (tile {t}): best residual {best[0]:.3f} with tile offset "
                  f"{best[1]:+d}, K block {best[2]} computed with block {best[3]}")
q = torch.tensor(quality)
print(f"columns {len(quality)}: residual <0.05 {int((q < 0.05).sum())}, <0.2 {int((q < 0.2).sum())}, "
      f"median {q.median():.3f}")
print("(tile offset, kb, kb') for residual < 0.2:", hist.most_common(12))
