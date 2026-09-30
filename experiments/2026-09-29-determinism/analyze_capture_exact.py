#!/usr/bin/env python3
"""Exact-operand explanation of the wrong tiles (captures with weights; CPU).

usage: analyze_capture_exact.py NODE_DIR   (cwd: the B12X checkout)

Decodes the exact MXFP8 input from the captured scratch and the packed down
weights, checks that it reproduces the correct (second) output, then for each
wrong 64-column tile scores, over all its wrong columns together, hypotheses
where one 32-wide K sub-block q was computed with: the A fragment of sub-block
q' (A replaced), the B fragment of sub-block q' of the column 64*d further on
(B replaced), or both; plus q missing or counted twice.
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from b12x.gemm._shared.block_fp8 import _block_fp8_linear_x_q_from_scratch  # noqa: E402

node = sys.argv[1]
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
for name, e in data.items():
    if not (e["latched"] and e["weights"]):
        continue
    b, w = e["buffers"], e["weights"]
    first, second = b["first"].double(), b["second"].double()
    rows = first.shape[0]
    values = w["values"]
    values = values.view(torch.float8_e4m3fn) if values.dtype != torch.float8_e4m3fn else values
    vals = values.float().reshape(values.shape[0], -1)
    n, k = vals.shape
    wsc = w["scale_rows"].view(torch.uint8).reshape(-1, n, k // 32)[0]
    W = (vals.view(n, -1, 32) * torch.exp2(wsc.float() - 127).unsqueeze(-1)).reshape(n, k).double()
    xq = _block_fp8_linear_x_q_from_scratch(b["first_scratch"].view(torch.uint8).contiguous(), tokens=rows,
                                            in_features=k, output_dtype=torch.bfloat16)
    asc = xq.scale_rows.view(torch.uint8)[0].float()                     # [rows, k/32]
    A = (xq.values.float().view(rows, -1, 32) * torch.exp2(asc - 127).unsqueeze(-1)).reshape(rows, k).double()
    ref = (A @ W.T)
    rel2 = ((ref.to(torch.bfloat16).double() - second).norm() / second.norm()).item()
    rel1 = ((ref.to(torch.bfloat16).double() - first).norm() / first.norm()).item()
    layer = name.split("layers.")[1].split(".")[0]
    print(f"layer {layer}: exact reference vs second {rel2:.2e}, vs first {rel1:.2e}")
    q = k // 32
    Aq, Wq = A.view(rows, q, 32), W.view(n, q, 32)
    Q = torch.einsum("rae,cbe->rcab", Aq, Wq)                             # [rows, n, qa, qw]
    diag = torch.diagonal(Q, dim1=2, dim2=3)                               # [rows, n, q] true partials
    diff = first - second
    wrong = (first != second).view(rows, -1, 64)
    for t in wrong.any(2).any(0).nonzero().flatten().tolist():
        cols = (t * 64 + wrong[:, t].any(0).nonzero().flatten())
        E = diff[:, cols]                                                  # [rows, C]
        norm = E.norm().item()
        best = []
        for qq in range(q):
            base = diag[:, cols, qq]                                       # [rows, C]
            best.append(((E + base).norm().item() / norm, f"q{qq} missing"))
            best.append(((E - base).norm().item() / norm, f"q{qq} twice"))
            a_rep = Q[:, cols, :, qq] - base.unsqueeze(-1)                  # A from q' : [rows, C, q']
            r = (a_rep - E.unsqueeze(-1)).norm(dim=(0, 1)) / norm
            v, i = r.min(0)
            best.append((v.item(), f"q{qq} A from q{int(i)}"))
            for d in range(-(t), 80 - t):
                cc = cols + 64 * d
                b_rep = Q[:, cc, qq, :] - base.unsqueeze(-1)               # B from (c+64d, q')
                r = (b_rep - E.unsqueeze(-1)).norm(dim=(0, 1)) / norm
                v, i = r.min(0)
                best.append((v.item(), f"q{qq} B from tile{d:+d} q{int(i)}"))
                both = torch.diagonal(Q[:, cc], dim1=2, dim2=3) - base.unsqueeze(-1)
                r = (both - E.unsqueeze(-1)).norm(dim=(0, 1)) / norm
                v, i = r.min(0)
                best.append((v.item(), f"q{qq} A+B from tile{d:+d} q{int(i)}"))
        best.sort()
        print(f"  tile {t} ({len(cols)} cols): " + "; ".join(f"{s} {v:.3f}" for v, s in best[:3]))
