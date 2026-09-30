#!/usr/bin/env python3
"""Exact bits of the first down-projection mismatch per layer (captures dump).

usage: analyze_capture.py NODE_DIR [LAYERS]

For each layer: mismatching calls / calls. For the first LAYERS (default 3)
that latched: where the two outputs differ (rows, 64-column tiles, element
counts), whether the scratch snapshots taken after each quantization agree
(and which byte ranges differ), and which output matches a float64 reference
computed from the captured input and the dequantized packed weights.
"""
import glob
import os
import sys

import torch

node = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3
files = sorted(glob.glob(os.path.join(node, "rank*-captures-*.pt")),
               key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
data = torch.load(files[-1])
latched = [(name, e) for name, e in data.items() if e["latched"]]
print(f"{os.path.basename(files[-1])}: {len(latched)}/{len(data)} layers latched; "
      f"mismatching calls {sum(e['mismatches'] for e in data.values())}/"
      f"{sum(e['calls'] for e in data.values())}")


def ranges(mask):
    idx = mask.nonzero().flatten().tolist()
    out, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append((start, prev + 1)); start = prev = i
    if start is not None:
        out.append((start, prev + 1))
    return out


for name, e in latched[:limit]:
    b, w = e["buffers"], e["weights"]
    first, second, act = b["first"].float(), b["second"].float(), b["act"]
    diff = first != second
    print(f"\n{name}: mismatches {e['mismatches']}/{e['calls']}, first at tags count {e['at']}")
    print(f"  output {tuple(first.shape)}: {int(diff.sum())} elements differ; rows {diff.any(1).nonzero().flatten().tolist()}")
    tiles = diff.view(diff.shape[0], -1, 64).any(2).any(0)
    print(f"  64-col tiles differing: {tiles.nonzero().flatten().tolist()} of {tiles.numel()}")
    per_tile = diff.view(diff.shape[0], -1, 64).sum((0, 2))
    bad = tiles.nonzero().flatten().tolist()
    print(f"  differing elements per bad tile: {[int(per_tile[t]) for t in bad][:24]}")
    fs, ss = b["first_scratch"].view(torch.uint8), b["second_scratch"].view(torch.uint8)
    sd = fs != ss
    print(f"  scratch {fs.numel()} bytes: {int(sd.sum())} differ; ranges {ranges(sd)[:12]}")
    if not w:
        print("  (weights not dumped for this layer)")
        continue
    values = w["values"]
    values = values.view(torch.float8_e4m3fn) if values.dtype != torch.float8_e4m3fn else values
    vals = values.float().reshape(values.shape[0], -1)            # [N, K]
    scales = w["scale_rows"].view(torch.uint8).reshape(-1, vals.shape[0], vals.shape[1] // 32)[0]
    deq = vals.view(vals.shape[0], -1, 32) * torch.exp2(scales.float() - 127).unsqueeze(-1)
    ref = act.double() @ deq.reshape(vals.shape).double().T
    for label, out in (("first", first), ("second", second)):
        err = (out.double() - ref).abs().view(out.shape[0], -1, 64).amax((0, 2))
        rel = (out.double() - ref).norm() / ref.norm()
        print(f"  {label}: relative error vs reference {rel:.3g}; worst tile error "
              f"{err.max():.3g} (tile {int(err.argmax())}); on differing tiles "
              f"{[round(float(err[t]), 3) for t in bad][:12]}")
