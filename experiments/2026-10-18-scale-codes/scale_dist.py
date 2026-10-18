"""Sampled distribution of DS4.1 Flash routed-expert UE8M0 block scales (read-only).

Per output row (one scale per 32 FP4 values along K), how wide is the exponent range, and how
many values fall outside a 2^b-wide window? Reads only the scale tensors of the sampled layers.
"""
import collections
import glob
import json
import math
import os
import re
import struct
import sys

import numpy as np

snap, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
want = re.compile(r"^(layers\.(\d+)|mtp\.0)\.ffn\.experts\.\d+\.(w[123])\.scale$")

index = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        m = want.match(k)
        if not m:
            continue
        layer = "mtp" if m.group(1) == "mtp.0" else int(m.group(2))
        if layer != "mtp" and layer not in layers:
            continue
        index.setdefault(f, []).append((k, layer, m.group(3), 8 + n + v["data_offsets"][0], v["shape"]))

hist = np.zeros(256, np.int64)
span_hist = collections.Counter()
rows = 0
values = 0
outside = {b: 0 for b in (2, 3, 4)}       # per-row window anchored at the row's max
outside_min = {b: 0 for b in (2, 3, 4)}   # anchored at the row's min
delta_hist = np.zeros(256, np.int64)      # row max minus value
per_matrix = collections.defaultdict(lambda: np.zeros(256, np.int64))
for f, items in index.items():
    fd = os.open(f, os.O_RDONLY)
    try:
        for k, layer, w, off, shape in sorted(items, key=lambda x: x[3]):
            count = shape[0] * shape[1]
            a = np.frombuffer(os.pread(fd, count, off), np.uint8).reshape(shape).astype(np.int16)
            hist += np.bincount(a.ravel(), minlength=256)
            per_matrix[w] += np.bincount(a.ravel(), minlength=256)
            mx, mn = a.max(axis=1), a.min(axis=1)
            span = mx - mn + 1
            span_hist.update(span.tolist())
            d = mx[:, None] - a
            delta_hist += np.bincount(d.ravel(), minlength=256)
            for b in outside:
                outside[b] += int((d >= (1 << b)).sum())
                outside_min[b] += int(((a - mn[:, None]) >= (1 << b)).sum())
            rows += shape[0]
            values += count
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)

print(f"sampled layers {layers} + mtp.0: {sum(len(v) for v in index.values())} tensors, "
      f"{rows:,} rows, {values:,} scales ({values / 2**30:.2f} GiB)")
nz = np.nonzero(hist)[0]
print("exponent values present:", nz.min(), "..", nz.max(), f"({len(nz)} distinct)")
top = np.argsort(hist)[::-1][:12]
print("most common (value: share):", ", ".join(f"{v}: {hist[v] / values:.3%}" for v in top))
print("special: e=0", hist[0], " e=255", hist[255])
for w, h in sorted(per_matrix.items()):
    z = np.nonzero(h)[0]
    print(f"  {w}: values {z.min()}..{z.max()}")
print("row span (max-min+1) distribution:")
total = sum(span_hist.values())
for lim in (1, 2, 4, 8, 16, 32):
    c = sum(n for s, n in span_hist.items() if s <= lim)
    print(f"  rows with span <= {lim:2d}: {c / total:.4%}")
print("values outside a per-row window (anchored at row max / at row min):")
for b in outside:
    print(f"  {b}-bit window ({1 << b:2d} exponents): {outside[b] / values:.4%} / {outside_min[b] / values:.4%}")
p = delta_hist / delta_hist.sum()
ent = -sum(x * math.log2(x) for x in p if x > 0)
print(f"entropy of (row max - value): {ent:.2f} bits")
print("delta distribution:", ", ".join(f"{d}: {p[d]:.2%}" for d in range(0, 12)))
