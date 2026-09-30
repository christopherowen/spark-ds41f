#!/usr/bin/env python3
"""Where do two identical requests first diverge? (per-row checksum logs)

usage: analyze_requests.py NODE_DIR

NODE_DIR holds one rank's dumps taken after request 1 and after request 2:
rank<r>-runner-<c1>.pt, rank<r>-runner-<c2>.pt, and the same for tags.
Request 2 is records [c1, c2); request 1 ends at c1 and starts at the latest
earlier record matching request 2's first record on its first four rows.
For each aligned record it compares rows whose input matches, and reports the
first record where such a row has a different output, with the layer name.
"""
import glob
import os
import sys

import torch

M = 64
node = sys.argv[1]


def dumps(kind):
    files = sorted(glob.glob(os.path.join(node, f"rank*-{kind}-*.pt")),
                   key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
    first, last = torch.load(files[0]), torch.load(files[-1])
    rows, c2, cap = last["rows"], last["count"], last["capacity"]
    if c2 > cap:
        k = c2 % cap
        rows = torch.cat([rows[k:], rows[:k]])
    base = max(0, c2 - cap)  # count value of rows[0]
    return rows, base, first["count"], c2, last["names"]


def align(rows, base, c1, c2, value_col, tag_col=None):
    s2 = c1 - base
    ref = rows[s2]
    n = min(4, int(ref[1]))
    for i in range(s2 - 1, -1, -1):
        r = rows[i]
        if int(r[0]) == int(ref[0]) and int(r[1]) == int(ref[1]) and (
            tag_col is None or int(r[tag_col]) == int(ref[tag_col])
        ) and torch.equal(r[value_col : value_col + n], ref[value_col : value_col + n]):
            return i, s2, c2 - base
    raise SystemExit("request 1 start not found")


rows, base, c1, c2, names = dumps("runner")
s1, s2, end = align(rows, base, c1, c2, 2)
print(f"runner: request 1 records {s2 - s1}, request 2 records {end - s2}")
for i in range(min(s2 - s1, end - s2)):
    a, b = rows[s1 + i], rows[s2 + i]
    if int(a[0]) != int(b[0]) or int(a[1]) != int(b[1]):
        print(f"record {i}: layer/rows differ: {names[int(a[0])]} {int(a[1])} vs {names[int(b[0])]} {int(b[1])}")
        break
    n = int(a[1])
    x_eq = a[2 : 2 + n] == b[2 : 2 + n]
    sh_eq = a[2 + M : 2 + M + n] == b[2 + M : 2 + M + n]
    ro_eq = a[2 + 2 * M : 2 + 2 * M + n] == b[2 + 2 * M : 2 + 2 * M + n]
    bad_sh, bad_ro = x_eq & ~sh_eq, x_eq & ~ro_eq
    if bool(bad_sh.any() or bad_ro.any()):
        print(f"record {i} {names[int(a[0])]} rows {n}: input-equal {int(x_eq.sum())}/{n}; "
              f"shared differs on {bad_sh.nonzero().flatten().tolist()}, "
              f"routed differs on {bad_ro.nonzero().flatten().tolist()}")
        break
    if not bool(x_eq[0]):
        print(f"record {i} {names[int(a[0])]} rows {n}: first input row already differs "
              f"(upstream of this MoE); input-equal {int(x_eq.sum())}/{n}")
        break
else:
    print("runner: no divergence")

rows, base, c1, c2, names = dumps("tags")
s1, s2, end = align(rows, base, c1, c2, 3, tag_col=2)
labels = ("input", "gate_up", "act", "down")
print(f"tags: request 1 records {s2 - s1}, request 2 records {end - s2}")
input_eq = None
for i in range(min(s2 - s1, end - s2)):
    a, b = rows[s1 + i], rows[s2 + i]
    n, tag = int(a[1]), int(a[2])
    eq = a[3 : 3 + n] == b[3 : 3 + n]
    if tag == 0:
        input_eq = eq
        if not bool(eq[0]):
            print(f"tag record {i} {names[int(a[0])]}: shared-expert input row 0 differs (upstream)")
            break
        continue
    bad = input_eq & ~eq
    if bool(bad.any()):
        print(f"tag record {i} {names[int(a[0])]} rows {n}: {labels[tag]} differs on rows "
              f"{bad.nonzero().flatten().tolist()} whose input was equal")
        for o in range(max(0, i - 3), i + 1):
            x, y = rows[s1 + o], rows[s2 + o]
            m = int(x[1])
            print(f"   {labels[int(x[2])]:8s} equal rows {int((x[3:3 + m] == y[3:3 + m]).sum())}/{m}")
        break
else:
    print("tags: no divergence")
