#!/usr/bin/env python3
"""Per-row comparison of two identical requests in the checksum logs.

usage: analyze_rows.py RUNNER_LOG.pt TAG_LOG.pt

Both logs were reset just before the requests, so request 1 starts at record
0 and request 2 at the next slot-0 record whose input rows equal record 0's.
For each record pair it counts rows whose input matches and, among those, rows
whose outputs match; the first record where a matching input row produces a
different output names the operation that diverged.
"""
import sys

import torch

M = 64


def load(path):
    log = torch.load(path)
    rows, count, cap = log["rows"], log["count"], log["capacity"]
    if count > cap:  # ring wrapped: unroll to chronological order
        k = count % cap
        rows = torch.cat([rows[k:], rows[:k]])
    return rows


def split(rows, slot_col, first_value_col, tag_col=None):
    """Return (start1, start2): the last two slot-0 records with equal input rows.

    Both requests start with an identical prefill, so request starts are the
    latest pair of slot-0 (tag-0) records whose per-row inputs match exactly.
    """
    # Requests' prefills are short (at most 32 rows); startup warmups are longer.
    # Compare the first four rows only: padding rows at the end carry stale values.
    cand = [i for i in range(rows.shape[0]) if int(rows[i, 0]) == 0
            and (tag_col is None or int(rows[i, tag_col]) == 0) and 6 < int(rows[i, 1]) <= 32]
    for j in reversed(cand):
        n = int(rows[j, 1])
        for i in reversed([c for c in cand if c < j]):
            if int(rows[i, 1]) == n and torch.equal(
                rows[i, first_value_col : first_value_col + 4],
                rows[j, first_value_col : first_value_col + 4],
            ):
                return i, j
    raise SystemExit("two identical prefills not found")


runner = load(sys.argv[1])
r0, s = split(runner, 0, 2)
runner = runner[r0:]
s -= r0
print(f"runner: request 1 at {r0}, request 2 at +{s}")
names = ("shared", "routed")
reported = False
for i in range(min(s, runner.shape[0] - s)):
    a, b = runner[i], runner[s + i]
    if int(a[0]) != int(b[0]) or int(a[1]) != int(b[1]):
        print(f"record {i}: shape/slot differs ({int(a[0])},{int(a[1])}) vs ({int(b[0])},{int(b[1])})")
        break
    n = int(a[1])
    x_eq = a[2 : 2 + n] == b[2 : 2 + n]
    outs = {nm: (a[2 + (k + 1) * M : 2 + (k + 1) * M + n] == b[2 + (k + 1) * M : 2 + (k + 1) * M + n])
            for k, nm in enumerate(names)}
    bad = {nm: int((x_eq & ~eq).sum()) for nm, eq in outs.items()}
    if any(bad.values()) and not reported:
        print(f"record {i} slot {int(a[0])} rows {n}: input-equal rows {int(x_eq.sum())}/{n}, "
              f"of those differing: " + ", ".join(f"{k} {v}" for k, v in bad.items()))
        rows_bad = [(j, nm) for nm, eq in outs.items() for j in range(n) if x_eq[j] and not eq[j]]
        print("   rows:", rows_bad[:12])
        reported = True
        break
if not reported:
    print("no input-equal row with a different output in the runner log")

tags = load(sys.argv[2])
t0, st = split(tags, 0, 3, tag_col=2)
tags = tags[t0:]
st -= t0
print(f"tags: request 1 at {t0}, request 2 at +{st}")
labels = ("input", "gate_up", "act", "down")
prev_input_eq = None
for i in range(min(st, tags.shape[0] - st)):
    a, b = tags[i], tags[st + i]
    if int(a[0]) != int(b[0]) or int(a[1]) != int(b[1]) or int(a[2]) != int(b[2]):
        print(f"tag record {i}: layout differs"); break
    n, tag = int(a[1]), int(a[2])
    eq = a[3 : 3 + n] == b[3 : 3 + n]
    if tag == 0:
        prev_input_eq = eq
        continue
    bad = prev_input_eq & ~eq
    if bool(bad.any()):
        print(f"tag record {i} slot {int(a[0])} rows {n}: {labels[tag]} differs on "
              f"{int(bad.sum())} rows whose shared-expert input was equal: {bad.nonzero().flatten().tolist()[:12]}")
        for o in range(max(0, i - 3), i + 1):
            x, y = tags[o], tags[st + o]
            m = int(x[1])
            print(f"   {o} slot {int(x[0])} {labels[int(x[2])]:8s} equal rows {int((x[3:3+m]==y[3:3+m]).sum())}/{m}")
        break
else:
    print("no shared-expert operation diverged on equal input")
