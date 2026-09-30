#!/usr/bin/env python3
"""Find the first MoE call where two identical requests diverge.

usage: analyze_checksums.py RANK_LOG.pt PROMPT_ROWS

The log (checksum_debug.py) holds startup rows, then request 1, then request 2.
Request starts are the last two rows whose layer slot is 0 and whose row count
is PROMPT_ROWS (the prefill's first MoE call; SP splits rows across ranks, so
pass this rank's prefill row count, or 0 to take slot-0 rows with the most rows).
Rows are (slot, rows, sum input, sum shared, sum routed).
"""
import sys

import torch

log = torch.load(sys.argv[1])
rows = log["rows"]
if log["count"] > rows.shape[0]:  # ring buffer wrapped: unroll to chronological order
    k = log["count"] % rows.shape[0]
    rows = torch.cat([rows[k:], rows[:k]])
prompt_rows = int(sys.argv[2])
print(f"{log['count']} rows logged")
slot0 = (rows[:, 0] == 0).nonzero().flatten().tolist()
if prompt_rows == 0:
    prompt_rows = int(max(rows[i, 1] for i in slot0))
starts = [i for i in slot0 if int(rows[i, 1]) == prompt_rows]
print(f"prefill rows {prompt_rows}: {len(starts)} candidate starts, last {starts[-4:]}")
a, b = starts[-2], starts[-1]
length = b - a
first = None
for offset in range(min(length, rows.shape[0] - b)):
    ra, rb = rows[a + offset], rows[b + offset]
    if not torch.equal(ra, rb):
        first = offset
        break
if first is None:
    print("requests identical over", length, "rows")
    sys.exit(0)
names = ("slot", "rows", "input", "shared", "routed")
print(f"first difference at call {first} of request (request length {length})")
for offset in range(max(0, first - 3), min(first + 6, length)):
    ra, rb = rows[a + offset], rows[b + offset]
    diff = [names[k] for k in range(5) if ra[k].item() != rb[k].item()]
    print(f"  call {offset:5d} slot {int(ra[0]):3d}/{int(rb[0]):3d} rows {int(ra[1]):4d}/{int(rb[1]):4d} "
          f"differs: {','.join(diff) or '-'}"
          + (f"  input {ra[2].item():.6e} vs {rb[2].item():.6e}" if "input" in diff else "")
          + (f"  shared {ra[3].item():.6e} vs {rb[3].item():.6e}" if "shared" in diff else "")
          + (f"  routed {ra[4].item():.6e} vs {rb[4].item():.6e}" if "routed" in diff else ""))
