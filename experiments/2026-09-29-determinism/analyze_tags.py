#!/usr/bin/env python3
"""First shared-expert operation where two identical requests diverge (tag log).

usage: analyze_tags.py RANK_TAG_LOG.pt PROMPT_ROWS

Rows are (slot, rows, tag, sum, abs sum); tags 0 input, 1 gate_up, 2 act, 3 down.
Request starts are the last two rows with slot 0, tag 0 and PROMPT_ROWS rows.
"""
import sys

import torch

log = torch.load(sys.argv[1])
rows = log["rows"]
if log["count"] > rows.shape[0]:  # ring buffer wrapped: unroll to chronological order
    k = log["count"] % rows.shape[0]
    rows = torch.cat([rows[k:], rows[:k]])
prompt_rows = int(sys.argv[2])
starts = [i for i in range(rows.shape[0])
          if int(rows[i, 0]) == 0 and int(rows[i, 2]) == 0 and int(rows[i, 1]) == prompt_rows]
print(f"{rows.shape[0]} rows, starts {starts[-4:]}")
a, b = starts[-2], starts[-1]
length = b - a
tags = ("input", "gate_up", "act", "down")
for offset in range(min(length, rows.shape[0] - b)):
    ra, rb = rows[a + offset], rows[b + offset]
    if not torch.equal(ra, rb):
        print(f"first difference at tagged call {offset} (request length {length})")
        for o in range(max(0, offset - 4), min(offset + 8, length)):
            x, y = rows[a + o], rows[b + o]
            same = torch.equal(x, y)
            print(f"  {o:5d} slot {int(x[0]):3d} rows {int(x[1]):3d} {tags[int(x[2])]:8s} "
                  f"{'same' if same else 'DIFF'}  sum {x[3].item():.6e} vs {y[3].item():.6e}")
        break
else:
    print("identical over", length, "tagged calls")
