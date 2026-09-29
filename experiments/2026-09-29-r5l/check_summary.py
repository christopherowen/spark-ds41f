#!/usr/bin/env python3
"""Summarize SPARK3_DS41_INDEXER_SPLIT_CHECK logs; exit 1 if the split fails.

usage: check_summary.py [--exact] LOG...   (docker logs lines from candidate-check)

Per layer: forwards checked, rows compared, and positions that differ from the
first full-row run, for the split and for a second full-row run. The radix
top-k keeps an arbitrary subset of the positions tied at its threshold, so the
second run measures the spread the split may show. The split fails on any
unwritten row or changed candidate list, on any difference in a layer whose
full-row runs agree, or on more than twice the full-row spread. With --exact
(a top-k that breaks ties deterministically) any difference fails.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(
    r"indexer split check: layer (\d+) rank \d+ rows (\d+)-(\d+) of \d+: "
    r"split vs full (\d+) rows, (\d+) positions \(max (\d+)\); "
    r"full vs full (\d+) rows, (\d+) positions \(max (\d+)\); "
    r"(\d+) unwritten rows; (\d+) candidate rows differ"
)
EXACT = "--exact" in sys.argv[1:]
totals = defaultdict(lambda: defaultdict(int))
for path in [arg for arg in sys.argv[1:] if arg != "--exact"]:
    for line in open(path, errors="replace"):
        match = LINE.search(line)
        if not match:
            continue
        layer = int(match[1])
        t = totals[layer]
        t["forwards"] += 1
        t["rows"] += int(match[3]) - int(match[2])
        t["split_rows"] += int(match[4])
        t["split_positions"] += int(match[5])
        t["split_max"] = max(t["split_max"], int(match[6]))
        t["repeat_rows"] += int(match[7])
        t["repeat_positions"] += int(match[8])
        t["repeat_max"] = max(t["repeat_max"], int(match[9]))
        t["unwritten"] += int(match[10])
        t["candidates"] += int(match[11])
if not totals:
    sys.exit("no indexer split check lines found")
failed = False
print("| Layer | Forwards | Rows | Split rows / positions (max) | Full-vs-full rows / positions (max) | Unwritten | Candidates |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for layer in sorted(totals):
    t = totals[layer]
    bad = (
        t["unwritten"]
        or t["candidates"]
        or (t["repeat_positions"] == 0 and t["split_positions"] > 0)
        or t["split_positions"] > 2 * t["repeat_positions"]
        or (EXACT and (t["split_positions"] or t["repeat_positions"]))
    )
    failed |= bool(bad)
    print(
        f"| {layer} | {t['forwards']} | {t['rows']} | {t['split_rows']} / "
        f"{t['split_positions']} ({t['split_max']}) | {t['repeat_rows']} / "
        f"{t['repeat_positions']} ({t['repeat_max']}) | {t['unwritten']} | "
        f"{t['candidates']} |{' FAIL' if bad else ''}"
    )
print("split check", "FAILED" if failed else "passed")
sys.exit(1 if failed else 0)
