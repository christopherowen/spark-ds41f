#!/usr/bin/env python3
"""The time saved against the base build, per row count and PDL build, from a
pdl_chain_ab.py receipt.

usage: ab_table.py RECEIPT [warm_us|cold_us]

Each round runs every build once in a fresh process. The process sets the larger chains'
time to about +-1%, so a build is judged on base - build in each round: its median and
mean, the mean's standard error, and the rounds the build was slower in.
"""

import json
import math
import statistics
import sys


def main() -> None:
    kind = sys.argv[2] if len(sys.argv) > 2 else "warm_us"
    rows = [
        json.loads(line)
        for line in open(sys.argv[1])
        if line.startswith('{"shape"') and f'"{kind}": {{"base"' in line
    ]
    builds = [name for name in rows[0][kind] if name != "base"]
    print(
        f"{kind}: base - build a round; mean +- standard error (% of base); slower in"
    )
    print(f"{'rows':>5} {'base us':>9} " + " ".join(f"{b:>26}" for b in builds))
    for summary in rows:
        line = summary[kind]
        cells = []
        for b in builds:
            saved = line[b]["rounds_saved"]
            mean = statistics.mean(saved)
            se = statistics.stdev(saved) / math.sqrt(len(saved))
            slower = sum(s < 0 for s in saved)
            cells.append(f"{100 * mean / line['base']:+6.2f} +- {100 * se / line['base']:4.2f} "
                         f"({slower:>2}/{len(saved)})")  # fmt: skip
        print(
            f"{summary['rows']:>5} {line['base']:9.1f} "
            + " ".join(f"{c:>26}" for c in cells)
        )


if __name__ == "__main__":
    main()
