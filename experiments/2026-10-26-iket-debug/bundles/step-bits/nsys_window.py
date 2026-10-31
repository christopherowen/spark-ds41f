#!/usr/bin/env python3
"""The kernels nsys saw between two launches of a marker kernel, as a timeline.

usage: nsys_window.py FILE.sqlite [--between MARKER_KERNEL] [--gap-us US] [--top N]

FILE is an nsys SQLite export (``nsys export --type sqlite``; with CUDA graphs, traced
with ``--cuda-graph-trace=node`` so that each graph node is a kernel record). The
timeline is iket_table.py's: wall time, the GPU's busy time (the union of the kernels'
durations) and idle gaps, and the segments that gaps over ``--gap-us`` separate. Then
the kernels by total duration in the window, with how many ran inside a graph.
"""

import argparse
import sqlite3
from collections import defaultdict

from iket_table import timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--between", default="iket_step_marker_kernel")
    parser.add_argument("--gap-us", type=float, default=100.0)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    rows = sqlite3.connect(args.sqlite).execute(
        "SELECT k.start, k.end, s.value, k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
    ).fetchall()
    marks = [i for i, row in enumerate(rows) if row[2] == args.between]
    if len(marks) < 2:
        raise SystemExit(f"{args.between}: {len(marks)} launches, need two")
    window = rows[marks[0] + 1 : marks[-1]]
    lines, _ = timeline([(start, end, name) for start, end, name, _ in window], args.gap_us)
    total = defaultdict(float)
    count = defaultdict(int)
    graphed = defaultdict(int)
    for start, end, name, node in window:
        total[name] += (end - start) / 1000
        count[name] += 1
        graphed[name] += node is not None
    busy = sum(total.values())
    lines += [f"\n## Kernels by total duration ({len(window)} launches, "
              f"{sum(graphed.values())} in graphs)\n",
              "| kernel | launches | in graphs | total us | share | mean us |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in sorted(total, key=lambda n: -total[n])[: args.top]:
        lines.append(f"| {name} | {count[name]} | {graphed[name]} | {total[name]:.1f} | "
                     f"{total[name] / busy:.1%} | {total[name] / count[name]:.2f} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
