#!/usr/bin/env python3
"""GPU energy per token for each arm's distinct-prompt stream samples.

usage: energy.py DIR ARM [ARM...]
       energy.py --bench DIR ARM [ARM...]   (decode phase of each timestamped bench-<arm>.txt)

DIR holds power-<node>.csv (epoch seconds, GPU watts, SM MHz, GPU C, hottest board zone in
milli-C, sampled every 0.5 s on each node) and c<N>-distinct-<arm>.jsonl from
scripts/distinct_streams.py, whose per-sample "wall" window aligns the two. Per sample: the
mean GPU power of each node inside the decode window, summed over the nodes; tokens per
joule = tokens per second / that power. GPU rail only, not the whole system.

With --bench, DIR holds bench-<arm>.txt, the bench output with each line prefixed by its
epoch time; the decode phase runs from the "decode:" line to the next section. Every arm runs
the same decode work, so its GPU energy compares arms directly.
"""
import csv
import glob
import json
import os
import re
import statistics
import sys


def power_logs(directory):
    logs = {}
    for path in glob.glob(os.path.join(directory, "power-*.csv")):
        rows = []
        for row in csv.reader(open(path)):
            try:
                rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]) / 1000))
            except (ValueError, IndexError):
                pass
        logs[os.path.basename(path)[6:-4]] = rows
    return logs


def window(logs, start, end):
    watts, clocks, gpu, board = 0.0, [], [], []
    for rows in logs.values():
        inside = [r for r in rows if start <= r[0] <= end]
        if not inside:
            return None
        watts += statistics.mean(r[1] for r in inside)
        clocks += [r[2] for r in inside]
        gpu += [r[3] for r in inside]
        board += [r[4] for r in inside]
    return watts, statistics.mean(clocks), max(gpu), max(board)


def bench_phases(directory, arms, logs):
    print(f"{'arm':9s} {'decode s':>8s} {'GPU W':>7s} {'GPU kJ':>7s} {'SM MHz':>7s} {'GPU max C':>9s} {'board max C':>11s}")
    for arm in arms:
        lines = [line.split(" ", 1) for line in open(os.path.join(directory, f"bench-{arm}.txt")) if line[:1].isdigit()]
        start = next(float(t) for t, text in lines if text.startswith("decode:"))
        end = next(float(t) for t, text in lines if float(t) > start and not text.startswith((" ", "decode:")))
        measured = window(logs, start, end)
        print(f"{arm:9s} {end - start:8.0f} {measured[0]:7.1f} {measured[0] * (end - start) / 1000:7.1f} "
              f"{measured[1]:7.0f} {measured[2]:9.0f} {measured[3]:11.1f}")


def main():
    if sys.argv[1] == "--bench":
        directory, arms = sys.argv[2], sys.argv[3:]
        bench_phases(directory, arms, power_logs(directory))
        return
    directory, arms = sys.argv[1], sys.argv[2:]
    logs = power_logs(directory)
    print(f"{'arm':9s} {'load':5s} {'tok/s':>7s} {'GPU W':>7s} {'tok/J':>6s} {'SM MHz':>7s} {'GPU max C':>9s} {'board max C':>11s}")
    for arm in arms:
        paths = sorted(glob.glob(os.path.join(directory, f"c*-distinct-{arm}.jsonl")),
                       key=lambda p: int(re.search(r"/c(\d+)-", p).group(1)))
        for path in paths:
            load = re.search(r"/(c\d+)-", path).group(1)
            rows = []
            for line in open(path):
                if line.startswith("{") and '"wall"' in line:
                    record = json.loads(line)
                    measured = window(logs, *record["wall"])
                    if measured:
                        rows.append((record["tps"], *measured))
            if rows:
                tps = statistics.mean(r[0] for r in rows)
                watts = statistics.mean(r[1] for r in rows)
                print(f"{arm:9s} {load:5s} {tps:7.1f} {watts:7.1f} {statistics.mean(r[0] / r[1] for r in rows):6.3f} "
                      f"{statistics.mean(r[2] for r in rows):7.0f} {max(r[3] for r in rows):9.0f} {max(r[4] for r in rows):11.1f}")


if __name__ == "__main__":
    main()
