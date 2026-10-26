#!/usr/bin/env python3
"""Tables from the schedule benchmark's output (a bundle's output.txt).

usage: harness_table.py OUTPUT.txt [--against head] [--schedules pertype,typed,chained,off]

Per row count: each schedule's step time against the first, with its spread over
rounds; then each window consumer's calls a step, p50, excess a step and L2-speed share
(its floor over its time) under the reference schedule and each other one, layer types
summed.
"""

import argparse
import json
from collections import defaultdict


def lines(path, key):
    out = {}
    for line in open(path):
        if line.startswith("{" + json.dumps(key)):
            data = json.loads(line)
            out[data[key]] = data
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--against", default="head")
    ap.add_argument("--schedules", default="pertype,typed,chained,off")
    args = ap.parse_args()
    times = lines(args.output, "time")
    consumers = lines(args.output, "consumers")
    floors = lines(args.output, "floors")
    others = args.schedules.split(",")
    for head, t in times.items():
        ref = t[args.against]["step_us"]
        print(f"\n{head}: {args.against} {ref / 1000:.2f} ms")
        for s in others:
            if s not in t:
                continue
            rounds = t[s]["rounds_vs_first_percent"]
            print(f"  {s:8} {t[s]['step_us'] / 1000:8.2f} ms  {t[s]['vs_first_percent']:+6.2f}%"
                  f"  (rounds {min(rounds):+.2f} to {max(rounds):+.2f})"
                  f"  {(t[s]['step_us'] - ref):+8.1f} us")
        c = consumers.get(head)
        if not c:
            continue
        agg = defaultdict(lambda: defaultdict(float))
        for key, row in c.items():
            if key == "consumers":
                continue
            schedule, rest = key.split(" ", 1)
            role = rest.rsplit(" [", 1)[0]
            a = agg[(role, schedule)]
            a["calls"] += row["per_step"]
            a["time"] += row["p50_us"] * row["per_step"]
            a["excess"] += row["excess_us_per_step"] or 0.0
            a["floor"] += (row["floor_us"] or 0.0) * row["per_step"]
        roles = sorted({role for role, _ in agg})
        print(f"  {'consumer':16} " + "  ".join(f"{s:>22}" for s in [args.against, *others]))
        print(f"  {'':16} " + "  ".join(f"{'excess us':>11}{'L2 share':>11}" for _ in [args.against, *others]))
        for role in roles:
            cells = []
            for s in [args.against, *others]:
                a = agg.get((role, s))
                if not a or not a["time"]:
                    cells.append(f"{'-':>11}{'-':>11}")
                    continue
                cells.append(f"{a['excess']:11.1f}{a['floor'] / a['time']:11.2f}")
            print(f"  {role:16} " + "  ".join(cells))
        print(f"  floors: {json.dumps({k: v for k, v in floors.get(head, {}).items() if k != 'floors'})}")


if __name__ == "__main__":
    main()
