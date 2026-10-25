#!/usr/bin/env python3
"""The expert-rate bundle's output as a table, a line fit, and what it says about the traces.

usage: expert_rate_table.py OUTPUT_TXT [--floor floor.json]

OUTPUT_TXT is the bundle's output.txt (one JSON line a case). For each row count it fits
FC1's and FC2's time to a + b * experts (least squares over the cases), reports each
case's rate against the plain read and the traced LM head (floor.json), and inverts the
six-row FC1 fit at the traced greedy window's mean FC1 time a layer: the distinct experts
that time implies, and FC1's share of the attainable rate there.
"""
import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--floor", default=os.path.join(HERE, "floor.json"))
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(args.output) if line.startswith("{\"case\"")]
    floor = json.load(open(args.floor))
    greedy = floor["windows"]["greedy"]
    head = greedy["attainable_gbs (LM head)"]
    reads = [r for r in rows if r["case"] == "read"]
    print("| Read | us | GB/s | best GB/s |\n| --- | ---: | ---: | ---: |")
    for r in reads:
        print(f"| {r['mib']} MiB | {r['us']} | {r['gbs']} | {r['gbs_best']} |")
    print(f"\nTraced LM head: {head} GB/s\n")
    print("| Rows | Experts | FC1 us | FC1 GB/s | of head | FC2 us | FC2 GB/s | FC2 / FC1 rate |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    exps = [r for r in rows if r["case"] == "experts"]
    for r in exps:
        print(f"| {r['rows']} | {r['experts']} | {r['fc1_us']} | {r['fc1_gbs']} | {r['fc1_gbs'] / head:.0%} "
              f"| {r['fc2_us']} | {r['fc2_gbs']} | {r['fc2_rate_vs_fc1']} |")
    fits = {}
    for n_rows in sorted({r["rows"] for r in exps}):
        sel = [r for r in exps if r["rows"] == n_rows]
        xs = [r["experts"] for r in sel]
        fits[n_rows] = {k: fit(xs, [r[k + "_us"] for r in sel]) for k in ("fc1", "fc2")}
        for k, (a, b) in fits[n_rows].items():
            per_expert = floor[f"{k}_bytes_per_expert"]
            print(f"\n{n_rows} rows, {k.upper()}: {a:.1f} us + {b:.2f} us an expert "
                  f"(marginal rate {per_expert / b / 1e3:.1f} GB/s)", end="")
    print()
    if 6 in fits:
        a, b = fits[6]["fc1"]
        t = greedy["fc1_ms"] * 1e3 / 40
        n = (t - a) / b
        share = n * floor["fc1_bytes_per_expert"] / t / 1e3 / head
        print(f"\nTraced greedy window: FC1 {t:.1f} us a layer -> {n:.1f} distinct experts a layer "
              f"at the bundle's six-row fit; FC1 there at {share:.0%} of the traced head's rate")


if __name__ == "__main__":
    main()
