#!/usr/bin/env python3
"""The routed experts' chain in TP4 production, layer by layer, from rank 0's traces.

usage: routed_chain.py [TRACES] [--windows greedy sampled] [--json OUT]

Each target layer's routed chain is five launches: dispatch, FC1 (9 column blocks at
TP4), the fused SwiGLU and cast, FC2 (40 column blocks) and the combine. Per window:

- each launch's median duration and the idle gap before it;
- FC2 regressed on FC1 over every layer and step. FC1's bytes are twice FC2's for
  the same experts, so if FC1 = a1 + b1 x experts (the expert-rate bundle's fit at six
  rows: a1 = 14.2 us, b1 = 11.41 us an expert), FC2 = a + b x FC1 gives FC2's own fixed
  cost a + b x a1 and its time an expert b x b1, to set against the bench's FC2
  (14.5 us + 5.77 us an expert) and the attainable rate (1.53 MB an expert at
  266.6 GB/s: 5.74 us).

Steps and kernels come from the fusion survey's ceilings.py, as in step_shares.py.
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-10-25-fusion-survey"))
import ceilings  # noqa: E402

TRACES = os.path.expanduser("~/projects/spark-ds41f/results/private/determinism/request-inventory-5/ports")
FC1_FIXED, FC1_PER_EXPERT = 14.2, 11.41  # us, experiments/2026-10-26-c1-100tps (expert-rate bundle)
FC2_BYTES = 5120 * 576 // 2 + 5120 * 576 // 64 + 5120
RATE = 266.6e3  # bytes/us


def chains(rows):
    """(dispatch, FC1, SwiGLU, FC2, combine) rows of each layer, in order."""
    routed = [r for r in rows if r["name"].startswith(("moe_dispatch", "moe_grouped", "moe_swiglu", "moe_combine"))]
    out, i = [], 0
    while i + 4 < len(routed):
        a, b, c, d, e = routed[i:i + 5]
        if (a["name"].startswith("moe_dispatch") and b["name"].startswith("moe_grouped") and b["grid"][0] == 9
                and c["name"].startswith("moe_swiglu") and d["name"].startswith("moe_grouped")
                and d["grid"][0] == 40 and e["name"].startswith("moe_combine")):
            out.append((a, b, c, d, e))
            i += 5
        else:
            i += 1
    return out


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    b = sxy / sxx
    return my - b * mx, b, sxy * sxy / (sxx * syy)


def window(path):
    ks = ceilings.load(path)
    layers = [chain for target, _, _, _ in ceilings.step_pairs(ks)[1:-1] for chain in chains(target)]
    names = ("dispatch", "FC1", "SwiGLU and cast", "FC2", "combine")
    launches = {name: {"dur_us": round(statistics.median(layer[i]["dur"] for layer in layers), 2),
                       "gap_us": round(statistics.median(layer[i]["gap"] for layer in layers), 2),
                       "excl_us": round(statistics.median(layer[i]["excl"] for layer in layers), 2)}
                for i, name in enumerate(names)}
    a, b, r2 = fit([layer[1]["dur"] for layer in layers], [layer[3]["dur"] for layer in layers])
    fc1 = statistics.mean(layer[1]["dur"] for layer in layers)
    return {
        "layers": len(layers), "launches": launches,
        "fc2_on_fc1": {"intercept_us": round(a, 2), "slope": round(b, 4), "r2": round(r2, 3)},
        "fc2_fixed_us": round(a + b * FC1_FIXED, 2), "fc2_us_per_expert": round(b * FC1_PER_EXPERT, 3),
        "fc2_us_per_expert_at_rate": round(FC2_BYTES / RATE, 3),
        "experts_per_layer_from_fc1": round((fc1 - FC1_FIXED) / FC1_PER_EXPERT, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default=TRACES)
    ap.add_argument("--windows", nargs="*", default=["greedy", "sampled"])
    ap.add_argument("--json")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.traces, "decode-trace", "dgx1", "*.pt.trace.json.gz")))
    by_name = dict(zip(ceilings.DECODE, files))
    result = {name: window(by_name[name]) for name in args.windows}
    for name, r in result.items():
        print(f"\n## c1 {name}: {r['layers']} layers, {r['experts_per_layer_from_fc1']} experts a layer "
              f"(from FC1 at the bench's fit)\n")
        print("| launch | median us | gap before us | excl us |")
        print("|---|---:|---:|---:|")
        for launch, v in r["launches"].items():
            print(f"| {launch} | {v['dur_us']} | {v['gap_us']} | {v['excl_us']} |")
        f = r["fc2_on_fc1"]
        print(f"\nFC2 = {f['intercept_us']} + {f['slope']} x FC1 (r2 {f['r2']}): FC2's fixed cost "
              f"{r['fc2_fixed_us']} us, {r['fc2_us_per_expert']} us an expert "
              f"(attainable {r['fc2_us_per_expert_at_rate']})")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
