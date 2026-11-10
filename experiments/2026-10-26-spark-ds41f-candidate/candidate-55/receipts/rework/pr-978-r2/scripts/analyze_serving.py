#!/usr/bin/env python3
"""Summarize round 1's serving-path bench (serving_path_bench.py) for #978 round 2:
the candidate arm against candidate-54, per workload and per turn class (turn 1, turns 2+),
over the boots given. A cell is faster (slower) beyond the interval when every candidate boot's
median is below (above) every base boot's median. Also: payload bytes, stripped items, and
whether conversation 0's per-turn prompt-token and mm-hash digests equal the base's.
Usage: analyze_serving.py <out.json> --base <json>... --cand <json>...
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", nargs="+", required=True)
ap.add_argument("--cand", nargs="+", required=True)
args = ap.parse_args()
base = [json.loads(Path(p).read_text()) for p in args.base]
cand = [json.loads(Path(p).read_text()) for p in args.cand]


def verdict(b, c):
    if max(c) < min(b):
        return "faster beyond the interval"
    if min(c) > max(b):
        return "slower beyond the interval"
    return "within the interval"


out = {"base_files": args.base, "cand_files": args.cand,
       "mm_prefix_fastpath": {"base": [r.get("mm_prefix_fastpath") for r in base],
                              "cand": [r.get("mm_prefix_fastpath") for r in cand]},
       "workloads": {}}
for w in base[0]["workloads"]:
    row = {}
    for key, name in (("turn1_median_ms", "turn1"), ("turn2plus_median_ms", "turn2plus")):
        b = [r["workloads"][w][key] for r in base]
        c = [r["workloads"][w][key] for r in cand]
        row[name] = {"base_ms": b, "cand_ms": c, "ratio_cand_over_base": round(sum(c) / sum(b), 3),
                     "speedup": round(sum(b) / sum(c), 2), "verdict": verdict(b, c)}
    row["turn1_bytes"] = {"base": base[0]["workloads"][w]["turn1_bytes"],
                          "cand": cand[0]["workloads"][w]["turn1_bytes"]}
    row["turn2plus_bytes"] = {"base": base[0]["workloads"][w]["turn2plus_bytes"],
                              "cand": cand[0]["workloads"][w]["turn2plus_bytes"]}
    row["stripped_items"] = {"base": [r["workloads"][w]["stripped_items"] for r in base],
                             "cand": [r["workloads"][w]["stripped_items"] for r in cand]}
    ref = base[0]["workloads"][w]["identity_conv0"]
    row["prompt_token_ids_and_mm_hashes_equal_base"] = all(
        r["workloads"][w]["identity_conv0"] == ref for r in base + cand)
    out["workloads"][w] = row
Path(args.out).write_text(json.dumps(out, indent=1))
for w, v in out["workloads"].items():
    print(f"{w:22} t1 base {v['turn1']['base_ms']} cand {v['turn1']['cand_ms']} x{v['turn1']['ratio_cand_over_base']} "
          f"{v['turn1']['verdict']}; t2+ base {v['turn2plus']['base_ms']} cand {v['turn2plus']['cand_ms']} "
          f"{v['turn2plus']['speedup']}x {v['turn2plus']['verdict']}; stripped {v['stripped_items']['cand']}; "
          f"identity {v['prompt_token_ids_and_mm_hashes_equal_base']}")
