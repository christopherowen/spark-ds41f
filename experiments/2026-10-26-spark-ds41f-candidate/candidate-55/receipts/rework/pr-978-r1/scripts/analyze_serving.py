#!/usr/bin/env python3
"""Summarize serving-path-bench (serving-<arm>-boot<k>.json): r1 against base per workload."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = {a: [json.loads((d / f"serving-{a}-boot{k}.json").read_text()) for k in (1, 2)] for a in ("base", "r1")}
out = {"workloads": {}}
for w in arms["base"][0]["workloads"]:
    row = {a: {k: [b["workloads"][w][k] for b in arms[a]]
               for k in ("turn1_median_ms", "turn2plus_median_ms", "turn2plus_bytes", "stripped_items")}
           for a in arms}
    for k in ("turn1_median_ms", "turn2plus_median_ms"):
        b, r = row["base"][k], row["r1"][k]
        row[f"{k}_verdict"] = ("faster beyond the interval" if max(r) < min(b) else
                               "slower beyond the interval" if min(r) > max(b) else "within the interval")
        row[f"{k}_ratio_r1_over_base"] = round(sum(r) / sum(b), 3)
    row["identity_equal"] = all(arms["base"][i]["workloads"][w]["identity_conv0"] ==
                                arms["r1"][i]["workloads"][w]["identity_conv0"] for i in (0, 1))
    out["workloads"][w] = row
out["mm_prefix_fastpath"] = {a: [b["mm_prefix_fastpath"] for b in arms[a]] for a in arms}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for w, v in out["workloads"].items():
    print(f"{w:22} t2+ base {v['base']['turn2plus_median_ms']} r1 {v['r1']['turn2plus_median_ms']} "
          f"x{v['turn2plus_median_ms_ratio_r1_over_base']} {v['turn2plus_median_ms_verdict']}; "
          f"t1 x{v['turn1_median_ms_ratio_r1_over_base']} {v['turn1_median_ms_verdict']}; "
          f"stripped {v['r1']['stripped_items']}; identity {v['identity_equal']}")
