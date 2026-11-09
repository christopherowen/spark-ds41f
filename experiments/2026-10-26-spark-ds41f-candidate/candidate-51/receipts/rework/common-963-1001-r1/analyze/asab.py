# usage: python3 asab.py JSON_DIR OUT.json
# #1001-r1's slot-balanced screen A/B (d963.py sab, c51r1 arm, two boots): four layers per
# shape and head dtype, built s4a2_a, s2_a, s2_b, s4a2_b (SCREEN_TILE_CONFIGS[128]), each
# timed by graph replay twice in one process (v1..v4, v4..v1).
#  config: per boot, each config's value is the mean of its two layers (an early and a late
#    slot); the two boots per config are compared by the interval rule (common.py);
#  aa: the two layers of one config (identical kernels, different slots), per boot.
import json
import statistics
import sys

from common import fmt, ratio, verdict

runs = [json.load(open(f"{sys.argv[1]}/d963-sab-boot{k}.json")) for k in (1, 2)]
out = {"errors": [r["errors"] for r in runs], "cells": {}, "counts": {}}
for key in runs[0]["cells"]:
    cells = [r["cells"][key] for r in runs if key in r["cells"]]
    if len(cells) < 2:
        continue
    med = lambda c, ln: statistics.mean(c[ln]["graph_us"])  # noqa: E731
    s4 = [round((med(c, "s4a2_a") + med(c, "s4a2_b")) / 2, 2) for c in cells]
    s2 = [round((med(c, "s2_a") + med(c, "s2_b")) / 2, 2) for c in cells]
    ki4 = [round((c["s4a2_a"]["kernel_isolated_us"] + c["s4a2_b"]["kernel_isolated_us"]) / 2, 2) for c in cells]
    ki2 = [round((c["s2_a"]["kernel_isolated_us"] + c["s2_b"]["kernel_isolated_us"]) / 2, 2) for c in cells]
    v = verdict(s4, s2)
    M = int(key.split("M=")[1])
    band = "97..128" if M > 96 else ("65..96" if M > 64 else "<=64 (identical kernels)")
    out["counts"].setdefault(band, {}).setdefault(v, 0)
    out["counts"][band][v] += 1
    aa = {f"boot{k + 1}": {"s4a2_a/s4a2_b": round(med(c, "s4a2_a") / med(c, "s4a2_b"), 4),
                           "s2_a/s2_b": round(med(c, "s2_a") / med(c, "s2_b"), 4)} for k, c in enumerate(cells)}
    out["cells"][key] = {"graph_us_s4a2_to_s2": fmt(s4, s2), "graph_ratio": ratio(s4, s2), "verdict": v,
                         "kernel_isolated_us_s4a2_to_s2": fmt(ki4, ki2),
                         "slot_aa_ratio": aa,
                         "bit_equal": all(c["bit_equal"] for c in cells)
                         and len({h for c in cells for h in c["hashes"].values()}) == 1}
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for key, r in out["cells"].items():
    print(f"{key:28s} graph {r['graph_us_s4a2_to_s2']} | kiso {r['kernel_isolated_us_s4a2_to_s2']} | eq {r['bit_equal']} | slot A/A {r['slot_aa_ratio']}")
print(json.dumps(out["counts"], indent=1))
