# usage: python3 a1001.py JSON_DIR OUT.json
# Reads vocab-{base,c51}-boot{1,2}.json (bench_vocab_2boot_c51.py).
#  in_process_c51: on the candidate arm, the PR's embedded base_1107 tile config
#    (s4a2) against the candidate config (s2), both boots: #1001 alone on the
#    accumulated stack (both carry #963's vocab.py);
#  in_process_base: the same on the base arm (without #963);
#  image_ab: each arm's own serving config (base arm's base1107 cell, c51 arm's candidate
#    cell): #1001 together with #963.
import json
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
R = {(a, k): load(f"{d}/vocab-{a}-boot{k}.json") for a in ("base", "c51") for k in (1, 2)}
out = {"tile_configs_128": {f"{a}-boot{k}": next(iter(next(iter(r["shapes"].values()))["dtypes"]["float32"].values()))["image_tile_configs_128"] for (a, k), r in R.items()},
       "cells": {}, "counts": {}}
PAIRS = {
    "proj_graph": ("proj_base1107_us", "proj_candidate_us"),
    "greedy_graph": ("greedy_epilogue_base1107_us", "greedy_epilogue_candidate_us"),
    "proj_argmax_graph": ("proj_plus_argmax_base1107_us", "proj_plus_argmax_candidate_us"),
    "proj_eager": ("eager_proj_base1107_us", "eager_proj_candidate_us"),
    "greedy_eager": ("eager_greedy_epilogue_base1107_us", "eager_greedy_epilogue_candidate_us"),
}
for shape in R[("c51", 1)]["shapes"]:
    for dt in R[("c51", 1)]["shapes"][shape]["dtypes"]:
        key = f"{shape}_{dt}"
        out["cells"][key] = {}
        for m in R[("c51", 1)]["shapes"][shape]["dtypes"][dt]:
            e = {ak: R[ak]["shapes"][shape]["dtypes"][dt][m] for ak in R}
            row = {"row_tile": e[("c51", 1)]["row_tile"],
                   "proj_hash_equal_all_runs": len({x["proj_hash"] for x in e.values()}) == 1,
                   "greedy_hash_equal_all_runs": len({x["greedy_hash"] for x in e.values()}) == 1,
                   "bit_equal_in_run": all(x["bit_equal"] for x in e.values())}
            band = "97..128" if int(m) > 96 else "<=96"
            for metric, (kb, kc) in PAIRS.items():
                for comp, b, c in (
                    ("in_process_c51", [e[("c51", 1)][kb], e[("c51", 2)][kb]], [e[("c51", 1)][kc], e[("c51", 2)][kc]]),
                    ("in_process_base", [e[("base", 1)][kb], e[("base", 2)][kb]], [e[("base", 1)][kc], e[("base", 2)][kc]]),
                    ("image_ab", [e[("base", 1)][kb], e[("base", 2)][kb]], [e[("c51", 1)][kc], e[("c51", 2)][kc]]),
                ):
                    row[f"{comp}:{metric}"] = fmt(b, c)
                    v = verdict(b, c)
                    out["counts"].setdefault(f"{comp}:{metric}", {}).setdefault(band, {}).setdefault(v, 0)
                    out["counts"][f"{comp}:{metric}"][band][v] += 1
            out["cells"][key][m] = row
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["tile_configs_128"], indent=1))
for key, rows in out["cells"].items():
    print("==", key)
    for m, row in rows.items():
        print(f"  M={m:>3} tile {row['row_tile']} hashes= {row['proj_hash_equal_all_runs']}/{row['greedy_hash_equal_all_runs']} in-run-equal {row['bit_equal_in_run']}")
        for k, v in row.items():
            if ":" in k and not k.endswith("eager"):
                print(f"      {k:36s} {v}")
print(json.dumps(out["counts"], indent=1))
