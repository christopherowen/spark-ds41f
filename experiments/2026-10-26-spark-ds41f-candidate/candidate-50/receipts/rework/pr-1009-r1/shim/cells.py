"""Own-mark cells from the ABBA runs (bench_abba.py) and #1009's own bench, two boots.

A = candidate-50 with the corrected 0778 (front_1009, and front_1107 in the same
process), B = candidate-50 without 0778 (front_1107). A cell gains beyond the interval
when A's slower boot of front_1009 is faster than B's faster boot of front_1107.

usage: python3 cells.py <json dir> <out.json>
"""

import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
BOOTS = (1, 2)
METRICS = ("graph", "kernels", "eager")


def load(name):
    return json.loads((d / name).read_text())


abba = {(arm, b): load(f"bench-abba-{arm}-boot{b}.json") for arm in ("fix", "no1009") for b in BOOTS}
own = {b: load(f"bench-own-fix-boot{b}.json") for b in BOOTS}
cells = []
for i, ref in enumerate(abba[("no1009", 1)]["cells"]):
    key = (ref["tp_size"], ref["num_tokens"])
    cell = dict(tp_size=key[0], num_tokens=key[1], local_rows=ref["local_rows"], real_rows=ref["real_rows"])
    for m in METRICS:
        a = [abba[("fix", b)]["cells"][i]["front_1009"][m]["median_us"] for b in BOOTS]
        a_in = [abba[("fix", b)]["cells"][i]["front_1107"][m]["median_us"] for b in BOOTS]
        bb = [abba[("no1009", b)]["cells"][i]["front_1107"][m]["median_us"] for b in BOOTS]
        for b in BOOTS:
            assert (abba[("fix", b)]["cells"][i]["tp_size"], abba[("fix", b)]["cells"][i]["num_tokens"]) == key
        cell[m] = dict(
            corrected_front_1009_us=a,
            no1009_front_1107_us=bb,
            corrected_tree_front_1107_us=a_in,
            speedup=round((sum(bb) / len(bb)) / (sum(a) / len(a)), 3),
            saved_us=round(sum(bb) / len(bb) - sum(a) / len(a), 3),
            faster_beyond_interval=max(a) < min(bb),
            front_1107_same_in_both_trees=round(abs(sum(a_in) - sum(bb)) / sum(bb) * 100, 2),
        )
    oc = [own[b]["benchmarks"][i] for b in BOOTS]
    for o in oc:
        assert (o["tp_size"], o["num_tokens"]) == key
    cell["own_bench_eager"] = dict(
        baseline_us=[o["baseline_us"]["median_us"] for o in oc],
        option_b_us=[o["option_b_us"]["median_us"] for o in oc],
        speedup=[o["speedup_option_b_vs_baseline"] for o in oc],
        faster_beyond_interval=max(o["option_b_us"]["median_us"] for o in oc)
        < min(o["baseline_us"]["median_us"] for o in oc),
    )
    cell["kernels_per_sequence"] = dict(
        front_1009=abba[("fix", 1)]["cells"][i]["front_1009"]["kernels"]["kernels_per_sequence"],
        front_1107=abba[("no1009", 1)]["cells"][i]["front_1107"]["kernels"]["kernels_per_sequence"],
    )
    cells.append(cell)

exact = []
for b in BOOTS:
    for e in own[b]["bit_exactness"]:
        exact.append(sum(e["option_b_diffs"].values()) + sum(e["option_a_diffs"].values()))
summary = dict(
    cells=cells,
    all_faster_beyond_interval={m: all(c[m]["faster_beyond_interval"] for c in cells) for m in METRICS},
    own_bench_all_faster_beyond_interval=all(c["own_bench_eager"]["faster_beyond_interval"] for c in cells),
    own_bench_bit_exact=dict(shapes_per_boot=len(own[1]["bit_exactness"]), total_differing=sum(exact)),
    has_1009=dict(fix=abba[("fix", 1)]["has_1009"], no1009=abba[("no1009", 1)]["has_1009"]),
)
Path(sys.argv[2]).write_text(json.dumps(summary, indent=1) + "\n")
for c in cells:
    print(
        f"TP{c['tp_size']} T={c['num_tokens']}: "
        + "; ".join(
            f"{m} {sum(c[m]['no1009_front_1107_us'])/2:.1f} -> {sum(c[m]['corrected_front_1009_us'])/2:.1f} us "
            f"(x{c[m]['speedup']}, beyond={c[m]['faster_beyond_interval']})"
            for m in METRICS
        )
    )
print(json.dumps({k: v for k, v in summary.items() if k != "cells"}))
