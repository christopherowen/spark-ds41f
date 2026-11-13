#!/usr/bin/env python3
"""A/A of the #979 rework bench: base0 (candidate-49 without #979) boots 1-2 (main sequence)
against base0 boots 3-4 (run right after it), the same arm and code, judged with the same
two-boot interval rule as the arms; and, for every cell where the corrected arm (corrB) reads
slower than base0 beyond the interval, whether the A/A reads beyond the interval there too and
how the corrected arm's two boots sit among base0's four.

usage: aa.py <raw-dir> <analysis-main.json> > aa.json
"""
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
main = json.load(open(sys.argv[2]))


def load(topo, boot):
    return json.loads((raw / f"{topo}-base0-boot{boot}.json").read_text())


def verdict(a, b):
    return "gain" if max(a) < min(b) else "loss" if min(a) > max(b) else "within"


out = {"what": "base0 boots 3-4 vs base0 boots 1-2: identical arm, identical code",
       "cells": {}, "beyond_max_rank": 0, "beyond_rank0": 0, "cells_total": 0, "corrB_loss_cells": []}
for topo in ("ring4", "direct"):
    runs = {b: load(topo, b) for b in (1, 2, 3, 4)}
    for op in ("project_exchange_rows", "gather_mxfp8"):
        for T in runs[1][op]:
            m = {b: runs[b][op][T]["integrated"]["max_rank"]["median_us"] for b in runs}
            r0 = {b: runs[b][op][T]["integrated"]["rank0"]["median_us"] for b in runs}
            dig = {b: [x["sha256_out"] for x in runs[b][op][T]["per_rank"]] for b in runs}
            vm = verdict([m[3], m[4]], [m[1], m[2]])
            v0 = verdict([r0[3], r0[4]], [r0[1], r0[2]])
            key = f"{topo} {op} T={T}"
            out["cells"][key] = {
                "base0_boot1_2_max_rank_us": [m[1], m[2]], "base0_boot3_4_max_rank_us": [m[3], m[4]],
                "max_rank_aa": vm, "max_rank_aa_change_pct": round(100 * ((m[3] + m[4]) / (m[1] + m[2]) - 1), 1),
                "base0_boot1_2_rank0_us": [r0[1], r0[2]], "base0_boot3_4_rank0_us": [r0[3], r0[4]],
                "rank0_aa": v0, "rank0_aa_change_pct": round(100 * ((r0[3] + r0[4]) / (r0[1] + r0[2]) - 1), 1),
                "digests_equal_to_boot1": all(dig[b] == dig[1] for b in runs),
            }
            out["cells_total"] += 1
            out["beyond_max_rank"] += vm != "within"
            out["beyond_rank0"] += v0 != "within"
            cell = main["topologies"][topo]["ops"][op][T]
            cmp = cell["corrB_vs_base0"]
            if cmp["max_rank"] == "loss" or cmp["rank0"] == "loss":
                cb = cell["arms"]["corrB"]
                b4 = sorted([m[1], m[2], m[3], m[4]])
                b4r = sorted([r0[1], r0[2], r0[3], r0[4]])
                out["corrB_loss_cells"].append({
                    "cell": key, "identical_code": cmp["identical_code"],
                    "corrB_max_rank_us": cb["max_rank_medians_us"], "base0_4_boots_max_rank_us": [m[1], m[2], m[3], m[4]],
                    "corrB_rank0_us": cb["rank0_medians_us"], "base0_4_boots_rank0_us": [r0[1], r0[2], r0[3], r0[4]],
                    "corrB_vs_base0": f"max-rank {cmp['max_rank']} ({cmp['change_pct']:+.1f}%), rank0 {cmp['rank0']} ({cmp['rank0_change_pct']:+.1f}%)",
                    "corrB_vs_base0_4_boots": "max-rank " + verdict(cb["max_rank_medians_us"], b4) + ", rank0 " + verdict(cb["rank0_medians_us"], b4r),
                    "aa_same_cell": f"max-rank {vm} ({out['cells'][key]['max_rank_aa_change_pct']:+.1f}%), rank0 {v0} ({out['cells'][key]['rank0_aa_change_pct']:+.1f}%)",
                })
out["false_positive_rate_max_rank"] = round(out["beyond_max_rank"] / out["cells_total"], 3)
out["false_positive_rate_rank0"] = round(out["beyond_rank0"] / out["cells_total"], 3)
print(json.dumps(out, indent=1))
