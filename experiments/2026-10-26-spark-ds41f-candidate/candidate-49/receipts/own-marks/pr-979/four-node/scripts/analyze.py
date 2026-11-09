#!/usr/bin/env python3
"""Summarize the #979 four-node ABBA runs: per topology, T and op, the candidate-49 and
baseline (candidate-49 without #979) boots' max-rank medians (and rank-0), their across-boot
interval, the change, whether it is beyond the interval, the anchors' drift, and bit-exactness
(per-rank output digests equal across all four runs, and equal to the in-process anchor).

usage: analyze.py <out-dir> > analysis.json
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
ARMS = {"c49": "candidate-49", "no979": "candidate-49 without #979 (0762..0765, 0767)"}
OPS = {"project_exchange_rows": "anchor_p0_serial", "gather_mxfp8": "anchor_g0_pynccl_group"}
summary = {"arms": ARMS, "metric": "median over 40 timed iterations of the per-iteration max over the 4 ranks (us); rank0 alongside",
           "interval": "the two boots' medians of each arm; 'beyond' when the arms' intervals do not overlap", "topologies": {}}
for topo in ("ring4", "direct"):
    runs = {}
    for arm in ARMS:
        for boot in (1, 2):
            p = out / f"{topo}-{arm}-boot{boot}.json"
            if p.exists() and p.stat().st_size:
                runs[(arm, boot)] = json.loads(p.read_text())
    if not runs:
        continue
    t = summary["topologies"][topo] = {"runs": sorted(f"{topo}-{a}-boot{b}" for a, b in runs), "ops": {}}
    for op, anchor in OPS.items():
        rows = t["ops"][op] = {}
        tokens = sorted({int(k) for r in runs.values() for k in r[op]})
        for T in tokens:
            cell = {}
            for arm in ARMS:
                boots = [runs[(arm, b)][op][str(T)] for b in (1, 2) if (arm, b) in runs]
                cell[arm] = {
                    "max_rank_medians_us": [x["integrated"]["max_rank"]["median_us"] for x in boots],
                    "max_rank_median_ci95_us": [x["integrated"]["max_rank"]["median_ci95_us"] for x in boots],
                    "max_rank_p10_p90_us": [[x["integrated"]["max_rank"]["p10_us"], x["integrated"]["max_rank"]["p90_us"]] for x in boots],
                    "rank0_medians_us": [x["integrated"]["rank0"]["median_us"] for x in boots],
                    "anchor_max_rank_medians_us": [x[anchor]["max_rank"]["median_us"] for x in boots],
                    "route": boots[0]["per_rank"][0]["route"] if boots else None,
                }
            c, b = cell["c49"]["max_rank_medians_us"], cell["no979"]["max_rank_medians_us"]
            if c and b:
                cm, bm = sum(c) / len(c), sum(b) / len(b)
                cell["change_pct"] = round(100 * (cm / bm - 1), 1)
                cell["change_us"] = round(cm - bm, 1)
                cell["beyond_interval"] = ("gain" if max(c) < min(b) else "loss" if min(c) > max(b) else "within")
                c0, b0 = cell["c49"]["rank0_medians_us"], cell["no979"]["rank0_medians_us"]
                cell["rank0_change_pct"] = round(100 * (sum(c0) / len(c0) / (sum(b0) / len(b0)) - 1), 1)
                cell["rank0_beyond_interval"] = ("gain" if max(c0) < min(b0) else "loss" if min(c0) > max(b0) else "within")
            digests = {}
            anchor_equal = True
            for (arm, boot), r in runs.items():
                per_rank = r[op][str(T)]["per_rank"]
                digests[f"{arm}-boot{boot}"] = [x["sha256_out"] for x in per_rank]
                anchor_equal &= all(x["equal_to_anchor"] for x in per_rank)
                if op == "project_exchange_rows":
                    anchor_equal &= all(x["then_equal"] for x in per_rank)
            ref = next(iter(digests.values()))
            cell["bit_exact_across_runs"] = all(v == ref for v in digests.values())
            cell["equal_to_in_process_anchor"] = anchor_equal
            cell["rank_digests_sha256_prefix"] = [d[:16] for d in ref]
            rows[str(T)] = cell
print(json.dumps(summary, indent=1))
