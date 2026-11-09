#!/usr/bin/env python3
"""Summarize the #979 rework's four-arm ABBA runs (corrA, corrB, base0, orig; 2 boots each,
ring4 and direct) from the bench's rank-0 JSONs.

Per topology, op and T: every arm's two boot medians (max over the 4 ranks per iteration, then
the median over 40 iterations; rank 0's median alongside), each arm against base0 (change of the
boot means, and 'gain'/'loss' when the two arms' boot intervals do not overlap, else 'within'),
the code path each arm runs at that cell, and bit-exactness (every rank's output digest equal
across all runs of the topology, and equal to the in-process anchor). Cells where two arms run
identical code give A/A comparisons; their 'beyond the interval' rate is reported as the
two-boot interval's false-positive rate.

usage: analyze4.py <raw-dir> > analysis.json
"""
import itertools
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
ARMS = ("corrA", "corrB", "base0", "orig")
OPS = {"project_exchange_rows": "anchor_p0_serial", "gather_mxfp8": "anchor_g0_pynccl_group"}


def path(arm, op, topo, route):
    if op == "project_exchange_rows":
        if arm == "base0":
            return "serial(1107)"
        if route["engram_slices"] >= 2:
            return "sliced-pipeline(#979)"
        if arm == "orig":
            return "side-stream-overlap-" + ("roce" if route["roce_should_all_to_all_rows"] else "pynccl")
        return "serial(1107)"
    fits = route["all_fit_roce"]
    if arm == "orig" and fits:
        return "roce-oneshot(#979)"
    if arm == "corrA" and fits and topo == "ring4":
        return "roce-oneshot(#979)"
    return "pynccl-group(1107)"


def verdict(a, b):
    return "gain" if max(a) < min(b) else "loss" if min(a) > max(b) else "within"


def mean(x):
    return sum(x) / len(x)


summary = {
    "metric": "median over 40 timed iterations of the per-iteration max over the 4 ranks (us); rank 0's median alongside",
    "interval": "the two boots' medians of an arm; 'gain'/'loss' when two arms' intervals do not overlap",
    "topologies": {},
}
aa = {"pairs": 0, "beyond_max_rank": 0, "beyond_rank0": 0, "cells": []}
for topo in ("ring4", "direct"):
    runs = {}
    for arm in ARMS:
        for boot in (1, 2):
            p = raw / f"{topo}-{arm}-boot{boot}.json"
            if p.exists() and p.stat().st_size:
                runs[(arm, boot)] = json.loads(p.read_text())
    if not runs:
        continue
    t = summary["topologies"][topo] = {"runs": sorted(f"{topo}-{a}-boot{b}" for a, b in runs), "ops": {}}
    t["sp_prefill_sha256"] = {f"{a}-boot{b}": sorted({x["sp_prefill_sha256"] for x in r["ranks"]}) for (a, b), r in runs.items()}
    t["gpu_warm_ms_min"] = min(w["gpu_busy_ms"] for r in runs.values() for w in r["gpu_warm"])
    for op, anchor in OPS.items():
        rows = t["ops"][op] = {}
        for T in sorted({int(k) for r in runs.values() for k in r[op]}):
            cell = {"arms": {}}
            for arm in ARMS:
                boots = [runs[(arm, b)][op][str(T)] for b in (1, 2) if (arm, b) in runs]
                if not boots:
                    continue
                route = boots[0]["per_rank"][0]["route"]
                cell["arms"][arm] = {
                    "path": path(arm, op, topo, route),
                    "max_rank_medians_us": [x["integrated"]["max_rank"]["median_us"] for x in boots],
                    "max_rank_median_ci95_us": [x["integrated"]["max_rank"]["median_ci95_us"] for x in boots],
                    "rank0_medians_us": [x["integrated"]["rank0"]["median_us"] for x in boots],
                    "anchor_max_rank_medians_us": [x[anchor]["max_rank"]["median_us"] for x in boots],
                    "anchor_rank0_medians_us": [x[anchor]["rank0"]["median_us"] for x in boots],
                    "route": route,
                }
            A = cell["arms"]
            for arm in ("corrA", "corrB", "orig"):
                if arm in A and "base0" in A and len(A[arm]["max_rank_medians_us"]) == 2 and len(A["base0"]["max_rank_medians_us"]) == 2:
                    m, b = A[arm]["max_rank_medians_us"], A["base0"]["max_rank_medians_us"]
                    m0, b0 = A[arm]["rank0_medians_us"], A["base0"]["rank0_medians_us"]
                    cell[f"{arm}_vs_base0"] = {
                        "change_pct": round(100 * (mean(m) / mean(b) - 1), 1),
                        "change_us": round(mean(m) - mean(b), 1),
                        "max_rank": verdict(m, b),
                        "rank0_change_pct": round(100 * (mean(m0) / mean(b0) - 1), 1),
                        "rank0": verdict(m0, b0),
                        "identical_code": A[arm]["path"] == A["base0"]["path"],
                    }
            for x, y in itertools.combinations([a for a in ARMS if a in A and len(A[a]["max_rank_medians_us"]) == 2], 2):
                if A[x]["path"] == A[y]["path"]:
                    vm = verdict(A[x]["max_rank_medians_us"], A[y]["max_rank_medians_us"])
                    v0 = verdict(A[x]["rank0_medians_us"], A[y]["rank0_medians_us"])
                    aa["pairs"] += 1
                    aa["beyond_max_rank"] += vm != "within"
                    aa["beyond_rank0"] += v0 != "within"
                    if vm != "within" or v0 != "within":
                        aa["cells"].append(f"{topo} {op} T={T} {x} vs {y} ({A[x]['path']}): max-rank {vm} "
                                           f"{A[x]['max_rank_medians_us']} vs {A[y]['max_rank_medians_us']}; rank0 {v0} "
                                           f"{A[x]['rank0_medians_us']} vs {A[y]['rank0_medians_us']}")
            # The in-process anchor is the same code in every arm: arm-vs-arm anchor comparisons are A/A too.
            digests, anchor_equal = {}, True
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
aa["false_positive_rate_max_rank"] = round(aa["beyond_max_rank"] / max(aa["pairs"], 1), 3)
aa["false_positive_rate_rank0"] = round(aa["beyond_rank0"] / max(aa["pairs"], 1), 3)
summary["identical_code_aa"] = aa
# anchors: the same code in every arm and boot of a topology; their per-arm 'beyond' verdicts vs base0
anc = {"pairs": 0, "beyond_max_rank": 0, "beyond_rank0": 0, "cells": []}
for topo, t in summary["topologies"].items():
    for op, rows in t["ops"].items():
        for T, cell in rows.items():
            A = cell["arms"]
            for arm in ("corrA", "corrB", "orig"):
                if arm in A and "base0" in A:
                    vm = verdict(A[arm]["anchor_max_rank_medians_us"], A["base0"]["anchor_max_rank_medians_us"])
                    v0 = verdict(A[arm]["anchor_rank0_medians_us"], A["base0"]["anchor_rank0_medians_us"])
                    anc["pairs"] += 1
                    anc["beyond_max_rank"] += vm != "within"
                    anc["beyond_rank0"] += v0 != "within"
                    if vm != "within" or v0 != "within":
                        anc["cells"].append(f"{topo} {op} T={T} anchor {arm} vs base0: max-rank {vm}, rank0 {v0}")
summary["anchor_aa"] = anc
print(json.dumps(summary, indent=1))
