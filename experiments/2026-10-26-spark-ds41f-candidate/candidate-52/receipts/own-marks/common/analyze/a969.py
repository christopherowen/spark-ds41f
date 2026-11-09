#!/usr/bin/env python3
"""a969.py OUTDIR: summary of PR #969's own-mark runs (bench equivalent x5 per boot, rows identity, tests).
Per cell (workload, M, mode): each boot's median step_us over its reps; an arm's interval is the range of
its two boot medians. 'faster' / 'slower' when the candidate's two boots lie wholly below / above the
base's two; otherwise 'within'. The single-call view (rep 1 only, the PR's one call per boot) is kept too."""
import json
import os
import re
import statistics
import sys

O = sys.argv[1]
ARMS = ["base-boot1", "c52-boot1", "c52-boot2", "base-boot2"]
WL = {"cold_disjoint_uniform": "cold", "zipfian_working_set": "zipf", "hot_working_set": "hot"}
MODES = ["two_plane_nocache", "two_plane_cache64k", "packed_nocache", "packed_cache64k"]


def cmp(c, b):
    if max(c) < min(b):
        return "faster"
    if min(c) > max(b):
        return "slower"
    return "within"


S = {"reader": {}, "cells": {}, "rows": {}, "pytest": {}}
data = {}
for a in ARMS:
    p = os.path.join(O, f"bench-cache-{a}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        data[a] = d
        S["reader"][a] = {k: d[k] for k in ("reader_extension", "reader_extension_sha256", "reader_source_digest")}
lines = []
counts = {"faster": 0, "within": 0, "slower": 0}
for wk, wn in WL.items():
    ms = [r["M"] for r in next(iter(data.values()))["reps"][0][wk]] if data else []
    for m in ms:
        for mode in MODES:
            cell = {"boots": {}, "rep1": {}, "hit_rate": {}, "read_calls_per_step": {}}
            for a, d in data.items():
                vals = []
                for rep in d["reps"]:
                    r = next(x for x in rep[wk] if x["M"] == m)["modes"][mode]
                    vals.append(r["step_us"])
                cell["boots"][a] = {"median": round(statistics.median(vals), 2), "reps": vals}
                cell["rep1"][a] = vals[0]
                r1 = next(x for x in d["reps"][0][wk] if x["M"] == m)["modes"][mode]
                cell["hit_rate"][a] = r1.get("hit_rate")
                cell["read_calls_per_step"][a] = r1.get("read_calls_per_step")
            b = [cell["boots"][a]["median"] for a in cell["boots"] if a.startswith("base")]
            c = [cell["boots"][a]["median"] for a in cell["boots"] if a.startswith("c52")]
            if b and c:
                cell["base_range"] = [min(b), max(b)]
                cell["c52_range"] = [min(c), max(c)]
                cell["c52_over_base_mean"] = round(statistics.fmean(c) / statistics.fmean(b), 4)
                cell["verdict_5rep_medians"] = cmp(c, b)
                b1 = [cell["rep1"][a] for a in cell["rep1"] if a.startswith("base")]
                c1 = [cell["rep1"][a] for a in cell["rep1"] if a.startswith("c52")]
                cell["verdict_rep1_only"] = cmp(c1, b1)
                if mode.endswith("cache64k"):
                    counts[cell["verdict_5rep_medians"]] += 1
            key = f"{wn}/M{m}/{mode}"
            S["cells"][key] = cell
            if mode.endswith("cache64k"):
                lines.append(f"{key:34s} base {cell.get('base_range')} c52 {cell.get('c52_range')} "
                             f"ratio {cell.get('c52_over_base_mean')} -> {cell.get('verdict_5rep_medians')} (rep1: {cell.get('verdict_rep1_only')})"
                             f" hit {set(cell['hit_rate'].values())} reads {set(cell['read_calls_per_step'].values())}")
S["cache64k_cell_counts"] = counts
lines.append("cache64k cells: " + json.dumps(counts))
for a in ARMS:
    p = os.path.join(O, f"rows-{a}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        S["rows"][a] = {"digest": d["digest"], "all_modes_agree": d["all_modes_agree"]}
if S["rows"]:
    S["rows_identical_across_arms"] = len({v["digest"] for v in S["rows"].values()}) == 1
    lines.append("rows: " + json.dumps(S["rows"]) + f" identical across arms: {S['rows_identical_across_arms']}")
for k in (1, 2):
    p = os.path.join(O, f"pytest-c52-boot{k}.txt")
    if os.path.exists(p):
        t = open(p).read()
        m = re.findall(r"=+ (.*?(?:passed|failed).*?) in [\d.]+s", t)
        S["pytest"][f"c52-boot{k}"] = m[-1] if m else t.strip().splitlines()[-1]
lines.append("pytest: " + json.dumps(S["pytest"]))
lines.append("reader: " + json.dumps(S["reader"]))
print("\n".join(lines))
json.dump(S, open(os.path.join(O, "summary.json"), "w"), indent=1)
open(os.path.join(O, "summary.txt"), "w").write("\n".join(lines) + "\n")
