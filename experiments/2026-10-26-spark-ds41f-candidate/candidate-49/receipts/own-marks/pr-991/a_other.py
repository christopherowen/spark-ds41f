#!/usr/bin/env python3
"""Summaries for PRs #993 (our score harness), #991 (its bench), #1009 (its bench),
#996 (our route_local harness). usage: a_other.py <pr> <json dir> <out.json>"""
import json
import sys
from pathlib import Path

pr, d, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])


def load(name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def cell(b, c, lower_is_better=True):
    mb, mc = sum(b) / len(b), sum(c) / len(c)
    return {
        "base_boots": [round(x, 3) for x in b],
        "cand_boots": [round(x, 3) for x in c],
        "speedup_of_means": round(mb / mc, 3),
        "gain_beyond_interval": max(c) < min(b),
        "worse_beyond_interval": min(c) > max(b),
    }


def tally(cells):
    return {
        "cells": len(cells),
        "gain_beyond_interval": sum(x["gain_beyond_interval"] for x in cells),
        "worse_beyond_interval": sum(x["worse_beyond_interval"] for x in cells),
        "within_interval": sum(not x["gain_beyond_interval"] and not x["worse_beyond_interval"] for x in cells),
    }


res = {}
if pr == "993":
    c = [load("score-candidate49-boot1.json"), load("score-candidate49-boot2.json")]
    a = [load("score-1107-boot1.json"), load("score-1107-boot2.json")]
    res["sanity"] = [x["sanity"] for x in c]
    res["exactness_all_equal"] = [x["exactness_all_equal"] for x in c]
    res["exactness_cases"] = len(c[0]["exactness"])
    res["exactness_hashes_equal_across_boots"] = [e["scores_sha256"] for e in c[0]["exactness"]] == [
        e["scores_sha256"] for e in c[1]["exactness"]
    ]
    res["cells"] = []
    allc = []
    for i, row in enumerate(c[0]["timing"]):
        key = f'{row["mode"]} p{row["page_size"]} w{row["width"]} rows={row["rows"]} L={row["L"]}'
        ent = {"case": key}
        for kind in ("warm", "cold"):
            b = [x["timing"][i][kind]["base"]["median_us"] for x in c]
            k = [x["timing"][i][kind]["cand"]["median_us"] for x in c]
            ent[kind] = cell(b, k)
            allc.append(ent[kind])
            if a[0] and a[1]:
                aa = [x["timing"][i][kind]["base"]["median_us"] / x["timing"][i][kind]["cand"]["median_us"] for x in a]
                ent[kind]["aa_1107_ratio_boots"] = [round(v, 3) for v in aa]
        res["cells"].append(ent)
    res["tally"] = tally(allc)
    if a[0] and a[1]:
        res["aa_1107_exactness_all_equal"] = [x["exactness_all_equal"] for x in a]
elif pr == "991":
    c = [load("bench-candidate49-boot1.json"), load("bench-candidate49-boot2.json")]
    res["sass"] = {k: {"sw": v["sw"]["static_sass_instructions"], "hw": v["hw"]["static_sass_instructions"],
                       "hw_f2fp": v["hw"]["f2fp_e2m1_pack_ab_count"]} for k, v in c[0]["sass"].items()}
    res["cells"] = {}
    allc = []
    bit = True
    for bench, rows in c[0]["benchmarks"].items():
        res["cells"][bench] = []
        for i, row in enumerate(rows):
            b = [x["benchmarks"][bench][i]["sw"]["median_us"] for x in c]
            k = [x["benchmarks"][bench][i]["hw"]["median_us"] for x in c]
            bit = bit and all(x["benchmarks"][bench][i]["bit_exact"] for x in c)
            cl = cell(b, k)
            cl["M"] = row["M"]
            cl["delta_pct_boots"] = [x["benchmarks"][bench][i]["delta_pct"] for x in c]
            res["cells"][bench].append(cl)
            allc.append(cl)
    res["bit_exact_all"] = bit
    res["tally"] = tally(allc)
elif pr == "1009":
    c = [load("bench-candidate49-boot1.json"), load("bench-candidate49-boot2.json")]
    res["bit_exactness"] = [
        [{k: v for k, v in e.items()} for e in x["bit_exactness"]] for x in c
    ]
    res["cells"] = []
    allc = []
    for i, row in enumerate(c[0]["benchmarks"]):
        b = [x["benchmarks"][i]["baseline_us"]["median_us"] for x in c]
        k = [x["benchmarks"][i]["option_b_us"]["median_us"] for x in c]
        cl = cell(b, k)
        cl["case"] = f'TP={row["tp_size"]} T={row["num_tokens"]} L={row["local_rows"]}'
        cl["saved_ms_54_layers_boots"] = [x["benchmarks"][i]["saved_ms_54_layers_option_b"] for x in c]
        res["cells"].append(cl)
        allc.append(cl)
    res["tally"] = tally(allc)
elif pr == "996":
    c = [load("route-local-candidate49-boot1.json"), load("route-local-candidate49-boot2.json")]
    r = [load("route-local-1107-boot1.json"), load("route-local-1107-boot2.json")]
    res["cells"] = []
    allc = {"cached_calls": [], "pass": []}
    for i, row in enumerate(c[0]["rows"]):
        key = f'TP={row["tp"]} T={row["T"]} rank={row["rank"]}{" tail" if row["tail"] else ""}'
        ent = {"case": key, "local_rows": row["local_rows"], "real_rows": row["real_rows"]}
        for metric, name in (("cached_calls_us_median", "cached_calls"), ("pass_us_per_layer_median", "pass"),
                             ("first_call_us_median", "first_call")):
            b = [x["rows"][i][metric] for x in r]
            k = [x["rows"][i][metric] for x in c]
            ent[name] = cell(b, k)
            if name in allc and row["tail"]:
                allc[name].append(ent[name])
        hc = {tuple(h) for x in c for h in x["rows"][i]["out_hashes_real_rows"]}
        hr = {tuple(h) for x in r for h in x["rows"][i]["out_hashes_real_rows"]}
        ent["real_rows_bit_identical_to_1107"] = hc == hr and len(hc) == 1
        ent["pad_ids_all_minus1"] = {"candidate49": [x["rows"][i]["pad_ids_all_minus1"] for x in c],
                                     "1107": [x["rows"][i]["pad_ids_all_minus1"] for x in r]}
        res["cells"].append(ent)
    res["tally_tail_rank"] = {k: tally(v) for k, v in allc.items()}
    res["all_real_rows_bit_identical"] = all(e["real_rows_bit_identical_to_1107"] for e in res["cells"])
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: v for k, v in res.items() if k in ("tally", "tally_tail_rank", "bit_exact_all",
                                                        "exactness_all_equal", "all_real_rows_bit_identical",
                                                        "sanity")}, indent=1))
