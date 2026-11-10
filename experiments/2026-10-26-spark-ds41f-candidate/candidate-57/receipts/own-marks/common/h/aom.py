"""Analysis of round 3's own marks (sp8, candidates 55-57). Pure Python; reads raw JSON.

Interval rule (as candidates 49-53's own marks): a cell is faster when the slower candidate
boot beats the faster base boot, slower when the faster candidate boot is slower than the
slower base boot, else the same.

usage: python3 aom.py 957p1k|957p1s|957p2|957p4|988 RAWDIR [PREFIX]
"""

import json
import statistics
import sys
from pathlib import Path


def verdict(base, cand):
    if max(cand) < min(base):
        return "faster"
    if min(cand) > max(base):
        return "slower"
    return "same"


def cmp_cell(base, cand, digits=2):
    b, c = [round(x, digits) for x in base], [round(x, digits) for x in cand]
    mb, mc = statistics.mean(base), statistics.mean(cand)
    return {"base": b, "cand": c, "delta": round(mc - mb, digits + 1),
            "ratio_base_over_cand": round(mb / mc, 4) if mc else None, "verdict": verdict(base, cand)}


def load(raw, pattern):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(raw).glob(pattern))}


def boots(files, prefix, arm):
    return [files[k] for k in sorted(files) if k.startswith(f"{prefix}-{arm}-boot")]


def a957p1k(raw):
    f = load(raw, "proj-*.json")
    b, c = boots(f, "proj", "base"), boots(f, "proj", "c56p1")
    out = {"layouts": {"base": b[0]["layouts"], "c56p1": c[0]["layouts"]}, "cells": {}, "step": {},
           "tp1_hash_equal_across_arms": all(x["tp1_hash"] == b[0]["tp1_hash"] for x in b + c)}
    keys = sorted(b[0]["cells"], key=lambda k: (k.split("_")[0], k.split("_")[1], int(k.split("_m")[1])))
    for k in keys:
        cell = {}
        for metric, get in (("graph_hot_us", lambda d: d["graph_hot"]["median_us"]),
                            ("graph_cold_us", lambda d: d["graph_cold"]["median_us"]),
                            ("kernel_us", lambda d: d["kernel"]["per_call_us"]),
                            ("eager_us", lambda d: d["eager_us"])):
            cell[metric] = cmp_cell([get(x["cells"][k]) for x in b], [get(x["cells"][k]) for x in c])
        out["cells"][k] = cell
    # Owner layers per step: kv_source_layer_ids [2, 8, 14, 20] have ratios 2, 2, 2, 1.
    for tp in (4, 3):
        for rows in sorted({int(k.split("_m")[1]) for k in keys}):
            r1, r2 = f"r1_tp{tp}_m{rows}", f"r2_tp{tp}_m{rows}"
            for metric, get in (("graph_hot_us", lambda d: d["graph_hot"]["median_us"]),
                                ("graph_cold_us", lambda d: d["graph_cold"]["median_us"]),
                                ("kernel_us", lambda d: d["kernel"]["per_call_us"])):
                bb = [3 * get(x["cells"][r2]) + get(x["cells"][r1]) for x in b]
                cc = [3 * get(x["cells"][r2]) + get(x["cells"][r1]) for x in c]
                pb = [2 * get(x["cells"][r2]) + 2 * get(x["cells"][r1]) for x in b]
                pc = [2 * get(x["cells"][r2]) + 2 * get(x["cells"][r1]) for x in c]
                out["step"].setdefault(f"tp{tp}_m{rows}", {})[metric] = {
                    "model_3xr2_1xr1": cmp_cell(bb, cc), "pr_formula_2xr2_2xr1": cmp_cell(pb, pc)}
    return out


def a957p1s(raw, prefix="step"):
    f = load(raw, f"{prefix}-*.json")
    b, c = boots(f, prefix, "base"), boots(f, prefix, "c56p1")
    out = {"model": {"base": b[0]["model"]["compressors"], "c56p1": c[0]["model"]["compressors"],
                     "compressor_bytes": {"base": b[0]["model"]["compressor_bytes"],
                                          "c56p1": c[0]["model"]["compressor_bytes"]},
                     "owner_plans": {"base": b[0]["model"]["owner_plans"], "c56p1": c[0]["model"]["owner_plans"]},
                     "l2_prefetch_enabled": [x["model"].get("l2_prefetch_enabled") for x in b + c]},
           "boot_s": {"base": [x["boot_s"] for x in b], "c56p1": [x["boot_s"] for x in c]},
           "shapes": {}}
    for i, row in enumerate(b[0]["rows"]):
        shape = row["shape"]
        get = lambda x, key: statistics.median(x["rows"][i][key])  # noqa: E731
        cell = {m: cmp_cell([get(x, m) for x in b], [get(x, m) for x in c], 3)
                for m in ("step_ms", "forward_ms", "drafter_ms") if row.get(m)}
        bm = statistics.mean(get(x, "step_ms") for x in b)
        cell["step_delta_pct"] = round(100 * (statistics.mean(get(x, "step_ms") for x in c) - bm) / bm, 2)
        cell["modes"] = {"base": b[0]["rows"][i]["modes"][:1], "c56p1": c[0]["rows"][i]["modes"][:1]}
        out["shapes"][shape] = cell
    return out


def a957p2(raw):
    f = load(raw, "mla-*.json")
    b, c = boots(f, "mla", "base"), boots(f, "mla", "c57p2")
    out = {"plans": {"base": b[0]["plans"], "c57p2": c[0]["plans"]}, "cells": {}}
    for k in b[0]["cells"]:
        get = {"graph_hot_us": lambda d: d["graph_hot"]["median_us"], "kernel_us": lambda d: d["kernel"]["per_call_us"],
               "eager_us": lambda d: d["eager_us"]}
        cell = {m: cmp_cell([g(x["cells"][k]) for x in b], [g(x["cells"][k]) for x in c]) for m, g in get.items()}
        cell["route"] = {"base": b[0]["cells"][k]["route"], "c57p2": c[0]["cells"][k]["route"]}
        hashes = {x["cells"][k]["out_hash"] for x in b + c}
        cell["outputs_equal_all_boots"] = len(hashes) == 1
        cell["step_30_layers_delta_us"] = round(30 * cell["graph_hot_us"]["delta"], 1)
        out["cells"][k] = cell
    return out


def a957p4(raw):
    f = load(raw, "conf-*.json")
    b, c = boots(f, "conf", "base"), boots(f, "conf", "c57p4")
    out = {"cells": {}}
    for k in b[0]["cells"]:
        get = {"graph_hot_us": lambda d: d["graph_hot"]["median_us"], "kernel_us": lambda d: d["kernel"]["per_call_us"],
               "eager_us": lambda d: d["eager_us"]}
        cell = {m: cmp_cell([g(x["cells"][k]) for x in b], [g(x["cells"][k]) for x in c]) for m, g in get.items()}
        cell["two_step_hash_equal_across_arms"] = len({x["cells"][k]["two_step_hash"] for x in b + c}) == 1
        cell["fused_vs_two_step"] = [x["cells"][k]["fused_vs_two_step"] for x in c]
        cell["in_process_two_step_graph_hot_us"] = [x["cells"][k]["two_step_in_process"]["graph_hot"]["median_us"]
                                                    for x in c]
        cell["launches"] = {"base": b[0]["cells"][k]["kernel"]["launches"], "c57p4": c[0]["cells"][k]["kernel"]["launches"]}
        out["cells"][k] = cell
    return out


def a988(raw):
    f = load(raw, "sampler-*.json")
    b, c = boots(f, "sampler", "base"), boots(f, "sampler", "c55dr")
    out = {"cells": {}}
    for k in b[0]["cells"]:
        get = {"events_median_us": lambda d: d["serving_events"]["median_us"],
               "kernel_us": lambda d: d["serving_kernel"]["per_call_us"]}
        cell = {m: cmp_cell([g(x["cells"][k]) for x in b], [g(x["cells"][k]) for x in c]) for m, g in get.items()}
        cell["live_rows_bit_exact"] = [x["cells"][k]["live_bit_exact_vs_nocut_nan_dead"] for x in c]
        cell["rejection_sample_equal"] = [x["cells"][k]["rejection_sample_equal"] for x in c]
        cell["live_hash_equal_across_arms"] = len({x["cells"][k]["live_hash_nocut"] for x in b + c}
                                                  | {x["cells"][k]["live_hash_cut"] for x in c}) == 1
        cell["cand_nocut_events_median_us"] = [x["cells"][k]["nocut_events"]["median_us"] for x in c]
        out["cells"][k] = cell
    return out


if __name__ == "__main__":
    which, raw = sys.argv[1], sys.argv[2]
    fn = {"957p1k": a957p1k, "957p2": a957p2, "957p4": a957p4, "988": a988}.get(which)
    res = fn(raw) if fn else a957p1s(raw, sys.argv[3] if len(sys.argv) > 3 else "step")
    print(json.dumps(res, indent=1))
