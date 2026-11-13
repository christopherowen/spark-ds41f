#!/usr/bin/env python3
"""Compare the corrected #996's route_local bench boots with the recorded own-mark boots
of 1107 and candidate-49 (same harness, same host), and summarize the inference-mode
exactness and timing runs.

usage: analyze_996.py <own-marks json dir> <rework json dir> <out.json>
"""
import json
import sys

own, rw, out_path = sys.argv[1:4]


def load(path):
    with open(path) as f:
        return json.load(f)


def rows(path):
    return {(r["tp"], r["T"], r["rank"]): r for r in load(path)["rows"]}


arms = {
    "1107": [rows(f"{own}/route-local-1107-boot{b}.json") for b in (1, 2)],
    "candidate-49": [rows(f"{own}/route-local-candidate49-boot{b}.json") for b in (1, 2)],
    "corrected": [rows(f"{rw}/route-local-corrected-boot{b}.json") for b in (1, 2)],
}
res = {"reference": "recorded own-mark boots (candidate-49/receipts/own-marks/pr-996/json), not re-measured",
       "rule": "gain beyond the interval: the slower corrected boot beats the faster 1107 boot; "
               "worse beyond it: the faster corrected boot is slower than the slower 1107 boot",
       "bench": [], "summary": {}}
for key in sorted(arms["1107"][0]):
    tp, T, rank = key
    cell = {"tp": tp, "T": T, "rank": rank, "tail": arms["1107"][0][key]["tail"]}
    for metric in ("cached_calls_us_median", "pass_us_per_layer_median"):
        vals = {a: [b[key][metric] for b in boots] for a, boots in arms.items()}
        ref_lo, ref_hi = min(vals["1107"]), max(vals["1107"])
        cor_lo, cor_hi = min(vals["corrected"]), max(vals["corrected"])
        c49_lo, c49_hi = min(vals["candidate-49"]), max(vals["candidate-49"])
        cell[metric] = {
            **{a: v for a, v in vals.items()},
            "speedup_vs_1107": round(sum(vals["1107"]) / sum(vals["corrected"]), 3),
            "vs_1107": "faster" if cor_hi < ref_lo else "slower" if cor_lo > ref_hi else "within",
            "vs_candidate-49": "faster" if cor_hi < c49_lo else "slower" if cor_lo > c49_hi else "within",
        }
    hashes = {a: sorted({h for b in boots for h in map(tuple, b[key]["out_hashes_real_rows"])})
              for a, boots in arms.items()}
    cell["real_rows_hashes_equal_1107"] = hashes["corrected"] == hashes["1107"]
    cell["stable_across_layers_and_steps"] = all(b[key]["outputs_stable_across_layers_and_steps"]
                                                 for b in arms["corrected"])
    cell["pad_ids_all_minus1"] = [b[key]["pad_ids_all_minus1"] for b in arms["corrected"]]
    res["bench"].append(cell)

tail = [c for c in res["bench"] if c["tail"]]
other = [c for c in res["bench"] if not c["tail"]]
for name, metric in (("cached", "cached_calls_us_median"), ("pass", "pass_us_per_layer_median")):
    res["summary"][f"tail_cells_{name}_faster_beyond_interval"] = (
        f"{sum(c[metric]['vs_1107'] == 'faster' for c in tail)} of {len(tail)}")
    res["summary"][f"tail_cells_{name}_vs_candidate49"] = {
        k: sum(c[metric]["vs_candidate-49"] == k for c in tail) for k in ("faster", "within", "slower")}
    res["summary"][f"non_tail_cells_{name}_vs_1107"] = {
        k: sum(c[metric]["vs_1107"] == k for c in other) for k in ("faster", "within", "slower")}
res["summary"]["real_rows_hashes_equal_1107"] = f"{sum(c['real_rows_hashes_equal_1107'] for c in res['bench'])} of {len(res['bench'])}"

ex = [load(f"{rw}/exact-corrected-boot{b}.json") for b in (1, 2)]
res["exact_inference_mode"] = [{
    "boot": b + 1, "reads_version": e["reads_version"], "all_layers_equal_1107": e["all_layers_equal"],
    "cells": len(e["cells"]), "layers_compared": sum(c["layers_compared"] for c in e["cells"]),
    "layers_equal": sum(c["layers_torch_equal_to_1107"] for c in e["cells"]),
    "gathered_equal_full_rows": f"{sum(f['gathered_torch_equal_to_full_rows'] for f in e['full_rows'])} of {len(e['full_rows'])}",
    "is_padding_is_inference": all(c["is_padding_is_inference"] for c in e["cells"]),
    "padded_ids_all_minus1": all(c["padded_ids_all_minus1"] in (True, None) for c in e["cells"]),
} for b, e in enumerate(ex)]

ib = [{(r["tp"], r["T"], r["rank"]): r for r in load(f"{rw}/ibench-corrected-boot{b}.json")["rows"]} for b in (1, 2)]
res["ibench_inference_mode"] = []
for key in sorted(ib[0]):
    tp, T, rank = key
    r = {"tp": tp, "T": T, "rank": rank, "tail": ib[0][key]["tail"]}
    for metric in ("cached_calls_us_median", "pass_us_per_layer_median"):
        a = [x[key]["1107"][metric] for x in ib]
        c = [x[key]["candidate"][metric] for x in ib]
        r[metric] = {"1107_mask_logic": a, "corrected": c,
                     "vs_1107": "faster" if max(c) < min(a) else "slower" if min(c) > max(a) else "within"}
    res["ibench_inference_mode"].append(r)
itail = [r for r in res["ibench_inference_mode"] if r["tail"]]
iother = [r for r in res["ibench_inference_mode"] if not r["tail"]]
for name, metric in (("cached", "cached_calls_us_median"), ("pass", "pass_us_per_layer_median")):
    res["summary"][f"ibench_tail_{name}_faster_beyond_interval"] = (
        f"{sum(r[metric]['vs_1107'] == 'faster' for r in itail)} of {len(itail)}")
    res["summary"][f"ibench_non_tail_{name}_vs_1107"] = {
        k: sum(r[metric]["vs_1107"] == k for r in iother) for k in ("faster", "within", "slower")}

with open(out_path, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res["summary"], indent=1))
print(json.dumps(res["exact_inference_mode"], indent=1))
for c in tail:
    m, p = c["cached_calls_us_median"], c["pass_us_per_layer_median"]
    print(f"TP{c['tp']} T{c['T']} r{c['rank']} cached 1107 {m['1107']} c49 {m['candidate-49']} corr {m['corrected']} "
          f"x{m['speedup_vs_1107']} {m['vs_1107']} | pass 1107 {p['1107']} c49 {p['candidate-49']} corr {p['corrected']} "
          f"x{p['speedup_vs_1107']} {p['vs_1107']}")
for c in other:
    m = c["cached_calls_us_median"]
    if m["vs_1107"] != "within":
        print(f"non-tail TP{c['tp']} T{c['T']} r{c['rank']} cached 1107 {m['1107']} corr {m['corrected']} {m['vs_1107']}")
for r in itail:
    m, p = r["cached_calls_us_median"], r["pass_us_per_layer_median"]
    print(f"ibench TP{r['tp']} T{r['T']} r{r['rank']} cached 1107-logic {m['1107_mask_logic']} corr {m['corrected']} {m['vs_1107']}"
          f" | pass {p['1107_mask_logic']} -> {p['corrected']} {p['vs_1107']}")
