#!/usr/bin/env python3
"""PR #996 rework r2 (ours): compare round 2's route_local bench boots with the recorded
own-mark boots of 1107 and candidate-49, round 1's recorded boots, and the round 1 tree
re-run in this session as a control (same harness, same host); summarize the
inference-mode exactness, the in-process A/B and the host-cost runs.

usage: analyze_r2.py <own-marks json dir> <round 1 json dir> <round 2 json dir> <out.json>
"""
import json
import sys

own, r1dir, r2dir, out_path = sys.argv[1:5]
METRICS = ("cached_calls_us_median", "pass_us_per_layer_median")


def load(path):
    with open(path) as f:
        return json.load(f)


def rows(path):
    return {(r["tp"], r["T"], r["rank"]): r for r in load(path)["rows"]}


arms = {
    "1107": [rows(f"{own}/route-local-1107-boot{b}.json") for b in (1, 2)],
    "candidate-49": [rows(f"{own}/route-local-candidate49-boot{b}.json") for b in (1, 2)],
    "r1": [rows(f"{r1dir}/route-local-corrected-boot{b}.json") for b in (1, 2)],
    "r1_control": [rows(f"{r2dir}/route-local-r1ctl-boot{b}.json") for b in (1, 2)],
    "r2": [rows(f"{r2dir}/route-local-r2-boot{b}.json") for b in (1, 2)],
}


def cmp(cand, ref):
    """The own mark's rule on two boots each: faster when the slower candidate boot beats
    the faster reference boot, slower when the faster candidate boot is slower than the
    slower reference boot, else within."""
    if max(cand) < min(ref):
        return "faster"
    if min(cand) > max(ref):
        return "slower"
    return "within"


res = {
    "reference": "1107 and candidate-49: the recorded own-mark boots (candidate-49/receipts/own-marks/pr-996/json), "
                 "r1: round 1's recorded boots (receipts/rework/pr-996-r1/json); none re-measured. r1_control: the "
                 "round 1 tree re-run in this session, ABBA with round 2 (r2 b1, r1 b1, r1 b2, r2 b2), supplementary",
    "rule": "faster beyond the interval: the slower r2 boot beats the faster 1107 boot; slower beyond it: the faster "
            "r2 boot is slower than the slower 1107 boot; else within",
    "bench": [], "summary": {}}
for key in sorted(arms["1107"][0]):
    tp, T, rank = key
    ref = arms["1107"][0][key]
    cell = {"tp": tp, "T": T, "rank": rank, "tail": ref["tail"], "local_rows": ref["local_rows"],
            "real_rows": ref["real_rows"], "path": "partials" if ref["local_rows"] <= 256 else "gate"}
    for metric in METRICS:
        vals = {a: [b[key][metric] for b in boots] for a, boots in arms.items()}
        r2, base = vals["r2"], vals["1107"]
        verdict = cmp(r2, base)
        margin = (round(min(r2) - max(base), 3) if verdict == "slower"
                  else round(min(base) - max(r2), 3) if verdict == "faster" else 0.0)
        cell[metric] = {
            **vals,
            "r2_vs_1107": verdict,
            "r2_vs_1107_margin_us": margin,
            "r2_speedup_vs_1107": round(sum(base) / sum(r2), 3),
            "r2_vs_r1_control_in_session": cmp(r2, vals["r1_control"]),
            "r1_control_vs_1107": cmp(vals["r1_control"], base),
            "r1_recorded_vs_1107": cmp(vals["r1"], base),
            "r1_control_vs_r1_recorded": cmp(vals["r1_control"], vals["r1"]),
        }
    hashes = {a: sorted({h for b in boots for h in map(tuple, b[key]["out_hashes_real_rows"])})
              for a, boots in arms.items()}
    cell["real_rows_hashes_equal_1107"] = hashes["r2"] == hashes["1107"]
    cell["stable_across_layers_and_steps"] = all(b[key]["outputs_stable_across_layers_and_steps"]
                                                 for b in arms["r2"])
    cell["pad_ids_all_minus1"] = [b[key]["pad_ids_all_minus1"] for b in arms["r2"]]
    res["bench"].append(cell)

S = res["summary"]
for group, cells in (("tail", [c for c in res["bench"] if c["tail"]]),
                     ("non_tail", [c for c in res["bench"] if not c["tail"]])):
    for name, metric in (("cached", METRICS[0]), ("pass", METRICS[1])):
        for field in ("r2_vs_1107", "r1_recorded_vs_1107", "r1_control_vs_1107",
                      "r2_vs_r1_control_in_session", "r1_control_vs_r1_recorded"):
            S[f"{group}_{name}_{field}"] = {k: sum(c[metric][field] == k for c in cells)
                                            for k in ("faster", "within", "slower")}
        S[f"{group}_{name}_r2_slower_cells"] = [
            f"TP{c['tp']} T{c['T']} r{c['rank']} (+{c[metric]['r2_vs_1107_margin_us']} us)"
            for c in cells if c[metric]["r2_vs_1107"] == "slower"]
S["real_rows_hashes_equal_1107"] = (f"{sum(c['real_rows_hashes_equal_1107'] for c in res['bench'])} "
                                    f"of {len(res['bench'])}")

ex = [load(f"{r2dir}/exact-r2-boot{b}.json") for b in (1, 2)]
res["exact_inference_mode"] = [{
    "boot": b + 1, "reads_version": e["reads_version"], "all_layers_equal_1107": e["all_layers_equal"],
    "cells": len(e["cells"]), "layers_compared": sum(c["layers_compared"] for c in e["cells"]),
    "layers_equal": sum(c["layers_torch_equal_to_1107"] for c in e["cells"]),
    "gathered_equal_full_rows": f"{sum(f['gathered_torch_equal_to_full_rows'] for f in e['full_rows'])} of {len(e['full_rows'])}",
    "is_padding_is_inference": all(c["is_padding_is_inference"] for c in e["cells"]),
    "padded_ids_all_minus1": all(c["padded_ids_all_minus1"] in (True, None) for c in e["cells"]),
    "cache_entries_per_step": {"tail": sorted({n for c in e["cells"] if c["tail"] for n in c["local_mask_cache_entries_per_step"]}),
                               "non_tail": sorted({n for c in e["cells"] if not c["tail"] for n in c["local_mask_cache_entries_per_step"]})},
} for b, e in enumerate(ex)]

inf_ref = [{(c["tp"], c["T"], c["rank"]): c for c in load(f"{own}/inference-mode-1107-boot{b}.json")["cases"]}
           for b in (1, 2)]
res["infer"] = []
for b in (1, 2):
    cases = load(f"{r2dir}/infer-r2-boot{b}.json")["cases"]
    res["infer"].append({
        "boot": b, "ok": sum(c["result"] == "ok" for c in cases), "cases": len(cases),
        "hash_equal_1107_recorded": sum(c.get("out_hash_real_rows") == inf_ref[0][(c["tp"], c["T"], c["rank"])].get("out_hash_real_rows")
                                        for c in cases)})

ib = [{(r["tp"], r["T"], r["rank"]): r for r in load(f"{r2dir}/ibench-r2-boot{b}.json")["rows"]} for b in (1, 2)]
res["ibench_inference_mode"] = []
for key in sorted(ib[0]):
    tp, T, rank = key
    r = {"tp": tp, "T": T, "rank": rank, "tail": ib[0][key]["tail"]}
    for metric in METRICS:
        a = [x[key]["1107"][metric] for x in ib]
        c = [x[key]["candidate"][metric] for x in ib]
        r[metric] = {"1107_mask_logic": a, "r2": c, "r2_vs_1107": cmp(c, a)}
    res["ibench_inference_mode"].append(r)
for group, cells in (("tail", [r for r in res["ibench_inference_mode"] if r["tail"]]),
                     ("non_tail", [r for r in res["ibench_inference_mode"] if not r["tail"]])):
    for name, metric in (("cached", METRICS[0]), ("pass", METRICS[1])):
        S[f"ibench_{group}_{name}_r2_vs_1107"] = {k: sum(r[metric]["r2_vs_1107"] == k for r in cells)
                                                  for k in ("faster", "within", "slower")}

cpu = [load(f"{r2dir}/cpu-r2-boot{b}.json")["rows"] for b in (1, 2)]
res["host_cost_ns_per_layer"] = []
for i, row in enumerate(cpu[0]):
    res["host_cost_ns_per_layer"].append({
        "mode": row["mode"], "tp": row["tp"], "T": row["T"], "rank": row["rank"], "path": row["path"],
        "tail": row["tail"], **{a: [cpu[b][i][a]["host_ns"] for b in (0, 1)] for a in ("1107", "r1", "r2")}})

with open(out_path, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(S, indent=1))
print(json.dumps(res["exact_inference_mode"], indent=1))
print(json.dumps(res["infer"], indent=1))
for c in res["bench"]:
    m, p = c["cached_calls_us_median"], c["pass_us_per_layer_median"]
    print(f"TP{c['tp']} T{c['T']:<5} r{c['rank']} {'tail' if c['tail'] else '    '} {c['path']:<8} cached 1107 {m['1107']} "
          f"r2 {m['r2']} {m['r2_vs_1107']} ({m['r2_vs_1107_margin_us']}) x{m['r2_speedup_vs_1107']} | r1ctl {m['r1_control']} "
          f"| pass 1107 {p['1107']} r2 {p['r2']} {p['r2_vs_1107']} ({p['r2_vs_1107_margin_us']})")
for h in res["host_cost_ns_per_layer"]:
    print(f"host {h['mode'][:5]} TP{h['tp']} T{h['T']:<5} r{h['rank']} {'tail' if h['tail'] else '    '} "
          f"1107 {h['1107']} r1 {h['r1']} r2 {h['r2']}")
