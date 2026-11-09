# usage: python3 a987b.py JSON_DIR OUT.json
# #987 supplement: boots 3-4 (h987b.py: + kernel_isolated, + WO-A alone), and the four-boot
# pooled comparison of every metric both scripts record (boots 1-4 of each arm; the same
# interval rule over four boots: faster when every c51 boot beats every base boot).
import json
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
A = {(a, k): load(f"{d}/wob-{a}-boot{k}.json") for a in ("base", "c51") for k in (1, 2)}
Bs = {(a, k): load(f"{d}/wobb-{a}-boot{k}.json") for a in ("base", "c51") for k in (3, 4)}
ALL = {**A, **Bs}
out = {"supplement_boots_3_4": {}, "pooled_boots_1_4": {}, "counts_supplement": {}, "counts_pooled": {},
       "errors": {f"{a}-boot{k}": r["errors"] for (a, k), r in ALL.items() if r["errors"]}}
m_sup = {
    "wob_graph_hot": lambda c: c["wob"]["graph_hot"]["median_us"],
    "wob_graph_cold": lambda c: c["wob"]["graph_cold"]["median_us"],
    "wob_kernel_isolated": lambda c: c["wob"]["kernel_isolated_us"],
    "e2e_graph_hot": lambda c: c["e2e"]["graph_hot"]["median_us"],
    "e2e_graph_cold": lambda c: c["e2e"]["graph_cold"]["median_us"],
    "e2e_kernel_isolated": lambda c: c["e2e"]["kernel_isolated_us"],
    "woa_graph_hot": lambda c: c["woa"]["graph_hot"]["median_us"],
    "woa_graph_cold": lambda c: c["woa"]["graph_cold"]["median_us"],
    "woa_kernel_isolated": lambda c: c["woa"]["kernel_isolated_us"],
}
m_pool = {k: v for k, v in m_sup.items() if "kernel_isolated" not in k and not k.startswith("woa")}
m_pool["wob_eager"] = lambda c: c["wob"]["eager_us"]
m_pool["e2e_eager"] = lambda c: c["e2e"]["eager_us"]
for key in Bs[("c51", 3)]["cells"]:
    out["supplement_boots_3_4"][key] = {}
    out["pooled_boots_1_4"][key] = {}
    for m in [x for x in Bs[("c51", 3)]["cells"][key] if x.isdigit()]:
        band = "97..128" if int(m) > 96 else "<=96"
        row = {"bit_exact_vs_prefill_all_runs": all(ALL[ak]["cells"][key][m]["wob_bit_exact_vs_prefill"] and ALL[ak]["cells"][key][m]["e2e_bit_exact_vs_prefill"] for ak in ALL),
               "hash_equal_all_8_runs": len({ALL[ak]["cells"][key][m]["wob_out_hash"] for ak in ALL}) == 1 and len({ALL[ak]["cells"][key][m]["e2e_out_hash"] for ak in ALL}) == 1}
        for name, get in m_sup.items():
            b = [get(Bs[("base", k)]["cells"][key][m]) for k in (3, 4)]
            c = [get(Bs[("c51", k)]["cells"][key][m]) for k in (3, 4)]
            row[name] = fmt(b, c)
            v = verdict(b, c)
            out["counts_supplement"].setdefault(f"{key}:{name}", {}).setdefault(band, {}).setdefault(v, 0)
            out["counts_supplement"][f"{key}:{name}"][band][v] += 1
        out["supplement_boots_3_4"][key][m] = row
        prow = {}
        for name, get in m_pool.items():
            b = [get(ALL[("base", k)]["cells"][key][m]) for k in (1, 2, 3, 4)]
            c = [get(ALL[("c51", k)]["cells"][key][m]) for k in (1, 2, 3, 4)]
            prow[name] = fmt(b, c)
            v = verdict(b, c)
            out["counts_pooled"].setdefault(f"{key}:{name}", {}).setdefault(band, {}).setdefault(v, 0)
            out["counts_pooled"][f"{key}:{name}"][band][v] += 1
        out["pooled_boots_1_4"][key][m] = prow
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for sec in ("supplement_boots_3_4", "pooled_boots_1_4"):
    print("=====", sec)
    for key, rows in out[sec].items():
        print(key)
        for m, row in rows.items():
            print(f"  M={m}")
            for k, v in row.items():
                print(f"     {k:28s} {v}")
print("errors:", out["errors"])
