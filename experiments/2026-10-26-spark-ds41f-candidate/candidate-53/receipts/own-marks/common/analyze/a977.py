"""#977 analysis: the PR's bench (base-boot*.jsonl / candidate-boot*.jsonl, image A/B), ids-*.jsonl
(copy_match output identity), gate-*.txt (benchmark_copy_drafts --gate --cases copy)."""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(__file__)); from common import cell, jl
d = sys.argv[1]
def load(p): return {(r["scenario"], r["reqs"], r["ctx"]): r for r in jl(p)}
b1, b2 = load(f"{d}/base-boot1.jsonl"), load(f"{d}/base-boot2.jsonl")
c1, c2 = load(f"{d}/candidate-boot1.jsonl"), load(f"{d}/candidate-boot2.jsonl")
cells = {}
for k in sorted(b1, key=lambda k: (k[0], k[1], k[2])):
    cells[k] = cell([b1[k]["med_us"], b2[k]["med_us"]], [c1[k]["med_us"], c2[k]["med_us"]])
long_ = {k: v for k, v in cells.items() if k[2] >= 65536}
short = {k: v for k, v in cells.items() if k[2] <= 32768}
summ = {"long_64K_1M": {"cells": len(long_), "better": sum(v["verdict"] == "better" for v in long_.values()),
                        "same": sum(v["verdict"] == "same" for v in long_.values()), "worse": sum(v["verdict"] == "worse" for v in long_.values())},
        "short_le_32K": {"cells": len(short), "better": sum(v["verdict"] == "better" for v in short.values()),
                         "same": sum(v["verdict"] == "same" for v in short.values()), "worse": sum(v["verdict"] == "worse" for v in short.values())},
        "worse_cells": {f"{k[0]}/c{k[1]}/{k[2]}": v for k, v in cells.items() if v["verdict"] == "worse"},
        "slower_by_delta_any": {f"{k[0]}/c{k[1]}/{k[2]}": v for k, v in cells.items() if v["delta"] > 0 and k[2] <= 32768}}
ids = {l: {(r["scenario"], r["reqs"], r["ctx"]): r["sources_sha256"] for r in jl(f"{d}/ids-{l}.jsonl")} for l in ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")}
idres = {"boot_repeat_identical": ids["base-boot1"] == ids["base-boot2"] and ids["c53-boot1"] == ids["c53-boot2"]}
diff = [k for k in ids["base-boot1"] if ids["base-boot1"][k] != ids["c53-boot1"][k]]
idres["cases"] = len(ids["base-boot1"])
idres["differ_le_32K"] = [f"{k[0]}/c{k[1]}/{k[2]}" for k in diff if k[2] <= 32768]
idres["differ_gt_32K"] = [f"{k[0]}/c{k[1]}/{k[2]}" for k in diff if k[2] > 32768]
gates = {}
for l in ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2"):
    p = f"{d}/gate-{l}.txt"
    if os.path.exists(p):
        rows = jl(p)
        gates[l] = {"summary": [r for r in rows if "summary" in r], "cases": {f'{r["requests"]}/{r["context"]}/{r["history"]}': r["us"] for r in rows if r.get("case") == "copy"}}
gcells = {}
if all(l in gates for l in ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")):
    for k in gates["base-boot1"]["cases"]:
        gcells[k] = cell([gates["base-boot1"]["cases"][k], gates["base-boot2"]["cases"][k]], [gates["c53-boot1"]["cases"][k], gates["c53-boot2"]["cases"][k]])
out = {"copy_match_cells": {f"{k[0]}/c{k[1]}/{k[2]}": v for k, v in cells.items()}, "summary": summ, "identity": idres,
       "gate": {l: g["summary"] for l, g in gates.items()}, "gate_e2e_cells": gcells}
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
print(json.dumps(summ["long_64K_1M"]), json.dumps(summ["short_le_32K"]))
for k, v in cells.items():
    if k[2] <= 32768 or k[2] == 1048576: print(k, v["base"], v["candidate"], v["delta_pct"], v["verdict"])
print("identity", json.dumps(idres)[:2000])
print("gate", json.dumps(out["gate"]))
for k, v in gcells.items(): print("gate-e2e", k, v["base"], v["candidate"], v["delta_pct"], v["verdict"])
