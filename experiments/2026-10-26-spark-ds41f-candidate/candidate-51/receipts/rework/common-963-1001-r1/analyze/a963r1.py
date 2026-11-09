# usage: python3 a963r1.py JSON_DIR OUT.json
# Round 1 of #963: reads pack12-{base,c51r1}-boot{1,2}.json (h963r1.py) and compares, per
# cell, the base arm's two boots with the corrected arm's two boots (interval rule in
# common.py), by graph replay (L2-hot), CUPTI kernel time (back to back and isolated) and
# eager. Adapted from the own-mark run's a963.py:
#  B/C at the 128-row tile: each arm's own TILE_CONFIGS[128] (conflates #1001) and the
#    same config on both arms (s4a2, s2: #963-r1 alone);
#  D: each screen-config set on both arms (s4a2 = base's SCREEN_TILE_CONFIGS[128],
#    r1 = #1001-r1's): the same set isolates #963-r1; own_vs_own is serving's pair.
import json
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
CA = "c51r1"
R = {(a, k): load(f"{d}/pack12-{a}-boot{k}.json") for a in ("base", CA) for k in (1, 2)}
B = [R[("base", 1)], R[("base", 2)]]
C = [R[(CA, 1)], R[(CA, 2)]]
out = {"errors": {f"{a}-boot{k}": r["errors"] for (a, k), r in R.items() if r["errors"]},
       "tile_configs_128": {f"{a}-boot{k}": r["tile_configs_128"] for (a, k), r in R.items()},
       "A_gemm": {}, "B_proj": {}, "C_greedy": {}, "D_screen": {}, "E_splitk": {}, "counts": {}}
MODES = {
    "graph": lambda c: c["graph"]["median_us"],
    "kernel": lambda c: c["kernel"]["per_call_us"],
    "kernel_isolated": lambda c: c["kernel_isolated_us"],
    "eager": lambda c: c["eager_us"],
}


def compare(section, bcells, ccells, hashkey, tag=""):
    row = {}
    for mode, get in MODES.items():
        b = [get(x) for x in bcells]
        c = [get(x) for x in ccells]
        row[f"{mode}_us"] = fmt(b, c)
        row[f"{mode}_ratio"] = ratio(b, c)
        v = verdict(b, c)
        out["counts"].setdefault(f"{section}{tag}", {}).setdefault(mode, {}).setdefault(v, 0)
        out["counts"][f"{section}{tag}"][mode][v] += 1
    row["hash_equal_all_runs"] = len({x[hashkey] for x in bcells + ccells}) == 1
    return row


def cfg_name(cfg):
    return "s4a2" if cfg.get("num_stages") == 4 and cfg.get("act_stages") == 2 else (
        "s2" if cfg.get("num_stages") == 2 and not cfg.get("act_stages") else str(cfg))


for name in C[0]["A_gemm"]:
    out["A_gemm"][name] = {}
    for m in [x for x in C[0]["A_gemm"][name] if x.isdigit()]:
        out["A_gemm"][name][m] = compare(f"A_gemm:{name}", [b["A_gemm"][name][m] for b in B],
                                         [c["A_gemm"][name][m] for c in C], "out_hash")

for sec, hk in (("B_proj", "out_hash"), ("C_greedy", "pairs_hash")):
    for name in C[0][sec]:
        out[sec][name] = {}
        vb = B[0]["B_proj"][name]["variants"]
        vc = C[0]["B_proj"][name]["variants"]
        for m in [x for x in C[0][sec][name] if x.isdigit()]:
            bm = str(C[0][sec][name][m]["block_M"])
            res = {"own_vs_own": compare(sec, [b[sec][name][m]["own"] for b in B],
                                         [c[sec][name][m]["own"] for c in C], hk, "_own")}
            if bm == "128":
                for cfg in ("s4a2", "s2"):
                    bv = next(v for v, c in vb["128"].items() if cfg_name(c) == cfg)
                    cv = next(v for v, c in vc["128"].items() if cfg_name(c) == cfg)
                    res[f"same_cfg_{cfg}"] = compare(sec, [b[sec][name][m][bv] for b in B],
                                                     [c[sec][name][m][cv] for c in C], hk, f"_tile128_{cfg}")
            else:
                out["counts"].setdefault(f"{sec}_same_cfg_all_tiles", {})
            res["block_M"] = bm
            out[sec][name][m] = res

for name in C[0]["D_screen"]:
    out["D_screen"][name] = {"own_set": {"base": B[0]["D_screen"][name]["own_set"], CA: C[0]["D_screen"][name]["own_set"]}}
    ob, oc = B[0]["D_screen"][name]["own_set"][0], C[0]["D_screen"][name]["own_set"][0]
    for m in [x for x in C[0]["D_screen"][name] if x.isdigit()]:
        res = {"block_M": C[0]["D_screen"][name][m]["block_M"]}
        for sname in ("s4a2", "r1"):
            res[f"same_set_{sname}"] = compare("D_screen", [b["D_screen"][name][m][sname] for b in B],
                                               [c["D_screen"][name][m][sname] for c in C], "pairs_hash", f"_same_set_{sname}")
        res["own_vs_own"] = compare("D_screen", [b["D_screen"][name][m][ob] for b in B],
                                    [c["D_screen"][name][m][oc] for c in C], "pairs_hash", "_own")
        for arm, runs in (("base", B), (CA, C)):
            res[f"1001r1_within_{arm}"] = compare("D_screen", [r["D_screen"][name][m]["s4a2"] for r in runs],
                                                  [r["D_screen"][name][m]["r1"] for r in runs], "pairs_hash", f"_1001r1_within_{arm}")
        out["D_screen"][name][m] = res

for name in C[0]["E_splitk"]:
    out["E_splitk"][name] = {}
    for m in [x for x in C[0]["E_splitk"][name] if x.isdigit()]:
        out["E_splitk"][name][m] = compare("E_splitk", [b["E_splitk"][name][m] for b in B],
                                           [c["E_splitk"][name][m] for c in C], "out_hash")

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for sec in ("A_gemm", "B_proj", "C_greedy", "D_screen", "E_splitk"):
    print("==", sec)
    for name, rows in out[sec].items():
        print(" ", name)
        for m, row in rows.items():
            if not m.isdigit():
                continue
            if "graph_us" in row:
                print(f"    M={m:>5} graph {row['graph_us']} | kiso {row['kernel_isolated_us']} | eager {row['eager_us']} | hash= {row['hash_equal_all_runs']}")
                continue
            for k, r in row.items():
                if isinstance(r, dict):
                    print(f"    M={m:>5} {k:22s} graph {r['graph_us']} | kiso {r['kernel_isolated_us']} | hash= {r['hash_equal_all_runs']}")
print(json.dumps(out["counts"], indent=1))
print("errors:", json.dumps(out["errors"])[:2000])
