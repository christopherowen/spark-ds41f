# usage: python3 a963.py JSON_DIR OUT.json
# Reads pack12-{base,c51}-boot{1,2}.json (h963.py): per cell, the base arm's two boots
# against the candidate arm's two boots, by graph replay (L2-hot), kernel time and eager.
# At the 128-row vocabulary tile the same TILE_CONFIGS[128] variant is compared across
# arms (s4a2 = base's, s2 = #1001's), which isolates #963 from #1001.
import json
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
R = {(a, k): load(f"{d}/pack12-{a}-boot{k}.json") for a in ("base", "c51") for k in (1, 2)}
B = [R[("base", 1)], R[("base", 2)]]
C = [R[("c51", 1)], R[("c51", 2)]]
out = {"errors": {f"{a}-boot{k}": r["errors"] for (a, k), r in R.items() if r["errors"]},
       "A_gemm": {}, "B_proj": {}, "C_greedy": {}, "D_screen": {}, "E_splitk": {}, "counts": {}}
MODES = {
    "graph": lambda c: c["graph"]["median_us"],
    "kernel": lambda c: c["kernel"]["per_call_us"],
    "kernel_isolated": lambda c: c["kernel_isolated_us"],
    "eager": lambda c: c["eager_us"],
}


def compare(section, name, m, bcells, ccells, hashkey, tag=""):
    row = {}
    for mode, get in MODES.items():
        b = [get(x) for x in bcells]
        c = [get(x) for x in ccells]
        row[f"{mode}_us"] = fmt(b, c)
        row[f"{mode}_ratio"] = ratio(b, c)
        v = verdict(b, c)
        out["counts"].setdefault(f"{section}{tag}", {}).setdefault(mode, {}).setdefault(v, 0)
        out["counts"][f"{section}{tag}"][mode][v] += 1
    hs = {x[hashkey] for x in bcells + ccells}
    row["hash_equal_all_runs"] = len(hs) == 1
    return row


def cfg_name(cfg):
    return "s4a2" if cfg.get("num_stages") == 4 and cfg.get("act_stages") == 2 else (
        "s2" if cfg.get("num_stages") == 2 and not cfg.get("act_stages") else str(cfg))


for name in C[0]["A_gemm"]:
    out["A_gemm"][name] = {}
    for m in [x for x in C[0]["A_gemm"][name] if x.isdigit()]:
        out["A_gemm"][name][m] = compare("A_gemm", name, m, [b["A_gemm"][name][m] for b in B],
                                         [c["A_gemm"][name][m] for c in C], "out_hash")

for sec, hk in (("B_proj", "out_hash"), ("C_greedy", "pairs_hash")):
    for name in C[0][sec]:
        out[sec][name] = {}
        vb = B[0]["B_proj"][name]["variants"]
        vc = C[0]["B_proj"][name]["variants"]
        for m in [x for x in C[0][sec][name] if x.isdigit()]:
            bm = str(C[0][sec][name][m]["block_M"])
            res = {}
            # each arm's own config (serving): conflates #1001 at the 128-row tile
            res["own_vs_own"] = compare(sec, name, m, [b[sec][name][m]["own"] for b in B],
                                        [c[sec][name][m]["own"] for c in C], hk, "_own")
            if bm == "128":
                for cfg in ("s4a2", "s2"):
                    bv = next(v for v, c in vb["128"].items() if cfg_name(c) == cfg)
                    cv = next(v for v, c in vc["128"].items() if cfg_name(c) == cfg)
                    res[f"same_cfg_{cfg}"] = compare(sec, name, m, [b[sec][name][m][bv] for b in B],
                                                     [c[sec][name][m][cv] for c in C], hk, f"_tile128_{cfg}")
                # #1001 within each arm (same process): s4a2 -> s2
                for arm, runs, vmap in (("base", B, vb), ("c51", C, vc)):
                    v42 = next(v for v, c in vmap["128"].items() if cfg_name(c) == "s4a2")
                    v2 = next(v for v, c in vmap["128"].items() if cfg_name(c) == "s2")
                    res[f"1001_within_{arm}"] = compare(sec, name, m, [r[sec][name][m][v42] for r in runs],
                                                        [r[sec][name][m][v2] for r in runs], hk, f"_1001_within_{arm}")
            res["block_M"] = bm
            out[sec][name][m] = res

for sec, hk in (("D_screen", "pairs_hash"), ("E_splitk", "out_hash")):
    for name in C[0][sec]:
        out[sec][name] = {}
        for m in [x for x in C[0][sec][name] if x.isdigit()]:
            out[sec][name][m] = compare(sec, name, m, [b[sec][name][m] for b in B],
                                        [c[sec][name][m] for c in C], hk)

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for sec in ("A_gemm", "B_proj", "C_greedy", "D_screen", "E_splitk"):
    print("==", sec)
    for name, rows in out[sec].items():
        print(" ", name)
        for m, row in rows.items():
            if "own_vs_own" in row:
                for k, r in row.items():
                    if k == "block_M":
                        continue
                    print(f"    M={m:>5} {k:18s} graph {r['graph_us']} | kernel {r['kernel_us']} | kiso {r['kernel_isolated_us']} | eager {r['eager_us']} | hash= {r['hash_equal_all_runs']}")
            else:
                print(f"    M={m:>5} graph {row['graph_us']} | kernel {row['kernel_us']} | kiso {row['kernel_isolated_us']} | eager {row['eager_us']} | hash= {row['hash_equal_all_runs']}")
print(json.dumps(out["counts"], indent=1))
print("errors:", json.dumps(out["errors"])[:2000])
