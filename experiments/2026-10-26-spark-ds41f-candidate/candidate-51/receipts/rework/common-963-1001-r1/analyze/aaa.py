# usage: python3 aaa.py JSON_DIR OUT.json
# The A/A rerun of #963-r1's identical-code sections: reads aa-DE-{base,c51r1}-boot{1,2}.json
# (h963aa.py, H963_SECTIONS=DE) and compares, per cell, the base arm's two boots with the
# corrected arm's two boots by the interval rule (common.py). Section D (certified screen,
# both screen-config sets) and E (split-K compressor GEMM) run byte-identical SASS in both arms
# (out/pr-common/sass-{base,c51r1}.txt), so every non-"same" verdict here is the rule's noise.
import json
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
CA = "c51r1"
R = {(a, k): load(f"{d}/aa-DE-{a}-boot{k}.json") for a in ("base", CA) for k in (1, 2)}
B = [R[("base", 1)], R[("base", 2)]]
C = [R[(CA, 1)], R[(CA, 2)]]
out = {"errors": {f"{a}-boot{k}": r["errors"] for (a, k), r in R.items() if r["errors"]},
       "D_screen": {}, "E_splitk": {}, "counts": {}}
get = lambda c: c["graph"]["median_us"]  # noqa: E731


def cmp(tag, bc, cc, hk):
    b, c = [get(x) for x in bc], [get(x) for x in cc]
    v = verdict(b, c)
    out["counts"].setdefault(tag, {}).setdefault(v, 0)
    out["counts"][tag][v] += 1
    return {"graph_us": fmt(b, c), "graph_ratio": ratio(b, c),
            "kernel_isolated_us": fmt([x["kernel_isolated_us"] for x in bc], [x["kernel_isolated_us"] for x in cc]),
            "hash_equal_all_runs": len({x[hk] for x in bc + cc}) == 1}


for name in C[0]["D_screen"]:
    out["D_screen"][name] = {}
    for m in [x for x in C[0]["D_screen"][name] if x.isdigit()]:
        out["D_screen"][name][m] = {s: cmp(f"D_screen_same_set_{s}", [b["D_screen"][name][m][s] for b in B],
                                             [c["D_screen"][name][m][s] for c in C], "pairs_hash") for s in ("s4a2", "r1")}
for name in C[0]["E_splitk"]:
    out["E_splitk"][name] = {}
    for m in [x for x in C[0]["E_splitk"][name] if x.isdigit()]:
        out["E_splitk"][name][m] = cmp("E_splitk", [b["E_splitk"][name][m] for b in B],
                                       [c["E_splitk"][name][m] for c in C], "out_hash")
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for sec in ("D_screen", "E_splitk"):
    print("==", sec)
    for name, rows in out[sec].items():
        print(" ", name)
        for m, row in rows.items():
            if "graph_us" in row:
                print(f"    M={m:>4} graph {row['graph_us']} | hash= {row['hash_equal_all_runs']}")
            else:
                for s, r in row.items():
                    print(f"    M={m:>4} {s:5s} graph {r['graph_us']} | hash= {r['hash_equal_all_runs']}")
print(json.dumps(out["counts"], indent=1))
print("errors:", json.dumps(out["errors"])[:2000])
