# usage: python3 adiag.py JSON_DIR SECTION [REF]
# d963.py diagnosis: per cell, each variant's graph-replay medians (2 in-process samples per
# boot, up to 2 boots) and its ratio to REF (default: the first variant); a variant is "faster"
# or "slower" when all four of its samples are beyond all four of REF's.
import json
import os
import sys

d, sec = sys.argv[1], sys.argv[2]
runs = [json.load(open(f"{d}/d963-{sec}-boot{k}.json")) for k in (1, 2) if os.path.exists(f"{d}/d963-{sec}-boot{k}.json")]
print(f"boots: {len(runs)}")
for r in runs:
    if r["errors"]:
        print("ERRORS", [e["where"] for e in r["errors"]])
for key in runs[0]["cells"]:
    c0 = runs[0]["cells"][key]
    if "hashes" not in c0:
        print(key, c0)
        continue
    vs = list(c0["hashes"])
    ref = sys.argv[3] if len(sys.argv) > 3 else vs[0]
    s = {v: sum((r["cells"][key][v]["graph_us"] for r in runs if key in r["cells"]), []) for v in vs}
    ki = {v: [r["cells"][key][v]["kernel_isolated_us"] for r in runs if key in r["cells"]] for v in vs}
    eq = all(r["cells"][key]["bit_equal"] for r in runs if key in r["cells"]) and len({h for r in runs if key in r["cells"] for h in r["cells"][key]["hashes"].values()}) == 1
    parts = []
    for v in vs:
        m = sum(s[v]) / len(s[v])
        mr = sum(s[ref]) / len(s[ref])
        tag = ""
        if v != ref:
            tag = "+" if max(s[v]) < min(s[ref]) else ("-" if min(s[v]) > max(s[ref]) else "=")
        parts.append(f"{v}={m:9.2f}({mr / m:.3f}{tag}) ki={'/'.join(f'{x:.1f}' for x in ki[v])}")
    print(f"{key:42s} eq={eq} " + "  ".join(parts))
