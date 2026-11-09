# usage: python3 a987.py JSON_DIR OUT.json
# Reads wob-{base,c51}-boot{1,2}.json (h987.py) and writes the per-cell comparison.
import sys

from common import fmt, load, ratio, verdict

d = sys.argv[1]
runs = {(a, k): load(f"{d}/wob-{a}-boot{k}.json") for a in ("base", "c51") for k in (1, 2)}
out = {"cells": {}, "summary": {}, "errors": {f"{a}-boot{k}": r["errors"] for (a, k), r in runs.items() if r["errors"]}}
metrics = [
    ("wob", "graph_hot", lambda c: c["wob"]["graph_hot"]["median_us"]),
    ("wob", "graph_cold", lambda c: c["wob"]["graph_cold"]["median_us"]),
    ("wob", "kernel", lambda c: c["wob"]["kernel"]["per_call_us"]),
    ("wob", "eager", lambda c: c["wob"]["eager_us"]),
    ("e2e", "graph_hot", lambda c: c["e2e"]["graph_hot"]["median_us"]),
    ("e2e", "graph_cold", lambda c: c["e2e"]["graph_cold"]["median_us"]),
    ("e2e", "kernel", lambda c: c["e2e"]["kernel"]["per_call_us"]),
    ("e2e", "eager", lambda c: c["e2e"]["eager_us"]),
]
counts = {}
for key in runs[("c51", 1)]["cells"]:
    out["cells"][key] = {}
    for m in [x for x in runs[("c51", 1)]["cells"][key] if x.isdigit()]:
        cells = {ak: runs[ak]["cells"][key].get(m) for ak in runs}
        if any(v is None for v in cells.values()):
            out["cells"][key][m] = "missing in a run"
            continue
        row = {
            "decode_tile_rows base/c51": f"{cells[('base', 1)]['decode_tile_rows']}/{cells[('c51', 1)]['decode_tile_rows']}",
            "wob_bit_exact_vs_prefill": {f"{a}-boot{k}": cells[(a, k)]["wob_bit_exact_vs_prefill"] for (a, k) in cells},
            "e2e_bit_exact_vs_prefill": {f"{a}-boot{k}": cells[(a, k)]["e2e_bit_exact_vs_prefill"] for (a, k) in cells},
            "wob_hash_equal_all_runs": len({cells[ak]["wob_out_hash"] for ak in cells}) == 1,
            "e2e_hash_equal_all_runs": len({cells[ak]["e2e_out_hash"] for ak in cells}) == 1,
        }
        for part, mode, get in metrics:
            b = [get(cells[("base", 1)]), get(cells[("base", 2)])]
            c = [get(cells[("c51", 1)]), get(cells[("c51", 2)])]
            row[f"{part}_{mode}_us"] = fmt(b, c)
            row[f"{part}_{mode}_ratio"] = ratio(b, c)
            v = verdict(b, c)
            band = "97..128" if int(m) > 96 else "<=96"
            counts.setdefault(f"{part}_{mode}", {}).setdefault(band, {}).setdefault(v, 0)
            counts[f"{part}_{mode}"][band][v] += 1
        out["cells"][key][m] = row
out["summary"]["verdict_counts_by_metric_and_rows"] = counts
sys.path  # noqa
import json  # noqa: E402

with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=1)
for key, rows in out["cells"].items():
    print(key)
    for m, row in rows.items():
        if isinstance(row, str):
            print(" ", m, row)
            continue
        print(f"  M={m} tile {row['decode_tile_rows base/c51']} exact(c51) {all(v for k, v in row['wob_bit_exact_vs_prefill'].items() if k.startswith('c51'))} hash= {row['wob_hash_equal_all_runs']}/{row['e2e_hash_equal_all_runs']}")
        for part, mode, _ in metrics:
            print(f"     {part:3s} {mode:10s} {row[f'{part}_{mode}_us']}")
print(json.dumps(counts, indent=1))
