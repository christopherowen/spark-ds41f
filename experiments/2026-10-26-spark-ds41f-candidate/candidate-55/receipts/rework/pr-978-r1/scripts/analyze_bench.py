#!/usr/bin/env python3
"""Summarize the own-mark session: bench-<arm>-boot<k>.json and identity-<arm>.json.

Usage: analyze_bench.py <receipts dir> <out.json>
Per workload and arm: turn-1 and turn-2+ medians of each boot, the payload bytes,
and the verdicts: r1 faster than base beyond the interval (r1's slower boot faster
than base's faster boot), and r1 against c55 (the boots' ranges overlap or not).
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = {"base": "base", "r1": "candidate-55-r1", "c55": "candidate-55"}
data = {a: [json.loads((d / f"bench-{a}-boot{k}.json").read_text()) for k in (1, 2)] for a in arms}
out = {"workloads": {}, "summary": {}}
names = list(data["base"][0]["workloads"])
for w in names:
    row = {}
    for a in arms:
        boots = [b["workloads"][w] for b in data[a]]
        row[a] = {
            "turn1_median_ms": [x["turn1_Cold"]["total_median_ms"] for x in boots],
            "turn2plus_median_ms": [x["turn2plus_Warm"]["total_median_ms"] for x in boots],
            "turn2plus_stage_median_ms": [
                {k: x["turn2plus_Warm"][k] for k in ("load_base64_median_ms", "processor_apply_median_ms", "ipc_encode_median_ms")}
                for x in boots],
            "ipc_bytes_turn1": [x["ipc_bytes_turn1"] for x in boots],
            "ipc_bytes_warm": [x["ipc_bytes_warm"] for x in boots],
            "prompt_tokens_turn1": [x["prompt_tokens_turn1"] for x in boots],
        }
    b, r, c = (row[a]["turn2plus_median_ms"] for a in ("base", "r1", "c55"))
    row["r1_vs_base"] = {
        "speedup_of_means": round((sum(b) / 2) / (sum(r) / 2), 2),
        "faster_beyond_interval": max(r) < min(b),
        "ipc_bytes_warm": f"{row['base']['ipc_bytes_warm'][0]} -> {row['r1']['ipc_bytes_warm'][0]}",
    }
    row["r1_vs_c55"] = {
        "ratio_of_means": round((sum(r) / 2) / (sum(c) / 2), 3),
        "ranges_overlap": not (max(r) < min(c) or max(c) < min(r)),
        "r1_slower_beyond_interval": min(r) > max(c),
        "same_ipc_bytes_warm": row["r1"]["ipc_bytes_warm"] == row["c55"]["ipc_bytes_warm"],
    }
    out["workloads"][w] = row
out["summary"] = {
    "r1_faster_than_base_beyond_interval": sum(v["r1_vs_base"]["faster_beyond_interval"] for v in out["workloads"].values()),
    "r1_slower_than_c55_beyond_interval": sum(v["r1_vs_c55"]["r1_slower_beyond_interval"] for v in out["workloads"].values()),
    "workloads": len(names),
}
ib = json.loads((d / "identity-base.json").read_text())["workloads"]
ir = json.loads((d / "identity-r1.json").read_text())["workloads"]
same = {w: ib[w] == ir[w] for w in ib}
out["identity"] = {"workloads_identical": same, "all_identical": all(same.values()),
                   "turns_compared": sum(len(v) for v in ib.values())}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(f"{'workload':24} {'base t2+ ms':>16} {'r1 t2+ ms':>14} {'c55 t2+ ms':>14} {'x':>6} beyond  r1/c55  bytes")
for w, v in out["workloads"].items():
    f = lambda a: "/".join(f"{x:.2f}" for x in v[a]["turn2plus_median_ms"])
    print(f"{w:24} {f('base'):>16} {f('r1'):>14} {f('c55'):>14} {v['r1_vs_base']['speedup_of_means']:>6} "
          f"{str(v['r1_vs_base']['faster_beyond_interval']):6} {v['r1_vs_c55']['ratio_of_means']:>6}  {v['r1_vs_base']['ipc_bytes_warm']}")
print(json.dumps(out["summary"]), "identity:", out["identity"]["all_identical"])
