#!/usr/bin/env python3
"""Normalize a portable native receipt without rerunning hardware."""

import argparse
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
parser.add_argument("--out", type=Path, default=HERE / "results.json")
args = parser.parse_args()
receipt = json.loads(args.receipt.read_text())
current = json.loads((HERE / "candidate.json").read_text())
base = json.loads((HERE / "base.json").read_text())
expected = "gather_logits"
jobs = receipt["jobs"]
assert all(
    r["exit"] == 0 and r["result"]["candidate"]["tree"] == current["tree"] for r in jobs
)
tests = [r for r in jobs if r["job"]["operation"] == "tests"]
bench = [r["result"] for r in jobs if r["job"]["operation"] == "bench"]
iket = [r["result"] for r in jobs if r["job"]["operation"] == "iket"]
assert len(tests) == len(bench) == len(iket) == 2
assert all(len(i["launches"]) == 12 for i in iket)
assert all(set(r["result"]["outcomes"]) == {"passed"} for r in tests)


def interval(values):
    mean = statistics.fmean(values)
    half = 12.706204736 * statistics.stdev(values) / math.sqrt(2)
    return dict(boots=values, mean=mean, ci95_t_df1=[mean - half, mean + half])


cases = {}
for boot in bench:
    for profile in boot["launch_inventory"].values():
        inventory = profile["candidate"]
        assert len(inventory["events"]) == 1 and expected in inventory["events"][0]
        assert not inventory["aten_kernels"]
    for case in boot["cases"]:
        for method in case["timings"].values():
            assert len(method["samples_us"]) == 21
            assert statistics.median(method["samples_us"]) == method["median_us"]
        cases.setdefault(json.dumps(case["case"], sort_keys=True), []).append(case)
rows = []
for key, pair in cases.items():
    assert len(pair) == 2
    stats = {
        method: interval([p["timings"][method]["median_us"] for p in pair])
        for method in ("reference", "candidate")
    }
    rows.append(
        dict(
            case=json.loads(key),
            timings_us=stats,
            change_percent=100
            * (stats["candidate"]["mean"] / stats["reference"]["mean"] - 1),
        )
    )
rows.sort(key=lambda r: json.dumps(r["case"], sort_keys=True))
result = dict(
    source_head=current["head"],
    source_tree=current["tree"],
    native_source=receipt["identity"]["commit"],
    native_root=receipt["native_root"],
    tests=[r["result"]["outcomes"] for r in tests],
    timings=rows,
    inventories=[b["launch_inventory"] for b in bench],
    iket=[i["launches"] for i in iket],
    minimum_host_available_bytes=min(m["available_bytes"] for m in receipt["memory"]),
    peak_allocated_bytes=max(
        v["peak_allocated_bytes"]
        for b in bench
        for c in b["cases"]
        for v in c["timings"].values()
    ),
    interval_method="95% Student t on two boot medians, df=1; 21 cold-L2 graph samples per method per boot. Ratios of means are descriptive and do not establish significance.",
    scope="isolated kernels and runner components; full-image and full-model release qualification remain",
)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(tests=result["tests"], cases=len(rows))))
