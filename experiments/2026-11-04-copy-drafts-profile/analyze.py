#!/usr/bin/env python3
"""Normalize repeat-boot timings and validate the retained native IKET data."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
root = args.root / "copy-drafts-v1"


def interval(values):
    mean = statistics.fmean(values)
    half = 12.706204736 * statistics.stdev(values) / math.sqrt(2)
    return {"boots": values, "mean": mean, "ci95_t_df1": [mean - half, mean + half]}


by_case = defaultdict(list)
for boot in (1, 2):
    for case in json.loads((root / f"copy-bench-{boot}/cases.json").read_text()):
        key = case["requests"], case["context"], case["history"], case["pattern"]
        by_case[key].append(case)
timings = []
for key, pair in sorted(by_case.items()):
    assert len(pair) == 2
    timings.append(
        {
            "requests": key[0],
            "context": key[1],
            "history": key[2],
            "pattern": key[3],
            "offered": pair[0]["offered"],
            "timings_us": {
                name: interval([p["timings"][name]["median_us"] for p in pair])
                for name in ("match", "alloc", "select", "pipeline")
            },
        }
    )

native = []
for boot in (1, 2):
    folder = root / f"copy-native-{boot}/profile"
    path = next((folder / "trace").rglob("iket.decoded_results.json"))
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    del raw
    counts, phases, spans = (
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(list),
    )
    launches = data["launches"] + [
        x for group in data["graphLaunches"].values() for x in group
    ]
    invalid = 0
    for launch in launches:
        kernel = launch["kernelName"]
        counts[kernel]["launches"] += 1
        counts[kernel]["warp_lifetimes"] += len(launch.get("warpLifetimes", []))
        sms = set()
        for event in launch["ranges"]:
            invalid += event["endTs"] < event["startTs"]
            counts[kernel]["ranges"] += 1
            phases[kernel][data["stringTable"][event["rangeNameIdx"]]] += 1
            loc = data["locationTable"][event["warpLocIdxs"][0]]
            sms.add((loc["gpcId"], loc["tpcId"], loc["smId"]))
        if launch["gridDimX"] == 1 and kernel == "copy_match_kernel":
            spans[kernel].append(len(sms))
    assert not invalid
    assert set(counts) == {
        "copy_match_kernel",
        "copy_alloc_kernel",
        "copy_select_kernel",
    }
    assert all(c["ranges"] > 0 and c["warp_lifetimes"] > 0 for c in counts.values())
    native.append(
        {
            "boot": boot,
            "decoded_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": checksum,
            "counts": dict(counts),
            "phases": dict(phases),
            "single_request_profiled_sm_counts": dict(spans),
            "invalid_ranges": invalid,
            "capacity": json.loads((folder / "event-capacity.json").read_text()),
            "scope": "4K histories; instrumented timings are not performance measurements",
        }
    )
    del data, launches

verification = [
    json.loads(
        (args.root / "copy-verify-v1" / f"copy-verify-{i}/result.json").read_text()
    )
    for i in (1, 2)
]
assert all(not c["slower"] for run in verification for c in run["cases"])
result = {
    "timings": timings,
    "native": native,
    "verification": verification,
    "interval_method": "95% Student t on two independent boot medians (df=1); 21 cold-L2 graph samples per boot. Two boots give limited generalization.",
    "scope": "candidate-38 isolated kernels and proposer tests; no distributed real-model throughput claim",
}
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "cases": len(timings),
            "native": [{"boot": n["boot"], "counts": n["counts"]} for n in native],
        }
    )
)
