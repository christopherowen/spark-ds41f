#!/usr/bin/env python3
"""Compare the candidate boots with the retained candidate-38 measurements."""

import argparse
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("receipts", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
old = json.loads(
    (here.parent / "2026-11-04-copy-drafts-profile/results.json").read_text()
)
receipts = json.loads(args.receipts.read_text())
run = receipts["runs"]["copy-drafts-fix-v2"]


def key(case):
    return tuple(case[k] for k in ("requests", "context", "history", "pattern"))


def interval(values):
    assert len(values) == 2
    mean = statistics.fmean(values)
    half = 12.706204736 * statistics.stdev(values) / math.sqrt(2)
    return dict(boots=values, mean=mean, ci95_t_df1=[mean - half, mean + half])


reference = {key(c): c for c in old["timings"]}
variants, tests = {}, {}
for variant in ("selection", "search"):
    boots = []
    for boot in (1, 2):
        test = run["jobs"][f"{variant}-tests-{boot}"]["result"]
        assert test["exit"] == 0 and test["result"]["pytest_exit"] == 0
        assert test["result"]["outcomes"] == {"passed": 40}
        tests[f"{variant}-{boot}"] = test["result"]["outcomes"]
        row = run["jobs"][f"{variant}-bench-{boot}"]["result"]
        assert row["exit"] == 0 and len(row["result"]["cases"]) == 96
        boots.append({key(c): c for c in row["result"]["cases"]})
    cases = []
    for identity, original in sorted(reference.items()):
        pair = [b[identity] for b in boots]
        assert pair[0]["offered"] == pair[1]["offered"]
        timings = {
            name: interval([c["timings"][name]["median_us"] for c in pair])
            for name in ("match", "alloc", "select", "pipeline")
        }
        cases.append(
            dict(
                zip(("requests", "context", "history", "pattern"), identity),
                offered=pair[0]["offered"],
                timings_us=timings,
                original=original,
                pipeline_change_percent=100
                * (
                    timings["pipeline"]["mean"]
                    / original["timings_us"]["pipeline"]["mean"]
                    - 1
                ),
            )
        )
    variants[variant] = cases
for first, second in zip(variants["selection"], variants["search"]):
    assert key(first) == key(second) and first["offered"] == second["offered"]
    second["change_from_selection_percent"] = 100 * (
        second["timings_us"]["pipeline"]["mean"]
        / first["timings_us"]["pipeline"]["mean"]
        - 1
    )
result = dict(
    tests=tests,
    variants=variants,
    interval_method="95% Student t on two boot medians (df=1), 21 cold-L2 graph samples per boot; limited generalization. Percentage changes are ratios of means, without a significance claim.",
    scope="candidate-38 kernel tests with verified source overlays; no real-model acceptance or distributed throughput qualification",
)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"tests": tests, "cases_per_variant": len(variants["search"])}))
