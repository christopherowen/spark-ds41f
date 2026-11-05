#!/usr/bin/env python3
"""Recompute both timing gates from raw evidence, including rejected cases."""

import argparse
import gzip
import json
import math
from pathlib import Path

from paired import classify

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


boots = {}
for folder in sorted(args.run.glob("*-pair-*")):
    path = folder / "result.json"
    if not path.exists():
        path = path.with_suffix(".json.gz")
    data = read(path)
    rows = {}
    for row in data["cases"]:
        logs = []
        for record in row["measured"]["pairs"]:
            averages = {}
            for role, batches in record["times_us"].items():
                values = [t for batch in batches for t in batch]
                assert len(values) == 16 and all(math.isfinite(t) and t > 0 for t in values)
                averages[role] = sum(values) / len(values)
            logs.append(math.log(averages["candidate"] / averages["reference"]))
        expected_logs = row["measured"]["log_ratios"]["candidate"]
        assert len(logs) == len(expected_logs)
        assert all(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13)
                   for a, b in zip(logs, expected_logs, strict=True))
        gate = classify(logs, row["identical_program"])
        assert gate["status"] == row["gate"]["status"]
        assert math.isclose(gate["upper_ratio"], row["gate"]["upper_ratio"], rel_tol=1e-12)
        assert not row["profiles"]["candidate"]["aten_kernels"], row["name"]
        rows[row["name"]] = dict(gate=gate, gpu_launches=len(row["profiles"]["candidate"]["kernels"]),
                                 previous_launches=len(row["profiles"]["previous"]["kernels"]),
                                 reference_launches=len(row["profiles"]["reference"]["kernels"]))
    boots[folder.name] = rows
assert len(boots) == 2
first, second = boots.values()
assert first.keys() == second.keys()
unresolved = {name: [first[name]["gate"], second[name]["gate"]] for name in first
              if any(boot[name]["gate"]["status"] == "not_proved" for boot in (first, second))}
result = dict(cases=len(first), boots=boots, unresolved=unresolved, passed=not unresolved,
              gate="Each boot must show an upper ratio <=1 at five standard errors, or exact GPU program/binding equality; no slowdown tolerance.")
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(cases=len(first), passed=not unresolved, unresolved=list(unresolved)), indent=2))
