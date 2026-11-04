#!/usr/bin/env python3
"""Verify each boot's strict GPU gate from its retained paired observations."""

import argparse
from collections import Counter
import json
import math
import statistics
from pathlib import Path

from paired import classify

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
parser.add_argument("--candidate", type=Path, default=HERE / "candidate.json")
parser.add_argument("--out", type=Path, default=HERE / "paired-results.json")
parser.add_argument("--rounds", type=int, default=120)
args = parser.parse_args()
r = json.loads(args.receipt.read_text())
c = json.loads(args.candidate.read_text())
assert all(
    j["exit"] == 0 and not j["memory_guard"] and j["result"]["candidate"] == c
    for j in r["jobs"]
)
assert r["closure"]["reservation_released"] and not r["completed"]["failed"]
jobs = {
    op: [j["result"] for j in r["jobs"] if j["job"]["operation"] == op]
    for op in ("tests", "pair", "iket")
}
assert all(len(v) == 2 for v in jobs.values())
assert all(set(t["outcomes"]) == {"passed"} for t in jobs["tests"])
assert all(len(i["launches"]) == 40 for i in jobs["iket"])
combined = {}
for boot, data in enumerate(jobs["pair"], 1):
    assert len(data["cases"]) == 120
    for row in data["cases"]:
        records = row["measured"]["pairs"]
        assert len(records) == args.rounds
        assert all(
            Counter(p["order"])
            == {k: 2 for k in ("reference", "candidate", "previous", "null")}
            for p in records
        )
        for p in records:
            for samples in p["times_us"].values():
                assert len(samples) == 2
                if isinstance(samples[0], list):
                    assert all(len(batch) == 8 for batch in samples)
                    samples = [t for batch in samples for t in batch]
                assert all(math.isfinite(t) and t > 0 for t in samples)
        logs = {role: [] for role in ("candidate", "previous", "null")}
        for record in records:

            def average(role):
                samples = record["times_us"][role]
                if isinstance(samples[0], list):
                    samples = [t for batch in samples for t in batch]
                return statistics.fmean(samples)

            reference = average("reference")
            for role in logs:
                logs[role].append(math.log(average(role) / reference))
        for role, values in logs.items():
            assert all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
                for a, b in zip(
                    values, row["measured"]["log_ratios"][role], strict=True
                )
            )
        same = (
            row["data"]["reference_preparation_aliases_input"]
            and row["data"]["shared_output_address"]
            and row["cuda_source_sha256"]["candidate"]
            == row["cuda_source_sha256"]["reference"]
        )
        assert same == row["same_program"]
        stats = classify(logs["candidate"], identical_program=same)
        assert stats.keys() == row["gate"].keys()
        for k, value in stats.items():
            if isinstance(value, float):
                # ARM and x86 libm may differ by an ULP in exp(); this only
                # checks the receipt, never relaxes the acceptance boundary.
                assert math.isclose(value, row["gate"][k], rel_tol=1e-14)
            else:
                assert value == row["gate"][k]
        key = json.dumps(row["case"], sort_keys=True)
        combined.setdefault(key, []).append(
            dict(
                boot=boot,
                data=row["data"],
                gate=stats,
                null=row["null_control"],
                previous=row["previous_gate"],
            )
        )
assert len(combined) == 120
rows = []
for key, pair in sorted(combined.items()):
    assert len(pair) == 2 and pair[0]["data"] == pair[1]["data"]
    rows.append(
        dict(
            case=json.loads(key),
            boots=pair,
            passed=all(p["gate"]["status"] != "not_proved" for p in pair),
        )
    )
failed = [row["case"] for row in rows if not row["passed"]]
result = dict(
    source_head=c["head"],
    source_tree=c["tree"],
    native_source=r["identity"]["commit"],
    native_root=r["native_root"],
    gate_passed=not failed,
    cases=rows,
    not_proved=failed,
    tests=[j["outcomes"] for j in jobs["tests"]],
    iket_launches=[len(j["launches"]) for j in jobs["iket"]],
    minimum_host_available_bytes=min(m["available_bytes"] for m in r["memory"]),
    rule="Every case on both boots: identical generated CUDA and aliased reference preparation with shared output, or a paired five-SE upper ratio at most 1; zero time tolerance; inconclusive is not a pass",
    scope="GPU operation cases; full-model and host enqueue qualification remain",
)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        dict(passed=len(rows) - len(failed), not_proved=len(failed), cases=len(rows))
    )
)
raise SystemExit(bool(failed))
