#!/usr/bin/env python3
"""Audit every observation and enforce the strict GPU performance gate."""

import argparse
import json
from pathlib import Path

from paired import audit, extend

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
parser.add_argument("--candidate", type=Path, default=HERE / "candidate.json")
parser.add_argument("--out", type=Path, default=HERE / "paired-results.json")
parser.add_argument("--rounds", type=int, default=120)
parser.add_argument("--extension", type=Path)
args = parser.parse_args()
candidate = json.loads(args.candidate.read_text())


def read(path, complete):
    receipt = json.loads(path.read_text())
    jobs = receipt["jobs"]
    assert all(
        j["exit"] == 0
        and not j["memory_guard"]
        and j["result"]["candidate"] == candidate
        for j in jobs
    )
    assert receipt["closure"]["reservation_released"]
    assert not receipt["completed"]["failed"]
    counts = {
        op: sum(j["job"]["operation"] == op for j in jobs)
        for op in ("tests", "pair", "iket")
    }
    assert counts == (
        dict(tests=2, pair=2, iket=2) if complete else dict(tests=0, pair=2, iket=0)
    )
    return receipt


primary = read(args.receipt, complete=True)
jobs = {
    op: [j["result"] for j in primary["jobs"] if j["job"]["operation"] == op]
    for op in ("tests", "pair", "iket")
}
assert all(set(t["outcomes"]) == {"passed"} for t in jobs["tests"])
assert all(len(i["launches"]) == 40 for i in jobs["iket"])
combined = {}
for boot, data in enumerate(jobs["pair"], 1):
    assert len(data["cases"]) == 120
    for row in data["cases"]:
        logs, stats = audit(row, args.rounds)
        key = json.dumps(row["case"], sort_keys=True)
        combined.setdefault(key, []).append(
            dict(
                boot=boot,
                data=row["data"],
                gate=stats,
                qualified=stats["status"] != "not_proved",
                null=row["null_control"],
                previous=row["previous_gate"],
                programs=row["cuda_source_sha256"],
                observations=logs["candidate"],
            )
        )
assert len(combined) == 120
for pair in combined.values():
    assert len(pair) == 2 and pair[0]["data"] == pair[1]["data"]

extension = None
if args.extension:
    targets = {
        key for key, pair in combined.items() if not all(p["qualified"] for p in pair)
    }
    assert targets, "no unresolved cases to extend"
    # An observed increase calls for a code fix, not averaging it away.
    assert all(p["gate"]["mean_ratio"] <= 1 for key in targets for p in combined[key])
    extension = read(args.extension, complete=False)
    for boot, job in enumerate(extension["jobs"]):
        cases = job["result"]["cases"]
        assert {json.dumps(r["case"], sort_keys=True) for r in cases} == targets
        assert len(cases) == len(targets)
        for row in cases:
            logs, stats = audit(row, 480)
            key = json.dumps(row["case"], sort_keys=True)
            old = combined[key][boot]
            assert old["data"] == row["data"]
            assert old["programs"] == row["cuda_source_sha256"]
            qualification = extend(old["observations"], logs["candidate"])
            old.update(
                primary_gate=old["gate"],
                extension_gate=stats,
                gate=qualification["pooled"],
                qualified=qualification["passed"],
            )

rows = []
for key, pair in sorted(combined.items()):
    for boot in pair:
        del boot["observations"]
    rows.append(
        dict(case=json.loads(key), boots=pair, passed=all(p["qualified"] for p in pair))
    )
failed = [row["case"] for row in rows if not row["passed"]]
receipts = [primary] + ([extension] if extension else [])
result = dict(
    source_head=candidate["head"],
    source_tree=candidate["tree"],
    native_source=primary["identity"]["commit"],
    native_root=primary["native_root"],
    extension_source=extension["identity"]["commit"] if extension else None,
    extension_root=extension["native_root"] if extension else None,
    gate_passed=not failed,
    cases=rows,
    not_proved=failed,
    tests=[j["outcomes"] for j in jobs["tests"]],
    iket_launches=[len(j["launches"]) for j in jobs["iket"]],
    minimum_host_available_bytes=min(
        m["available_bytes"] for r in receipts for m in r["memory"]
    ),
    rule="Every case in both boot groups: identical generated CUDA and aliased reference preparation with shared output, or a paired five-SE upper ratio at most 1; zero time tolerance; inconclusive is not a pass",
    extension_rule="If used: every primary mean must be at most 1; fixed 480 additional rounds on two boots for every unresolved case; each extension boot and each pooled primary-plus-extension group must independently pass; all primary observations retained",
    scope="GPU operation cases; full-model and host enqueue qualification remain",
)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        dict(passed=len(rows) - len(failed), not_proved=len(failed), cases=len(rows))
    )
)
raise SystemExit(bool(failed))
