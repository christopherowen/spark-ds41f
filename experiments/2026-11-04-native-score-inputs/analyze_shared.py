#!/usr/bin/env python3
"""Audit shared-staging timings and exact reuse of qualified GPU programs."""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from paired import audit

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
parser.add_argument("--candidate", type=Path, default=HERE / "candidate.json")
parser.add_argument("--changed-cases", type=int, default=8)
parser.add_argument("--partial", action="store_true")
parser.add_argument("--out", type=Path, default=HERE / "paired-results-v7.json")
args = parser.parse_args()
r = json.loads(args.receipt.read_text())
candidate = json.loads(args.candidate.read_text())
base_path = HERE / "paired-results-v6.json"
base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
base = json.loads(base_path.read_text())
old = {json.dumps(row["case"], sort_keys=True): row for row in base["cases"]}
assert r["closure"]["reservation_released"]
if args.partial:
    assert r.get("stop_reason") or r.get("completed")
else:
    assert not r["completed"]["failed"]
assert Counter(j["job"]["operation"] for j in r["jobs"]) == (
    dict(pair=2) if args.partial else dict(pair=2, tests=2, iket=2)
)
assert all(
    j["exit"] == 0 and not j["memory_guard"] and j["result"]["candidate"] == candidate
    for j in r["jobs"]
)
rows, kinds = {}, Counter()
for j in r["jobs"]:
    result = j["result"]
    op = j["job"]["operation"]
    if op == "tests":
        assert result["outcomes"] == {"passed": 673}
        continue
    if op == "iket":
        assert len(result["launches"]) == 40
        continue
    assert len(result["cases"]) == 120
    measured = 0
    for row in result["cases"]:
        key = json.dumps(row["case"], sort_keys=True)
        previous = old[key]
        assert all(row["data"] == p["data"] for p in previous["boots"])
        programs = row["cuda_source_sha256"]
        assert all(
            programs["reference"] == p["programs"]["reference"]
            and programs["previous"] == p["programs"]["candidate"]
            for p in previous["boots"]
        )
        if "transfer" in row:
            assert row["transfer"] == dict(evidence=base_path.name, sha256=base_sha)
            assert previous["passed"]
            assert programs["candidate"] == programs["previous"]
            assert row["gate"] == dict(status="equal_to_qualified_v6")
            kind = "identical_qualified_program"
            passed = True
            stats = None
        else:
            measured += 1
            _, stats = audit(row, 480)
            kind = "new_shared_program"
            passed = stats["status"] != "not_proved"
        kinds[kind] += 1
        rows.setdefault(key, []).append(
            dict(
                boot=j["job"]["boot"],
                kind=kind,
                passed=passed,
                gate=stats,
                programs=programs,
            )
        )
    assert measured == args.changed_cases
assert len(rows) == 120 and all(len(v) == 2 for v in rows.values())
cases = [
    dict(case=json.loads(k), boots=v, passed=all(b["passed"] for b in v))
    for k, v in sorted(rows.items())
]
failed = [c["case"] for c in cases if not c["passed"]]
result = dict(
    source_head=candidate["head"],
    source_tree=candidate["tree"],
    native_source=r["identity"]["commit"],
    native_root=r["native_root"],
    gate_passed=not failed and not args.partial,
    cases=cases,
    not_proved=failed,
    proof_counts_per_boot={k: n // 2 for k, n in kinds.items()},
    prior_analysis_sha256=base_sha,
    minimum_host_available_bytes=min(m["available_bytes"] for m in r["memory"]),
    tests_per_boot=0 if args.partial else 673,
    iket_launches_per_boot=0 if args.partial else 40,
    rule="Every case: same reference and same previously qualified CUDA program with identical input hashes and layouts, or new-program 480-round five-SE upper time ratio at most 1 on both boots; zero tolerance; no unqualified case reused",
    scope="GPU operations; host enqueue and full-model qualification remain",
)
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        dict(
            passed=120 - len(failed),
            not_proved=len(failed),
            cases=120,
            kinds=result["proof_counts_per_boot"],
        )
    )
)
raise SystemExit(bool(failed))
