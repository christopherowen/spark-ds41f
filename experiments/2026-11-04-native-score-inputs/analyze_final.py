#!/usr/bin/env python3
"""Verify every selected GPU program and the final numerical/IKET checks."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("receipt", type=Path)
args = parser.parse_args()
r = json.loads(args.receipt.read_text())
c = json.loads((HERE / "candidate.json").read_text())
parents, hashes = {}, {}
for name in ("paired-results-v6.json", "paired-results-v7.json"):
    raw = (HERE / name).read_bytes()
    hashes[name] = hashlib.sha256(raw).hexdigest()
    parents[name] = json.loads(raw)
assert (
    parents["paired-results-v7.json"]["prior_analysis_sha256"]
    == hashes["paired-results-v6.json"]
)
previous = {
    name: {json.dumps(row["case"], sort_keys=True): row for row in data["cases"]}
    for name, data in parents.items()
}
assert r["closure"]["reservation_released"] and not r["completed"]["failed"]
assert Counter(j["job"]["operation"] for j in r["jobs"]) == dict(
    pair=2, tests=2, iket=2
)
assert all(
    j["exit"] == 0 and not j["memory_guard"] and j["result"]["candidate"] == c
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
        assert len(result["launches"]) == 56
        probe = result["probe_numerics"]
        assert len(probe) == 4
        for case in probe:
            assert case["score_layouts"] == [
                dict(layout="sliced", prefetch=False),
                dict(layout="transposed", prefetch=True),
            ]
            assert case["vocab"] == 16385 and case["logits_stride"] == 16400
            assert case["vector_run"] == (8 if case["dtype"] == "torch.bfloat16" else 4)
        continue
    assert len(result["cases"]) == 120 and result["timed_cases"] == 0
    counts = Counter()
    for row in result["cases"]:
        key = json.dumps(row["case"], sort_keys=True)
        evidence = row["transfer"]["evidence"]
        assert row["transfer"]["sha256"] == hashes[evidence]
        old = previous[evidence][key]
        assert old["passed"]
        data = previous["paired-results-v6.json"][key]
        assert all(row["data"] == b["data"] for b in data["boots"])
        programs = row["cuda_source_sha256"]
        assert all(
            programs["candidate"] == b["programs"]["candidate"]
            and programs["reference"] == b["programs"]["reference"]
            for b in old["boots"]
        )
        assert all(
            programs["previous"] == b["programs"]["candidate"] for b in data["boots"]
        )
        assert row["gate"] == dict(status="identical_qualified_program")
        counts[evidence] += 1
        rows.setdefault(key, []).append(
            dict(
                boot=j["job"]["boot"], evidence=evidence, programs=programs, passed=True
            )
        )
    assert counts == {"paired-results-v6.json": 112, "paired-results-v7.json": 8}
    kinds.update(counts)
assert len(rows) == 120 and all(len(v) == 2 for v in rows.values())
result = dict(
    source_head=c["head"],
    source_tree=c["tree"],
    native_source=r["identity"]["commit"],
    native_root=r["native_root"],
    gate_passed=True,
    cases=[
        dict(case=json.loads(k), boots=v, passed=True) for k, v in sorted(rows.items())
    ],
    qualified_sources_per_boot={k: n // 2 for k, n in kinds.items()},
    parent_analysis_sha256=hashes,
    tests_per_boot=673,
    iket_launches_per_boot=56,
    minimum_host_available_bytes=min(m["available_bytes"] for m in r["memory"]),
    rule="Every one of 120 GPU cases on both fresh boots must match identical reference and candidate CUDA, input hashes and strides from a per-case program qualified on both prior timing boots; no unqualified case or new program is accepted",
    scope="GPU operations; host enqueue and full-model qualification remain",
)
(HERE / "paired-results-final.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        dict(
            passed=120,
            cases=120,
            proof_counts=result["qualified_sources_per_boot"],
            tests_per_boot=673,
            iket_per_boot=56,
        )
    )
)
