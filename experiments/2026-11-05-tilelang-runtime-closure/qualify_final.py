"""Recompute final qualification remotely from the complete retained records."""

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics

from paired import classify, extend

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--runs", type=Path, default=HERE / "runs")
p.add_argument("--full", type=Path, default=HERE / "runs/full-native-embedding-v20")
p.add_argument("--prior-full", type=Path, default=HERE / "runs/full-initial")
p.add_argument("--run-map", type=Path, default=HERE / "accepted-runs.json")
p.add_argument("--out", type=Path, default=HERE / "qualification.json")
a = p.parse_args()
run_map = json.loads(a.run_map.read_text())


def read(path):
    if not path.exists():
        path = path.with_name(path.name + ".gz")
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as handle:
        return json.load(handle)


def point(run, operation, boot):
    return read(a.runs / run / f"{HERE.name}-{operation}-{boot}" / "result.json")


def flatten(values):
    for value in values:
        if isinstance(value, list):
            yield from flatten(value)
        else:
            assert math.isfinite(value) and value > 0
            yield value


def verify(row):
    measured = row.get("measured", row)
    logs = []
    for pair in measured["pairs"]:
        counts = Counter(pair["order"])
        assert set(counts) == set(pair["times_us"])
        assert all(count == 2 for count in counts.values())
        means = {role: statistics.fmean(flatten(values))
                 for role, values in pair["times_us"].items()}
        logs.append(math.log(means["candidate"] / means["reference"]))
    recorded = measured["log_ratios"]
    if isinstance(recorded, dict):
        recorded = recorded["candidate"]
    assert len(logs) == len(recorded)
    assert all(math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-13)
               for x, y in zip(logs, recorded, strict=True))
    gate = classify(logs, row.get("identical_program", False))
    assert gate["status"] == row["gate"]["status"]
    assert math.isclose(gate["upper_ratio"], row["gate"]["upper_ratio"], rel_tol=1e-12)
    if "profiles" in row:
        assert not row["profiles"]["candidate"]["aten_kernels"]
    return logs, gate


def brief(row):
    _, gate = verify(row)
    return dict(gate=gate, passed=gate["status"] != "not_proved")


core, embedding, drafting, transfer, wire, tests, iket = ({ } for _ in range(7))
for boot in (1, 2):
    key = str(boot)
    common = {r["name"]: r for r in point("combined-v4", "pair", boot)["cases"]}
    sampler = {r["name"]: r for r in point("final-sampler-v6", "pair", boot)["cases"]}
    tails = {r["name"]: r for r in point("tail-activity-v21", "activity", boot)["cases"]}
    extensions = {r["name"]: r for r in point("logits-extension-v6", "pair", boot)["cases"]}
    core[key] = {}
    for name, row in common.items():
        run = "combined-v4"
        if name.startswith("sampler-"):
            row, run = sampler[name], "final-sampler-v6"
        elif name.startswith("gather-tail-"):
            row, run = tails[name], "tail-activity-v21"
        result = dict(run=run, **brief(row))
        if name in extensions:
            result["extension"] = extend(verify(row)[0], verify(extensions[name])[0])
            result["passed"] = result["extension"]["passed"]
        if name in tails:
            # Preserve both clocks from the same graph; do not relabel event results.
            result["timer"] = row["measured"]["timer"]
            event_logs = [math.log(statistics.fmean(flatten(x["candidate"])) /
                                   statistics.fmean(flatten(x["reference"])))
                          for x in row["measured"]["event_intervals_us"]]
            result["secondary_event_interval_gate"] = classify(event_logs)
        core[key][name] = result
    assert len(core[key]) == 80 and len(tails) == 4

    primary = point("embedding-performance-v18", "embedding_bench", boot)
    extra = point("embedding-extension-v20", "embedding_bench", boot)
    assert primary["weight_sha256"] == extra["weight_sha256"]
    assert len(primary["unchanged_parallel"]) == 4
    assert all(x["same_cuda"] and x["new_cuda_sha256"] == x["old_cuda_sha256"]
               for x in primary["unchanged_parallel"])
    additional = {r["name"]: r for r in extra["cases"]}
    embedding[key] = {}
    for row in primary["cases"]:
        result = brief(row)
        if row["name"] in additional:
            followup = additional[row["name"]]
            assert row["input_sha256"] == followup["input_sha256"]
            assert len(verify(followup)[0]) == 2048
            result["extension"] = extend(verify(row)[0], verify(followup)[0])
            result["passed"] = result["extension"]["passed"]
        embedding[key][row["name"]] = result
    assert len(embedding[key]) == 32 and len(additional) == 3

    draft_primary = {r["name"]: r for r in point(run_map["drafting"][key], "draft_bench", boot)["cases"]}
    drafting[key] = {name: brief(row) for name, row in draft_primary.items()}
    draft_extra = point(run_map["draft_extension"][key], "draft_bench", boot)["cases"]
    assert len(draft_extra) == 1
    for followup in draft_extra:
        original = draft_primary[followup["name"]]
        for field in ("weight_sha256", "input_sha256", "ids_sha256"):
            assert original[field] == followup[field], field
        assert len(verify(followup)[0]) == 2048
        extension = extend(verify(original)[0], verify(followup)[0])
        drafting[key][followup["name"]].update(extension=extension, passed=extension["passed"])
    assert len(drafting[key]) == 6
    transfer[key] = {str((r["dtype"], r["rows"], r["stride"])): brief(r)
                     for r in point("final-wire-v9", "transfer", boot)["cases"]}
    assert len(transfer[key]) == 12
    wires = point("final-wire-v9", "wire", boot)["cases"]
    assert len(wires) == 16 and all(not r["aten_kernels"] for r in wires)
    wire[key] = dict(cases=len(wires), aten_kernels=0)
    test = point(run_map["tests"][key], "tests", boot)
    assert test["pytest_exit"] == 0 and test["outcomes"] == {"passed": 1446}
    assert test["family"] == "all"
    assert test["candidate"]["head"] == read(HERE / "candidate.json")["head"]
    tests[key] = test["outcomes"]
    instrumented = point(run_map["iket"][key], "iket", boot)
    assert instrumented["numerics"]["count"] == 33 and len(instrumented["required"]) == 14
    iket[key] = dict(numerical_cases=33, eager_and_graph=True,
                     required_factories=instrumented["required"],
                     decoded_launches=len(instrumented["launches"]))
assert tests["1"] == tests["2"]

full, reference = {}, None
if a.prior_full:
    old = read(a.prior_full / "tp4-c1-nsys-1/result/result.json")
    reference = (old["token_ids"], old["logprobs"])
for shape in (3, 4):
    for boot in (1, 2):
        name = f"tp{shape}-c1-nsys-{boot}"
        inventory = read(a.full / name / "inventory.json")
        request = inventory["phases"]["request"]
        assert request["launches"] == 7662 and set(request["totals"]) == {"tilelang"}
        result = read(a.full / name / "result/result.json")
        outputs = (result["token_ids"], result["logprobs"])
        if reference is None:
            reference = outputs
        assert outputs == reference
        full[name] = dict(request=request["totals"], outputs_unchanged=True,
                          trace_sha256=inventory["trace_sha256"])

families = dict(core=core, embedding=embedding, drafting=drafting, transfer=transfer)
unresolved = {family: {boot: [name for name, row in cases.items() if not row["passed"]]
                      for boot, cases in boots.items()}
              for family, boots in families.items()}
passed = not any(names for boots in unresolved.values() for names in boots.values())
result = dict(candidate=read(HERE / "candidate.json"), run_map=run_map, passed=passed,
              families=families, unresolved=unresolved, tests=tests, wire=wire,
              iket=iket, full_model=full,
              scope="Changed GPU paths, complete changed-path CPU result transfers, and the recorded 43-layer dummy-weight request. Both boots must pass a five-standard-error upper ratio <=1 or proven identical GPU program/bindings.",
              remaining_release_gates=["built image", "real-weight quality and throughput",
                                      "Spaceport site/profile reconciliation: TP4 fabric/master address and TP3 local overlay",
                                      "physical multi-node networking, including bulk NCCL",
                                      "untraced production features/workloads, including Engram and vision"])
a.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(passed=passed, unresolved=unresolved, tests=tests,
                      full_model_request_kernels=7662), indent=2))
