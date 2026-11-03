#!/usr/bin/env python3
"""Audit a downloaded qualification phase and compare both boots to frozen records."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from analyze_functional import IMAGE, NODES, PIN, ROOT, load, require, sha

sys.path.insert(0, str(ROOT))
from servebench import behaviour, suite  # noqa: E402
from servebench.__main__ import compare_results, read_json  # noqa: E402
from scripts.servebench_suite_table import AA_FLOOR_PCT  # noqa: E402


def audit(snapshot):
    receipt = load(snapshot / "receipt.json")
    raw = snapshot / "raw"
    files = receipt["files"]
    require(sha((snapshot / "raw.tar.gz").read_bytes()) == receipt["archive_sha256"], "archive hash differs")
    require({str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()} == set(files), "manifest/file set differs")
    for name, item in files.items():
        path = (raw / name).resolve()
        require(path.is_relative_to(raw.resolve()), "manifest path escapes snapshot")
        data = path.read_bytes()
        require(len(data) == item["size"] and sha(data) == item["sha256"], f"artifact hash differs: {name}")
    phase = receipt["phase"]
    require(phase in ("release-default", "release-copy"), "not a performance phase")
    spec_path = Path(__file__).parent / (phase + ".json")
    relative = str(spec_path.relative_to(ROOT))
    tested_spec = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{relative}"])
    require(tested_spec == spec_path.read_bytes(), "spec differs from tested commit")
    spec = load(spec_path)
    require(receipt["run"] == spec["run"], "run identity differs")
    job = spec["jobs"][0]
    command = load(raw / "control" / (phase + ".command.json"))
    result = load(raw / "control" / (phase + ".result.json"))
    require(command["deployment_commit"] == result["deployment_commit"] == PIN and result["exit_code"] == 0, "phase failed or source differs")
    require(command["argv"] == result["argv"] and relative in command["argv"], "phase command differs")
    require(sha((raw / "control" / (phase + ".log")).read_bytes()) == result["log_sha256"], "phase log differs")
    boots, directories, identities = [], {}, []
    for arm in job["arms"]:
        label = arm["label"]
        observations, started = {}, None
        for number, name in enumerate(job["suite"]):
            suffix = "" if number == 0 else "-" + name
            directory = raw / "results/private/suites" / (spec["run"] + suffix) / label
            directories[label, name] = directory
            index = load(directory / "suite.json")
            definition = suite.load(name)
            require(index["suite"] == definition and index["suite_sha256"] == suite.digest(definition), f"suite definition differs: {label}/{name}")
            require(index.get("finished_utc") and len(index["steps"]) == len(definition["steps"]), f"incomplete suite: {label}/{name}")
            require([(s["id"], s["command"], s["exit"]) for s in index["steps"]] == [(s["id"], s["command"], 0) for s in definition["steps"]], f"failed or different ordered steps: {label}/{name}")
            metadata = index["metadata"]
            identity = metadata["identity"]
            require(not identity["live_problems"] and not identity["client_dirty"] and identity["client_commit"] == PIN, f"live/client mismatch: {label}/{name}")
            require(identity["cluster_config"] == arm["config"] and set(identity["nodes"]) == set(NODES), f"profile/rank set differs: {label}/{name}")
            fingerprints = set()
            times = {}
            for node, state in identity["nodes"].items():
                require(state["image_id"] == IMAGE and state["checkout"] == PIN, f"image/source differs: {label}/{name}/{node}")
                require(state["image_labels"]["local.spark-ds41f.vllm.tree"] == "c061e5f6931306fa4655ba531428f639c365bf10", "vLLM tree differs")
                fingerprints.add((tuple(state["image_layers"]), state["image_content_sha256"]))
                times[node] = state["container_started_at"]
            require(len(fingerprints) == 1, f"rank image content differs: {label}/{name}")
            if started is None:
                started = times
            require(times == started, f"companion is on another boot: {label}/{name}")
            conditions = load(directory / "conditions.json")
            require(conditions["halted"] is None and not conditions["fan_stop_failures"], f"condition/fan halt: {label}/{name}")
            for field in ("memory", "thermal"):
                require(set(conditions[field]) == set(NODES), f"conditions omit a rank: {label}/{name}/{field}")
            require({r["node"] for r in conditions["cooling"]} == set(NODES) and all(r["final_c"] < 55 for r in conditions["cooling"]), "cooling incomplete or above floor")
            require({r["node"] for r in conditions["fan_curve"]} == set(NODES), "fan curve omits a rank")
            for node in NODES:
                memory, thermal = conditions["memory"][node], conditions["thermal"][node]
                require(memory["min_available_gib"] >= 3 and memory["swap_growth_gib"] == 0, f"memory/swap problem: {label}/{name}/{node}")
                require(thermal["thermal_slowdown_ms"] == 0, f"thermal slowdown: {label}/{name}/{node}")
            steps = {}
            for step in index["steps"]:
                data = read_json(str(directory / step["output"]))
                schema = data["schema"]
                if schema == "servebench/1":
                    summary, table = data["summary"], data["request_table"]
                    requests = summary["requests"]
                    require(requests["sent"] > 0 and all(requests[k] == 0 for k in ("failed", "rejected", "foreign", "approximate_token_times")), f"request failure/foreign traffic: {label}/{name}/{step['id']}")
                    require(summary["output_tps"]["n"] == 10 and len(summary["output_tps"]["batches"]) == 10, "missing timing windows")
                    rows = [dict(zip(table["columns"], row)) for row in table["rows"]]
                    require(len(rows) == requests["sent"] and all(not r["error"] and r["completion_tokens"] == data["workload"]["max_tokens"] for r in rows), "raw request table differs from summary")
                    steps[step["id"]] = {"requests": requests, "output_tps": summary["output_tps"], "workload": data["workload"]}
                elif schema == "servebench-prefill/1":
                    require(not data["workload"]["skipped"] and len(data["samples"]) == 15, "missing prefill samples")
                    require({(s["size"], s["repeat"]) for s in data["samples"]} == {(size, repeat) for size in data["workload"]["sizes"] for repeat in range(3)}, "prefill repeat coverage differs")
                    require(all(not s["error"] and s["tps"] > 0 for s in data["samples"]) and all(r["failed"] == 0 and r["tps"]["n"] == 3 for r in data["prefill"]), "prefill request failed")
                    steps[step["id"]] = {"prefill": data["prefill"], "samples": data["samples"]}
                elif schema == "servebench-stall/1":
                    require(not data["workload"]["skipped"] and len(data["stall"]) == 1 and all(r["failed"] == 0 and r["tps"]["n"] == 3 for r in data["stall"]), "stall request failed or repeat missing")
                    steps[step["id"]] = data["stall"]
                elif schema == "servebench-engine/1":
                    require(len(data["engine"]) == 15 and all(len(r["samples"]) == 100 and r["step_ms"]["n"] == 100 for r in data["engine"]), "engine grid incomplete")
                    steps[step["id"]] = {"shapes": 15, "replays_each": 100}
                elif schema == "servebench-model/1":
                    generation, scores = data["generation"]["sequences"], data["scores"]
                    require(len(generation) == len(scores) == 256 and all(not r.get("error") and r["output_ids"] for r in generation) and all(not r.get("error") and r["tokens"] for r in scores), "model requests failed or missing")
                    require({r["id"] for r in generation} == {r["id"] for r in scores} == set(range(256)), "model prompt coverage differs")
                    corpus_generation = generation if data["corpus"]["source"] == "self" else read_json(str(directories[job["arms"][0]["label"], name] / "model.json.gz"))["generation"]["sequences"]
                    require(behaviour.corpus_digest(corpus_generation) == data["corpus"]["sha256"], "model corpus digest differs")
                    expected_tokens = {r["id"]: r["output_ids"] for r in corpus_generation}
                    require(all(r["tokens"] == expected_tokens[r["id"]] for r in scores), "model scores do not cover the declared corpus")
                    require(all(len(r["tokens"]) == len(r["rank"]) == len(r["logprob"]) for r in scores), "model scores incomplete")
                    require(all(math.isfinite(lp) for r in scores for lp in r["logprob"]) and all(math.isfinite(lp) for r in scores for position in r["top_logprobs"] for lp in position), "non-finite model log probabilities")
                    steps[step["id"]] = {"generated": len(generation), "scored": len(scores), "corpus_sha256": data["corpus"]["sha256"], "self_agreement": data["self_agreement"]}
                else:
                    raise ValueError(f"unrecognized schema: {schema}")
            observations[name] = {"steps": steps, "conditions": conditions, "suite_sha256": index["suite_sha256"]}
            telemetry_path = raw / "results/private/determinism" / spec["run"] / f"suite-{label}{suffix}.telemetry.json"
            telemetry = load(telemetry_path)
            require(set(telemetry["nodes"]) == set(NODES) and all(r["samples"] > 0 for r in telemetry["nodes"].values()), f"telemetry incomplete: {label}/{name}")
            observations[name]["telemetry"] = telemetry
            identities.append({"label": label, "suite": name, "identity": identity})
        memory_dir = raw / "results/private/determinism" / spec["run"]
        startup = load(memory_dir / f"startup-memory-{label}.json")
        require(set(startup) == set(NODES), "startup memory omits a rank")
        for node in NODES:
            lines = (memory_dir / f"startup-memory-{label}-{node}.txt").read_text().splitlines()
            require(len(lines) == startup[node]["samples"] and len(lines) > 0, "startup memory samples incomplete")
            require(round(min(float(line.split()[1]) for line in lines) / 1048576, 3) == startup[node]["min_available_gib"], "startup memory minimum differs from raw samples")
            require(startup[node]["min_available_gib"] >= 5, "startup memory below guard floor")
        boots.append({"label": label, "started_at": started, "startup_memory": startup, "suites": observations})
    require(len(boots) == 2 and all(boots[0]["started_at"][n] != boots[1]["started_at"][n] for n in NODES), "repeat is not another boot")
    comparisons, references = {}, {}
    first, repeat = [arm["label"] for arm in job["arms"]]
    judge = lambda a, b: compare_results(a, b, 3.0)
    for number, name in enumerate(job["suite"]):
        reference = ROOT / (job["reference"] + ("-" + name if number else ""))
        references[name] = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in reference.iterdir() if p.is_file()}
        for relative, digest in references[name].items():
            require(sha(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{relative}"])) == digest, f"reference changed since test publication: {relative}")
        a, b = directories[first, name], directories[repeat, name]
        changes = suite.compare(read_json, str(reference), str(a), judge)
        aa = suite.compare(read_json, str(a), str(b), judge)
        repeated = suite.compare(read_json, str(reference), str(b), judge)
        comparisons[name] = {"reference_to_boot1": changes, "reference_to_boot2": repeated, "boot1_to_boot2": aa, "headlines": suite.with_floor(changes, aa, floors=AA_FLOOR_PCT, reference_to_repeat=repeated)}
    adverse = [{"suite": name, **row} for name, comparison in comparisons.items() for row in comparison["headlines"] if row["verdict"] == "worse"]
    return {"phase": phase, "evidence_passed": True, "verified_files": len(files), "deployment_commit": PIN, "tested_spec_sha256": sha(tested_spec), "image_id": IMAGE, "boots": boots, "identities": identities, "reference_files": references, "comparisons": comparisons, "adverse_headline_verdicts": adverse, "limitations": ["The release reference omits model text; candidate repeat comparisons retain it privately.", "524K and 1M reference prompts differ in actual token counts; retain the comparison warnings.", "The extended reference's outer prefill duration is stale; compare per-size samples.", "Performance suites record deployment identity at suite entry, not a per-step rank-layout CRC."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.snapshot)
    if args.output:
        require(not args.output.exists(), "refusing to overwrite analysis")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("boots", "identities", "reference_files", "comparisons")}))


if __name__ == "__main__":
    main()
