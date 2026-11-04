#!/usr/bin/env python3
"""Audit the frozen candidate-39 functional data with explicit collector revalidation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PIN = "2e6bc10ed8ae6a5f36459e9985011aca070179e3"
IMAGE = "sha256:17550af8ae03a42919d72fac7d2771a18f0f30300b45f0b06edc1ac83625f885"
NODES = ("dgx1", "dgx2", "dgx3")
FAULT = re.compile(
    r"ERROR|CRITICAL|Traceback|device.side assert|CUDA error|"
    r"illegal memory access|gate logits contain|EngineCore.*died", re.I,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def json_lines(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # Trace/decomposition extras also emit human-readable lines.
    require(rows, f"no JSON records: {path}")
    return rows


def verify_export(snapshot):
    receipt = load(snapshot / "receipt.json")
    raw = snapshot / "raw"
    files = receipt["files"]
    require(sha((snapshot / "raw.tar.gz").read_bytes()) == receipt["archive_sha256"], "archive hash differs")
    require({str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()} == set(files), "manifest/file set differs")
    for name, record in files.items():
        p = (raw / name).resolve()
        require(p.is_relative_to(raw.resolve()), "manifest path escapes snapshot")
        data = p.read_bytes()
        require(len(data) == record["size"] and sha(data) == record["sha256"], f"artifact hash differs: {name}")
    control = load(raw / "results/private/lab/candidate-39-round-attempt-3/receipt.json")
    require(control["commit"] == PIN and control.get("completed") and control.get("restored") and control.get("ended"), "controller incomplete/source differs")
    require(not control.get("error") and not control.get("restore_error") and all(s.get("exit") == 0 for s in control["steps"]), "controller phase/restoration failed")
    require(load(raw / "control/exit.json")["exit"] == 0, "actual controller wait exit failed")
    return receipt, raw, files, control


def phase_context(snapshot, phase):
    receipt, raw, files, control = verify_export(snapshot)
    spec_path = Path(__file__).parent / (phase + ".json")
    relative = str(spec_path.relative_to(ROOT))
    tested_spec = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{relative}"])
    require(tested_spec == spec_path.read_bytes(), "spec differs from the executed commit")
    spec = json.loads(tested_spec)
    step = next(s for s in control["steps"] if s["log"] == phase + ".log")
    require(step["argv"][1:] == ["scripts/lab.py", "run", relative, "--production-config", "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json", "--keep-open"], "phase command differs")
    return receipt, raw, files, spec_path, tested_spec, spec


def audit(snapshot):
    receipt, raw, files, spec_path, tested_spec, spec = phase_context(snapshot, "functional")
    phase = "functional"
    job = spec["jobs"][0]
    log = (raw / "results/private/lab/candidate-39-round-attempt-3/functional.log").read_bytes()
    actual = re.findall(r"\b(\S+\.py) (candidate-39(?:-repeat)?) exit (\d+)\b", log.decode())
    expected = [(Path(script).name, arm["label"], "1" if stem == "runtime" else "0")
                for arm in job["arms"] for script, _, stem in job["extras"]]
    require(actual == expected, "ordered workload exits differ beyond the preserved stale-ID collectors")
    stale_issue = "image ID differs from the verified candidate-38 image"
    old_image = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
    original_collector = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:experiments/2026-11-03-candidate-38-reproduction/collect_runtime.py"])
    directory = raw / "results/private/determinism" / spec["run"]
    boots, sampled_rows = [], []
    for arm in job["arms"]:
        label = arm["label"]
        summaries = {}
        for _, arguments, stem in job["extras"]:
            path = directory / f"{stem}-{label}.jsonl"
            rows = json_lines(path)
            final = rows[-1]
            require((directory / f"{stem}-{label}.jsonl.telemetry.json").is_file(), f"missing telemetry: {stem}/{label}")
            if stem in ("determinism", "greedy", "determinism-16k"):
                require(final.get("identical") is True and all(r["token_diffs"] == r["logprob_diffs"] == 0 for r in final["scenarios"].values()), f"determinism failed: {stem}/{label}")
                names = {"c8", "staggered", "mixed", "cached", "again"} | ({"mixed15"} if "--wide" in arguments else set())
                require(set(final["scenarios"]) == names and len(rows) == len(names) + 1, f"determinism scenarios incomplete: {stem}/{label}")
                for row in rows[:-1]:
                    count = 16 if row["scenario"] == "mixed15" else 9 if row["scenario"] == "mixed" else 8
                    require(row["scenario"] in names and len(row["requests"]) == count and row["token_diffs"] == row["logprob_diffs"] == 0, f"determinism request count/differences: {stem}/{label}")
                    require(all(r.get("first_token_diff") is None and r.get("first_char_diff") is None and r.get("first_logprob_diff") is None and r.get("logprobs_equal") is not False for r in row["requests"]), f"raw determinism request differs: {stem}/{label}")
            elif stem in ("control-spellings", "controls"):
                require(final["summary"]["pass"] is True, f"API controls failed: {stem}/{label}")
            elif stem == "probes":
                require(final["summary"]["prompts"] == 16 and not final["summary"]["collapsed"], f"collapse probe failed: {label}")
            elif stem in ("sink-ordinary", "sink-after-probes"):
                require(final["summary"]["fallthroughs"] == 0, f"sink fallthrough: {label}")
                if stem == "sink-ordinary":
                    require(final["summary"]["flagged_rows"] == 0, f"ordinary traffic flagged: {label}")
            elif stem in ("controls-digest", "prompt-logprobs"):
                require(not final["failed"] and final["digests"], f"controls request failed: {stem}/{label}")
            elif stem == "images":
                require(final["digests"]["alone"] == final["digests"]["c4"], f"image batch difference: {label}")
            elif stem == "ced-cross-shape":
                require(final["passed"] and all(v["repeats_identical"] and v["cold"] for v in final["lengths"].values()), f"CED check failed: {label}")
            elif stem == "runtime":
                require(final["expected_image_id"] == old_image and final["collector_sha256"] == sha(original_collector), "original collector identity differs")
                require(final["profile"] == arm["config"] and final["profile_sha256"] == sha(subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:" + arm["config"]])), "collector profile differs")
                require(final["passed"] is False and final["issues"] == {n: [stale_issue] for n in NODES} and final["collected_nodes"] == list(NODES), f"runtime collector has additional/unexpected issues: {label}")
            elif stem in ("c4-distinct", "c8-distinct"):
                count = int(stem[1])
                require(final["samples"] == 4 and final["foreign_requests"] == 0 and len(rows) == 5, f"concurrency sample count differs: {label}/{stem}")
                require(all(r.get("tokens") == count * 256 and r.get("foreign_requests") == 0 for r in rows[:-1]), f"concurrency workload differs: {label}/{stem}")
            elif stem == "copy-probe":
                require([r["category"] for r in rows] == ["prose", "copy", "edit", "tool"] and all(r["requests"] == 12 and r["errors"] == 0 and r["output_tokens"] > 0 for r in rows), f"copy probe failed: {label}")
                final = {"categories": [{k: v for k, v in r.items() if k not in ("suffix", "dspark_acceptance_by_position")} for r in rows]}
            elif stem == "memory":
                ranks = {r["node"]: r for r in rows if r["node"] in NODES}
                require(set(ranks) == set(NODES), f"memory snapshot omits a rank: {label}")
                require(all(r["mem_available_gib"] > 0 and r["mem_total_gib"] > 0 and r["container"] for r in ranks.values()), f"memory snapshot failed on a rank: {label}")
                # The legacy extra also reads dgx4; it is an observer, not a TP rank.
                final = {"ranks": ranks, "other_observations": [r for r in rows if r["node"] not in NODES]}
            elif stem == "trace":
                trace = load(raw / final["output"])
                require(len(trace["requests"]) > 12 and all(not r.get("error") and r.get("completion_tokens") == 256 for r in trace["requests"]), f"latency trace request failed: {label}")
                for name, count, seconds in (("c4", 4, 120), ("c8", 8, 60)):
                    require(trace["phases"][name]["streams"] == count and trace["phases"][name]["seconds"] == seconds and final["summary"][name]["foreign_requests"] == 0, f"latency trace workload/traffic differs: {label}")
            if stem == "sampled":
                require(not final["scenarios"]["again"]["differing"], f"isolated sampled repeat differs: {label}")
                sampled_rows.append(rows)
            summaries[stem] = final
        ranks, fingerprints = {}, set()
        for node in NODES:
            p = directory / ("runtime-" + label) / node
            record = load(p / "runtime.json")
            engine = (p / "engine.log").read_bytes()
            c, image = record["container"], record["image"]
            require(record["issues"] == [stale_issue] and not record.get("error") and c["image_id"] == image["id"] == IMAGE and c["profile"] == arm["config"], f"runtime identity differs beyond the preserved stale-ID check: {label}/{node}")
            require(image["labels"]["local.spark-ds41f.vllm.tree"] == "8e33015b1fa1d34250e686db7d1f8061021d6d5a" and image["labels"]["local.spark-ds41f.vllm.patch-head"] == "31412339aafdd2c1392c7294975355dbc523c5e1", "runtime source labels differ")
            require(c["status"] == record["status_after_collection"] == "running" and c["restart_count"] == 0 and record["same_container_after_collection"], f"runtime changed: {label}/{node}")
            require(record["log"]["truncated"] is False and sha(engine) == record["log"]["sha256"] and not FAULT.search(engine.decode()), f"rank log failed or contains faults: {label}/{node}")
            fingerprints.add((tuple(image["layers"]), image["config_sha256"]))
            ranks[node] = {"container_id": c["id"], "started_at": c["started_at"], "image_id": image["id"], "log_sha256": sha(engine)}
        require(len(fingerprints) == 1, f"image Config/layers differ between ranks: {label}")
        startup = load(directory / f"startup-memory-{label}.json")
        require(set(startup) == set(NODES), f"startup memory omits a rank: {label}")
        for node in NODES:
            lines = (directory / f"startup-memory-{label}-{node}.txt").read_text().splitlines()
            require(len(lines) == startup[node]["samples"] and len(lines) > 0, f"startup memory samples incomplete: {label}/{node}")
        boots.append({"label": label, "summaries": summaries, "ranks": ranks, "startup_memory": startup})
    require(len(boots) == 2, "requires two boots")
    for node in NODES:
        a, b = [boot["ranks"][node] for boot in boots]
        require(a["container_id"] != b["container_id"] and a["started_at"] != b["started_at"], f"repeat reuses a boot: {node}")
    for stem in ("determinism", "greedy", "determinism-16k", "prompt-logprobs", "images", "ced-cross-shape"):
        if stem in boots[0]["summaries"]:
            require(boots[0]["summaries"][stem] == boots[1]["summaries"][stem], f"cross-boot digest differs: {stem}")
    a, b = [boot["summaries"].get("controls-digest", {}).get("digests", {}) for boot in boots]
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    require(set(changed) <= {"penalties-sampled/c8"}, "unexpected control digest changes")
    return {
        "phase": phase, "original_native_collectors_passed": False, "explicit_revalidation_passed": True, "verified_files": len(files),
        "ordered_script_exits": len(actual), "deployment_commit": PIN,
        "tested_spec_sha256": sha(tested_spec), "image_id": IMAGE, "boots": boots,
        "sampled_rows_equal_across_boots": sampled_rows[0] == sampled_rows[1],
        "stochastic_control_digest_changes": changed,
        "limitations": [
            "Seeded sampling can vary with batching; identical:false remains a recorded limitation, not a pass for batch invariance.",
            "The release record omits model text; this audit establishes these candidate repeats, not full text equality with the released build.",
            "No per-step rank-layout CRC or full-token CopyDrafter correctness recorder was used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.snapshot)
    if args.output:
        require(not args.output.exists(), "refusing to overwrite analysis")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "boots"}))


if __name__ == "__main__":
    main()
