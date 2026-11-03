#!/usr/bin/env python3
"""Audit a completed, downloaded TP3 functional phase without network or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PIN = "cb657ab1f5dba4e7cca48a48b0c2fe4e8d4a711f"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
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


def audit(snapshot):
    receipt = load(snapshot / "receipt.json")
    raw = snapshot / "raw"
    require(sha((snapshot / "raw.tar.gz").read_bytes()) == receipt["archive_sha256"], "archive hash differs")
    files = receipt["files"]
    require({str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file()} == set(files), "manifest/file set differs")
    for name, record in files.items():
        p = (raw / name).resolve()
        require(p.is_relative_to(raw.resolve()), "manifest path escapes snapshot")
        data = p.read_bytes()
        require(len(data) == record["size"] and sha(data) == record["sha256"], f"artifact hash differs: {name}")
    phase = receipt["phase"]
    require(phase in ("functional-default", "functional-copy"), "not a functional phase")
    spec_path = Path(__file__).parent / (phase + ".json")
    spec_relative = str(spec_path.relative_to(ROOT))
    tested_spec = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{spec_relative}"])
    require(tested_spec == spec_path.read_bytes(), "functional spec differs from the tested commit")
    spec = load(spec_path)
    require(receipt["run"] == spec["run"], "run identity differs")
    job = spec["jobs"][0]
    result = load(raw / "control" / (phase + ".result.json"))
    command = load(raw / "control" / (phase + ".command.json"))
    require(result["deployment_commit"] == command["deployment_commit"] == PIN and result["exit_code"] == 0, "phase command failed or source differs")
    require(result["argv"] == command["argv"] and str(spec_path.relative_to(ROOT)) in command["argv"], "recorded command differs")
    log = (raw / "control" / (phase + ".log")).read_bytes()
    require(sha(log) == result["log_sha256"], "phase log hash differs")
    actual = re.findall(r"\b(\S+\.py) (candidate-38(?:-copy)?(?:-repeat)?) exit (\d+)\b", log.decode())
    expected = [(Path(script).name, arm["label"], "0") for arm in job["arms"] for script, _, _ in job["extras"]]
    require(actual == expected, "ordered workload exits differ or failed")
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
                require(final["passed"] and not final["issues"] and final["collected_nodes"] == list(NODES), f"runtime collector failed: {label}")
            elif stem in ("c4-distinct", "c8-distinct"):
                count = int(stem[1])
                require(final["samples"] == 4 and final["foreign_requests"] == 0 and len(rows) == 5, f"concurrency sample count differs: {label}/{stem}")
                require(all(r.get("tokens") == count * 256 and r.get("foreign_requests") == 0 for r in rows[:-1]), f"concurrency workload differs: {label}/{stem}")
            elif stem == "copy-probe":
                require([r["category"] for r in rows] == ["prose", "copy", "edit", "tool"] and all(r["requests"] == 12 and r["errors"] == 0 and r["output_tokens"] > 0 for r in rows), f"copy probe failed: {label}")
                final = {"categories": [{k: v for k, v in r.items() if k not in ("suffix", "dspark_acceptance_by_position")} for r in rows]}
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
            require(not record["issues"] and c["image_id"] == image["id"] == IMAGE and c["profile"] == arm["config"], f"runtime identity differs: {label}/{node}")
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
        "phase": phase, "passed": True, "verified_files": len(files),
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
