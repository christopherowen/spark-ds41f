#!/usr/bin/env python3
"""Offline audit of both CopyDrafter reproduction boots and their transfer receipt.

Read a snapshot containing receipt.json and raw/. Missing repeat evidence exits 2
without writing outputs. Complete evidence writes a pass/fail analysis and a
receipt binding its exact inputs and command; existing outputs are never replaced.
This script makes no network calls and never runs an inference workload.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = "candidate-38-copy-m8-nodead-sp"
LABELS = ("copy-m8-nodead", "copy-m8-nodead-repeat")
NODES = ("sp5", "sp6", "sp7")
PROFILE = "experiments/2026-11-03-candidate-38-reproduction/arm-copy-m8-nodead.json"
PROFILE_SHA = "ada6fe4a3b41749fbe7b5cc35f994841e0bb33016afd922e826d3a0310cb095a"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
VLLM_HEAD = "1f33102547e4da5c64712390604695d1abd577c4"
VLLM_TREE = "c061e5f6931306fa4655ba531428f639c365bf10"
SPARKNET_TREE = "8cc28a5cd3e683c1f114ed843453e3ce34f323e0"
SUITE_SHA = "7c59b27e8503a6bec126b02a58756995b0d16fec42c28989519d567b67eaaefb"
CORPUS_SHA = "f5efd3d7a272672d9d039370d0a551ede96617e01ffde7d9a8fa17889cf1c7a0"
REVIEWED_COMMIT = "806c579763876bc042c8d44e7c10f42e59b6168f"
FAULT = re.compile(
    r"\bERROR\b|\bCRITICAL\b|Traceback|device.side assert|CUDA error|"
    r"illegal memory access|gate logits contain|EngineCore.*died|"
    r"Worker.*(?:crash|failed)|\b(?:nan|inf|nonfinite|non-finite)\b", re.I,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def read(path):
    return json.loads(path.read_text())


def json_lines(path, mixed=False):
    result = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except ValueError:
            if not mixed:
                raise ValueError(f"non-JSON line in {path.name}") from None
    return result


def audit_boot(raw, files, label):
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    def artifact(relative):
        check(str(relative) in files, f"artifact absent from receipt: {relative}")
        return raw / relative

    directory = Path("results/private/determinism") / RUN
    suite_dir = Path("results/private/suites") / (RUN + "-single-stream") / label
    probe = json_lines(artifact(directory / f"copy-probe-{label}.jsonl"))
    check([r.get("category") for r in probe] == ["prose", "copy", "edit", "tool"],
          "copy probe category order/count differs")
    check(all(r.get("requests") == 3 and r.get("errors") == 0
              and r.get("output_tokens", 0) > 0 and r.get("drafts", 0) > 0
              for r in probe), "copy probe did not complete all 12 requests cleanly")

    index = read(artifact(suite_dir / "suite.json"))
    c1 = read(artifact(suite_dir / "c1.json"))
    conditions = read(artifact(suite_dir / "conditions.json"))
    check(index.get("suite_sha256") == SUITE_SHA, "single-stream suite identity differs")
    steps = index.get("steps", [])
    check(len(steps) == 1 and steps[0].get("id") == "c1"
          and steps[0].get("exit") == 0 and steps[0].get("seconds", 0) >= 360,
          "single-stream c1 was refused, failed, missing, or too short")
    expected_workload = {"mode": "closed", "concurrency": 1, "warmup_s": 60.0,
                         "duration_s": 300.0, "batches": 10, "max_tokens": 256,
                         "fixed_length": True, "seed": 0, "prompts": "default",
                         "prompts_sha256": CORPUS_SHA}
    check(all(c1["workload"].get(k) == v for k, v in expected_workload.items()),
          "single-stream workload differs")
    check(c1["workload"].get("extra_body", {}).get("chat_template_kwargs", {}).get("thinking") is False,
          "single-stream thinking setting differs")
    requests = c1["summary"]["requests"]
    check(all(requests.get(k) == 0 for k in ("failed", "rejected", "foreign", "approximate_token_times")),
          "single-stream request errors, foreign traffic, or approximate token timings")
    check(requests.get("sent", 0) > 0 and requests.get("started_in_window", 0) > 0,
          "single-stream measured no requests")
    rates = c1["summary"]["output_tps"]
    check(rates.get("n") == 10 and len(rates.get("batches", [])) == 10
          and all(math.isfinite(v) and v > 0 for v in rates.get("batches", [])),
          "single-stream lacks 10 nonempty measured windows")
    table = c1["request_table"]
    rows = [dict(zip(table["columns"], row, strict=True)) for row in table["rows"]]
    check(table.get("window") == [60.0, 360.0] and len(rows) == requests.get("sent"),
          "single-stream request table/window is incomplete")
    check(bool(rows) and all(r.get("error") is None and r.get("completion_tokens") == 256 for r in rows),
          "single-stream request table has errors or shortened completions")
    check(bool(rows) and min(r["sent"] for r in rows) < 1 and max(r["end"] for r in rows) >= 360,
          "single-stream request timestamps do not span warmup and measurement")
    wrapper_path = artifact(directory / f"single-stream-{label}.jsonl")
    wrapper = json_lines(wrapper_path, mixed=True)
    check(bool(wrapper) and wrapper[-1].get("exit") == 0 and wrapper[-1].get("label") == label,
          "single-stream wrapper exit is absent or failed")
    text = wrapper_path.read_text()
    check("warm-up 60 s, then 300 s in 10 windows" in text
          and re.findall(r"window (\d+)/10 done", text) == [str(n) for n in range(1, 11)],
          "single-stream execution log does not show all 10 windows")
    check(conditions.get("halted") is None and not conditions.get("fan_stop_failures"),
          "single-stream halted or failed to restore fan state")

    concurrent = {}
    for count in (8, 16):
        data = json_lines(artifact(directory / f"c{count}-distinct-{label}.jsonl"))
        samples = [r for r in data if "sample" in r]
        summaries = [r for r in data if "summary" in r]
        check(len(data) == 4 and [r.get("sample") for r in samples] == [0, 1, 2],
              f"c{count} sample count/order differs")
        check(all(r.get("streams") == count and r.get("tokens") == count * 1024
                  and r.get("foreign_requests") == 0 and r.get("tps", 0) > 0
                  and len(r.get("wall", [])) == 2 and r["wall"][1] > r["wall"][0]
                  for r in samples), f"c{count} incomplete samples or foreign traffic")
        check(len(summaries) == 1 and summaries[0].get("samples") == 3
              and summaries[0].get("streams") == count and summaries[0].get("foreign_requests") == 0,
              f"c{count} final summary differs")
        concurrent[f"c{count}"] = {"samples": samples, "summary": summaries}

    stages = []
    for stage in ("copy-probe", "single-stream", "c8-distinct", "c16-distinct", "runtime"):
        telemetry = read(artifact(directory / f"{stage}-{label}.jsonl.telemetry.json"))
        check(set(telemetry.get("nodes", {})) == set(NODES)
              and all(n.get("samples", 0) > 0 for n in telemetry.get("nodes", {}).values()),
              f"{stage} lacks telemetry from every rank")
        stages.append({"stage": stage, "started_utc": telemetry["started_utc"],
                       "seconds": telemetry["seconds"], "gpu_clocks_lock": telemetry.get("gpu_clocks_lock"),
                       "max_gpu_c": {n: v.get("temperature.gpu", {}).get("max")
                                     for n, v in telemetry.get("nodes", {}).items()}})
    check(all(stamp(b["started_utc"]) >= stamp(a["started_utc"]) + a["seconds"] - 1.5
              for a, b in zip(stages, stages[1:])), "workload stages overlap or are out of order")
    startup = read(artifact(directory / f"startup-memory-{label}.json"))
    check(set(startup) == set(NODES), "startup memory observations omit a rank")
    for node in NODES:
        samples = artifact(directory / f"startup-memory-{label}-{node}.txt").read_text().splitlines()
        check(len(samples) == startup[node].get("samples") and len(samples) > 0
              and startup[node].get("seconds", 0) > 0,
              f"{node} startup memory samples are missing or incomplete")

    identity = index["metadata"]["identity"]
    check(identity.get("cluster_config") == PROFILE and identity.get("cluster_config_sha256") == PROFILE_SHA,
          "single-stream profile identity differs")
    check(identity.get("client_dirty") is False and not identity.get("live_problems"),
          "single-stream has a dirty client checkout or live configuration problems")
    check(set(identity.get("nodes", {})) == set(NODES), "single-stream identity rank set differs")
    runtime_dir = directory / f"runtime-{label}"
    runtime_summary = read(artifact(runtime_dir / "summary.json"))
    check(runtime_summary.get("passed") is True and not runtime_summary.get("issues")
          and runtime_summary.get("expected_nodes") == list(NODES)
          and runtime_summary.get("collected_nodes") == list(NODES)
          and runtime_summary.get("profile_sha256") == PROFILE_SHA,
          "runtime collector failed, missed a rank, or used another profile")
    records, fingerprints = {}, set()
    for node in NODES:
        runtime = read(artifact(runtime_dir / node / "runtime.json"))
        log = artifact(runtime_dir / node / "engine.log").read_bytes()
        container, image = runtime["container"], runtime["image"]
        check(container.get("image_id") == image.get("id") == IMAGE,
              f"{node} image is not the immutable candidate38 image")
        for field, expected in (("vllm.patch-head", VLLM_HEAD), ("vllm.tree", VLLM_TREE),
                                ("sparknet.tree", SPARKNET_TREE)):
            check(image["labels"].get("local.spark-ds41f." + field) == expected,
                  f"{node} source label mismatch: {field}")
        check(not runtime.get("issues") and runtime.get("same_container_after_collection") is True
              and runtime.get("status_after_collection") == container.get("status") == "running"
              and container.get("restart_count") == 0 and container.get("profile") == PROFILE,
              f"{node} container identity/state changed or collector reported issues")
        check(not any(p.startswith("/opt/spark-ds41f/candidate") for p in container["mount_destinations"]),
              f"{node} has a candidate source overlay")
        check(runtime["log"].get("exit_code") == 0 and runtime["log"].get("truncated") is False
              and bool(log) and digest(log) == runtime["log"].get("sha256"),
              f"{node} engine log is missing, truncated, failed, or has the wrong hash")
        matches = [{"line": i, "text": line} for i, line in enumerate(log.decode(errors="replace").splitlines(), 1)
                   if FAULT.search(line)]
        check(not matches, f"{node} engine log contains fault/error patterns")
        prior = identity["nodes"].get(node, {})
        check(prior.get("image_id") == IMAGE
              and prior.get("container_started_at") == container.get("started_at")
              and prior.get("image_layers") == image["layers"]
              and prior.get("checkout") == identity.get("client_commit"),
              f"{node} suite snapshot and post-work runtime identity disagree")
        check(stamp(container["started_at"]) < stamp(stages[0]["started_utc"])
              and stamp(runtime["collected_utc"]) >= stamp(stages[-1]["started_utc"]),
              f"{node} runtime timestamps do not cover the workload sequence")
        fingerprints.add((tuple(image["layers"]), image["config_sha256"]))
        records[node] = {"container_id": container["id"], "started_at": container["started_at"],
                         "collected_utc": runtime["collected_utc"], "restart_count": container["restart_count"],
                         "image_id": image["id"], "config_sha256": image["config_sha256"],
                         "layer_count": len(image["layers"]), "log": runtime["log"], "fault_matches": matches}
    check(len(fingerprints) == 1, "running image layers or Config differ between ranks")
    return {
        "label": label, "passed": not failures, "failures": failures,
        "copy_probe": [{k: v for k, v in r.items() if k not in ("suffix", "dspark_acceptance_by_position")}
                       for r in probe],
        "single_stream": {"suite_sha256": index["suite_sha256"], "steps": steps,
                          "workload": c1["workload"], "requests": requests,
                          "output_tps": rates, "request_table_rows": len(rows),
                          "request_table_window": table["window"], "conditions": conditions,
                          "client_commit": identity.get("client_commit"),
                          "live_warnings": identity.get("live_warnings", [])},
        "concurrent": concurrent, "stage_order": stages, "runtime": records,
        "startup_memory": startup,
        "per_step_rank_layout_observed": False,
        "generated_text_or_token_identity_checked": False,
    }


def audit_control(raw, files, boots):
    failures = []
    names = ("copy.log", "copy.command.json", "copy.result.json", "copy_runner.py")
    for name in names:
        relative = f"control/{name}"
        if relative not in files:
            failures.append(f"control artifact absent from receipt: {relative}")
        if not (raw / relative).is_file():
            raise FileNotFoundError(2, "missing control artifact", str(raw / relative))
    command = read(raw / "control/copy.command.json")
    result = read(raw / "control/copy.result.json")
    expected_argv = ["/usr/bin/python3", "scripts/lab.py", "run",
                     "experiments/2026-11-03-candidate-38-reproduction/lab-copy.json",
                     "--production-config", "config/candidate-37-tp3-switched-dev.local.json"]
    if command.get("argv") != expected_argv or result.get("exit_code") != 0:
        failures.append("recorded lab command differs or exited unsuccessfully")
    if any(result.get(k) != command.get(k) for k in ("argv", "cwd", "start_utc", "reviewed_commit")):
        failures.append("recorded lab command and result disagree")
    if command.get("cwd") != "/home/chris/projects/spark-ds41f" or command.get("reviewed_commit") != REVIEWED_COMMIT:
        failures.append("lab was not launched from the reviewed source identity")
    if any(boot["single_stream"]["client_commit"] != REVIEWED_COMMIT for boot in boots):
        failures.append("workload source identity differs from the reviewed lab launch")
    start, end = stamp(command["start_utc"]), stamp(result["end_utc"])
    if not all(start <= stamp(boot["stage_order"][0]["started_utc"])
               and stamp(boot["stage_order"][-1]["started_utc"]) <= end for boot in boots):
        failures.append("lab launch timestamps do not enclose both workload sequences")
    text = (raw / "control/copy.log").read_text()
    exits = re.findall(r"\b(copy_probe\.py|suite_single_stream\.py|distinct_streams\.py|collect_runtime\.py) "
                       r"(copy-m8-nodead(?:-repeat)?) exit (\d+)\b", text)
    expected = [(script, label, "0") for label in LABELS
                for script in ("copy_probe.py", "suite_single_stream.py", "distinct_streams.py",
                               "distinct_streams.py", "collect_runtime.py")]
    if exits != expected:
        failures.append("lab log lacks the ordered five successful script exits for each boot")
    return {"command": command, "result": result,
            "script_exits": [{"script": script, "label": label, "exit_code": int(code)}
                             for script, label, code in exits]}, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()
    if bool(args.output) != bool(args.receipt_output):
        parser.error("--output and --receipt-output must be supplied together")
    snapshot = args.snapshot.resolve()
    raw = snapshot / "raw"
    input_receipt = snapshot / "receipt.json"
    receipt = read(input_receipt)
    failures = []
    verified_files = 0
    archive = snapshot / "raw.tar.gz"
    if not archive.is_file() or digest(archive.read_bytes()) != receipt.get("archive_sha256"):
        failures.append("source archive is missing or does not match its receipt hash")
    for relative, expected in receipt["files"].items():
        path = (raw / relative).resolve()
        if not path.is_relative_to(raw.resolve()):
            raise SystemExit("receipt path escapes raw snapshot")
        if not path.is_file():
            failures.append(f"missing receipt artifact: {relative}")
            continue
        data = path.read_bytes()
        if len(data) != expected["size"] or digest(data) != expected["sha256"]:
            failures.append(f"receipt size/hash mismatch: {relative}")
        else:
            verified_files += 1
    boots, missing = [], []
    if not failures:
        for label in LABELS:
            try:
                boots.append(audit_boot(raw, receipt["files"], label))
            except FileNotFoundError as error:
                missing.append({"label": label, "missing": str(error.filename)})
            except (KeyError, TypeError, ValueError) as error:
                failures.append(f"{label}: malformed evidence: {type(error).__name__}: {error}")
    control = None
    if len(boots) == 2 and not missing:
        for node in NODES:
            if boots[0]["runtime"][node]["container_id"] == boots[1]["runtime"][node]["container_id"]:
                failures.append(f"{node}: repeat used the same container, not another boot")
        try:
            control, control_failures = audit_control(raw, receipt["files"], boots)
            failures.extend(control_failures)
        except FileNotFoundError as error:
            missing.append({"label": "control", "missing": str(error.filename)})
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"control: malformed evidence: {type(error).__name__}: {error}")
    complete = len(boots) == 2 and not missing
    passed = complete and not failures and all(boot["passed"] for boot in boots)
    analysis = {"schema_version": 1, "run": RUN, "complete": complete, "passed": passed,
                "verdict": "fault not observed in the two completed reproduction boots" if passed
                           else "incomplete evidence" if missing else "evidence checks failed",
                "failures": failures, "missing": missing, "verified_receipt_files": verified_files,
                "expected_image_id": IMAGE, "vllm_patch_head": VLLM_HEAD, "vllm_tree": VLLM_TREE,
                "boots": boots, "control": control,
                "limitations": ["No per-step rank layout CRC or GPU offsets were captured.",
                                "The original copy probe retains aggregates, not generated text or token sequences; this is not a text-exactness result.",
                                "A clean bounded reproduction is not a guarantee against every possible NaN.",
                                "Observed speeds and memory conditions are recorded without a baseline comparison or performance verdict."]}
    if args.output and complete:
        if args.output.exists() or args.receipt_output.exists():
            raise SystemExit("refusing to overwrite existing analysis or receipt")
        provenance = {"schema_version": 1, "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
                      "analyzer_path": str(Path(__file__).resolve().relative_to(ROOT)),
                      "analyzer_sha256": digest(Path(__file__).read_bytes()),
                      "snapshot": str(snapshot), "input_receipt_sha256": digest(input_receipt.read_bytes()),
                      "input_archive_sha256": receipt.get("archive_sha256"), "files": receipt["files"]}
        encoded = (json.dumps(analysis, indent=2) + "\n").encode()
        provenance["analysis_sha256"] = digest(encoded)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(encoded)
        with args.receipt_output.open("x") as handle:
            handle.write(json.dumps(provenance, indent=2) + "\n")
    printed = {k: v for k, v in analysis.items() if k != "boots"}
    printed["boot_checks"] = [{k: boot[k] for k in ("label", "passed", "failures")} for boot in boots]
    print(json.dumps(printed, indent=2))
    return 0 if passed else 2 if missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
