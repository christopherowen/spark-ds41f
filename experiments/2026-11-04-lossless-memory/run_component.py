#!/usr/bin/env python3
"""Gate one isolated storage arm, measure two boots, and restore its owned window."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark
from spark_ds41f import lab
from spark_ds41f import sonar

suite = sonar.module("suite")  # servebench's suite module, now Sonar's

HOLDER = "spark-lab: owner-authorized lossless memory qualification"
STREAM_POLICY = "recorded-1030"
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
MANAGEMENT_ERRORS = ("ssh: connect", "cannot reach", "kex_exchange_identification",
                     "connection timed out during banner exchange", "memory telemetry stopped reporting",
                     "memory telemetry did not start")

IO_PROGRAM = r'''
import datetime, json, pathlib, re, subprocess, sys, time
started = time.monotonic_ns()
lines = subprocess.check_output(["docker", "top", sys.argv[1], "-eo", "pid"], text=True).splitlines()
processes = {}
for line in lines[1:]:
    pid = int(line.strip())
    root = pathlib.Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    counters = {key: int(value) for key, value in
                (line.split(":", 1) for line in (root / "io").read_text().splitlines())}
    processes[str(pid)] = {"start_ticks": int(stat[stat.rfind(")") + 2:].split()[19]), "io": counters}
disks = {}
for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
    fields = line.split()
    if re.fullmatch(r"nvme[0-9]+n[0-9]+", fields[2]):
        disks[fields[2]] = [int(value) for value in fields[3:]]
print(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "monotonic_start_ns": started, "monotonic_end_ns": time.monotonic_ns(),
                  "container": sys.argv[1], "processes": processes, "nvme_diskstats": disks}))
'''


def io_gate(value, container):
    required = {"rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes"}
    if value.get("container") != container or not value.get("processes") or not value.get("nvme_diskstats"):
        raise ValueError("I/O observation omits serving processes or NVMe counters")
    if not 0 < value["monotonic_start_ns"] <= value["monotonic_end_ns"]:
        raise ValueError("I/O observation has an invalid clock extent")
    for pid, row in value["processes"].items():
        if not pid.isdecimal() or row["start_ticks"] <= 0 or not required <= row["io"].keys():
            raise ValueError("I/O observation has incomplete process identity or counters")
        if any(type(number) is not int or number < 0 for number in row["io"].values()):
            raise ValueError("invalid process I/O counter")
    for row in value["nvme_diskstats"].values():
        if len(row) < 11 or any(type(number) is not int or number < 0 for number in row):
            raise ValueError("invalid NVMe disk counter")


def preflight(arm, through):
    kinds = ["image", "kernel", "full-gate"]
    if through == "release":
        kinds.append("release-tp3")
    problems = []
    for kind in kinds:
        spec = json.loads((arm / (kind + ".json")).read_text())
        problems.extend(kind + ": " + problem for problem in lab.plan_problems(lab.plan(spec)))
    return problems


def bundle_hashes(bundle):
    relative = str(bundle.relative_to(ROOT))
    tracked = subprocess.check_output(["git", "ls-files", "-z", "--", relative], cwd=ROOT).decode().split("\0")
    expected = {str((ROOT / path).relative_to(bundle)) for path in tracked if path}
    files = {str(path.relative_to(bundle)): path for path in bundle.rglob("*") if path.is_file()}
    if not expected or set(files) != expected:
        raise ValueError("bundle contains missing or non-Git inputs")
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}


def retained_external_sources(bundle, previous):
    candidate = json.loads((bundle / "candidate.json").read_text())
    commit = previous.get("commit", "")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("retained gate lacks a full published commit identity")
    result = {}
    for source, _destination in candidate.get("mounts", []):
        path = (bundle / source).resolve()
        if path.is_relative_to(bundle) and path.is_file():
            continue  # Covered by the complete native bundle input map.
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("cannot reuse unpinned external bundle input")
        name = str(path.relative_to(ROOT))
        original = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        current = path.read_bytes()
        if current != original:
            raise ValueError("external bundle source changed since its retained gate")
        result[name] = hashlib.sha256(current).hexdigest()
    return result


def native_kernel_gate(value, images, *, inputs=None):
    rows = value.get("verdicts", [])
    if (value.get("agree") is not True or len(rows) != len(images)
            or {r.get("node") for r in rows} != set(images)):
        raise ValueError("kernel verdict omits a rank or disagrees")
    for row in rows:
        if (row.get("passed") is not True or row.get("exit") != 0
                or row.get("errors") or row.get("failed")
                or row.get("image") != images[row["node"]]["Id"]):
            raise ValueError("kernel gate failed or image differs")
        if inputs is not None and row.get("inputs_sha256") != inputs:
            raise ValueError("native bundle inputs differ from current tracked source")


def native_suite_gate(directory, definition, config, commit, images):
    index = json.loads((directory / "suite.json").read_text())
    conditions = json.loads((directory / "conditions.json").read_text())
    expected = [(s["id"], s["command"], 0) for s in definition["steps"]]
    if (index.get("suite") != definition or index.get("suite_sha256") != suite.digest(definition)
            or not index.get("finished_utc")
            or [(s["id"], s["command"], s["exit"]) for s in index.get("steps", [])] != expected
            or conditions.get("halted") is not None or conditions.get("fan_stop_failures")):
        raise ValueError("incomplete/failed suite or condition halt")
    identity = index["metadata"]["identity"]
    if (identity.get("live_problems") or identity.get("client_dirty")
            or identity.get("client_commit") != commit or identity.get("cluster_config") != config
            or identity.get("client_stream_policy") != STREAM_POLICY
            or set(identity.get("nodes", {})) != set(images)):
        raise ValueError("suite has a different client, profile or rank set")
    if (set(conditions["memory"]) != set(images) or set(conditions["thermal"]) != set(images)
            or {r["node"] for r in conditions["cooling"]} != set(images)
            or any(r["final_c"] >= 55 for r in conditions["cooling"])
            or {r["node"] for r in conditions["fan_curve"]} != set(images)):
        raise ValueError("suite cooling/fan conditions omit a rank or exceed the starting floor")
    starts = {}
    for node, image in images.items():
        state = identity["nodes"][node]
        if (state["image_id"] != image["Id"] or state["checkout"] != commit
                or state["image_layers"] != image["Layers"]):
            raise ValueError("suite image/source differs from the gated image")
        starts[node] = state["container_started_at"]
        memory, thermal = conditions["memory"][node], conditions["thermal"][node]
        if (memory["min_available_gib"] < 3 or memory["swap_growth_gib"] != 0
                or thermal["thermal_slowdown_ms"] != 0):
            raise ValueError("memory, swap or thermal condition failed")
    return starts


def determinism_gate(text, *, logprobs):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    summaries = [r for r in rows if r.get("summary") == "temperature-0 determinism"]
    if (len(summaries) != 1 or summaries[0].get("identical") is not True
            or summaries[0].get("logprobs") is not logprobs
            or not summaries[0].get("reference_text_sha256") or not summaries[0].get("long_text_sha256")):
        raise ValueError("token/logprob equivalence failed, missing, or silently downgraded")
    return summaries[0]


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recorded_model_reference():
    provenance = HERE / "model-reference.json"
    value = json.loads(provenance.read_text())
    path = Path(value["model_path"])
    if (not path.is_relative_to(ROOT / "results/private")
            or path.stat().st_size != value["model_bytes"]
            or file_sha256(path) != value["model_sha256"]):
        raise ValueError("recorded model reference is absent or differs from pinned evidence")
    return {"path": str(path), "sha256": value["model_sha256"],
            "provenance_sha256": file_sha256(provenance),
            "analysis_sha256": file_sha256(HERE / "model_identity.py"),
            "sequences": value["sequences"], "positions": value["scored_positions"]}


def model_gate_command(candidate, output, image, reference):
    # This process runs only after the complete primary suite. It has no GPU,
    # network or serving namespace and cannot charge buffers to serving's cap.
    mounts = [(Path(reference["path"]), "/evidence/reference.json.gz", "ro"),
              (candidate, "/evidence/candidate.json.gz", "ro"),
              (HERE / "model-reference.json", "/evidence/provenance.json", "ro"),
              (HERE / "model_identity.py", "/evidence/model_identity.py", "ro"),
              (output, "/out", "rw")]
    argv = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--memory=1g", "--memory-swap=1g", "--cpus=1",
            "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=32", "--env", "CUDA_VISIBLE_DEVICES=", "--env", "NVIDIA_VISIBLE_DEVICES=void"]
    for source, target, mode in mounts:
        argv += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if mode == "ro" else "")]
    return [*argv, "--entrypoint", "python3", image, "/evidence/model_identity.py",
            "--reference", "/evidence/reference.json.gz", "--provenance", "/evidence/provenance.json",
            "--candidate", "/evidence/candidate.json.gz", "--out", "/out/report.json"]


def native_model_gate(value, candidate, reference):
    if (value.get("all_identical") is not True or value.get("inference_requests") != 0
            or value.get("reference_sha256") != reference["sha256"]
            or value.get("provenance_sha256") != reference["provenance_sha256"]
            or value.get("analysis_sha256") != reference["analysis_sha256"]
            or len(value.get("candidates", [])) != 1):
        raise ValueError("missing, failed or unpinned recorded-model comparison")
    row = value["candidates"][0]
    if (row.get("path") != "/evidence/candidate.json.gz"
            or row.get("sha256") != file_sha256(candidate)
            or row.get("identical") is not True or row.get("workload_identical") is not True
            or row.get("corpus_identical") is not True or row.get("error")
            or row.get("generation_differences") != [] or row.get("score_differences") != []
            or any(row.get(prefix + "_sequences") != reference["sequences"] for prefix in ("reference", "candidate"))
            or any(row.get(prefix + "_positions") != reference["positions"] for prefix in ("reference", "candidate"))
            or any(not row.get("reference_" + field + "_sha256")
                   or row.get("reference_" + field + "_sha256") != row.get("candidate_" + field + "_sha256")
                   for field in ("generation", "scores"))):
        raise ValueError("recorded tokens or returned score vectors differ or are incomplete")


def functional_digest_gate(text, reference):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    summaries = [row for row in rows if row.get("summary") == reference["summary"]]
    if (len(summaries) != 1 or summaries[0].get("digests") != reference["digests"]
            or any(summaries[0].get(key) != reference[key] for key in ("tokens", "seed") if key in reference)
            or summaries[0].get("failed", []) != [] or any(row.get("failed") or row.get("error") for row in rows)):
        raise ValueError("missing, changed or failed deterministic control/vision outputs")
    return summaries[0]


def agent_workload_gate(text):
    rows = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    if (len(rows) != 4 or {row.get("category") for row in rows} != {"prose", "copy", "edit", "tool"}
            or any(row.get("errors") != 0 or row.get("requests") != 12
                   or row.get("output_tokens", 0) <= 0 or row.get("error") for row in rows)):
        raise ValueError("ordinary agent workload has missing or failed requests")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("expert-scales", "engram"), required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--through", choices=("gate", "release"), default="release")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-gates-from", type=Path,
                       help="reuse completed native image gates before any serving measurement")
    reuse.add_argument("--reuse-kernel-from", type=Path,
                       help="reuse an image and passed component gate, then run a new full gate")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    arm = HERE / args.component
    config = str((arm / "tp3.json").relative_to(ROOT))
    release = json.loads((arm / "release-tp3.json").read_text())
    out = (ROOT / args.out).resolve()
    if not out.is_relative_to(ROOT / "results/private"):
        parser.error("receipts must stay under results/private")
    if not args.apply:
        problems = preflight(arm, args.through)
        print(json.dumps({"component": args.component, "restore": RESTORE, "window": args.window,
                          "build_and_gate": [str((arm / (n + ".json")).relative_to(ROOT))
                                             for n in ("image", "kernel", "full-gate")],
                          "release_spec": release, "through": args.through,
                          "out": str(out), "restores_at_end": True, "problems": problems}, indent=2))
        return int(bool(problems))
    def owns():
        hold = lab.read_hold()
        return bool(hold and hold.get("window") == args.window and hold.get("holder") == HOLDER
                    and hold.get("production_config") == RESTORE)
    if os.uname().nodename != "dgx1" or out.exists() or not owns():
        raise SystemExit("wrong host, prior receipts or foreign window; no operation")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("checkout is dirty")
    previous = None
    reuse_path = args.reuse_gates_from or args.reuse_kernel_from
    if reuse_path:
        path = (ROOT / reuse_path).resolve()
        if not path.is_relative_to(ROOT / "results/private"):
            parser.error("gate receipts must be retained native results")
        previous = json.loads(path.read_text())
        if (previous.get("component") != args.component or previous.get("restored") is not True
                or args.reuse_gates_from and previous.get("gates_passed") is not True
                or args.reuse_kernel_from and not previous.get("kernel")
                or any(s.get("log", "").startswith("boot-") for s in previous.get("steps", []))):
            raise SystemExit("prior execution lacks safely completed gates or already started serving measurement")
        if args.reuse_kernel_from:
            full = json.loads((arm / "full-gate.json").read_text())
            if (ROOT / "results/private/lab" / full["run"] / "verdicts.json").exists():
                raise SystemExit("full-gate observations already exist; use a distinct retained run")
    elif any((ROOT / "results/private/lab" / json.loads((arm / (n + ".json")).read_text())["run"]
              / "verdicts.json").exists() for n in ("kernel", "full-gate")):
        raise SystemExit("prior native gates exist; retain and review them before reusing")
    if args.through == "release" and any((ROOT / prefix / release["run"]).exists()
                                        for prefix in ("results/private/suites", "results/private/determinism")):
        raise SystemExit("prior serving observations exist; no overwrite or repeated measurement")
    lab.PRODUCTION_CONFIG = RESTORE
    out.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"started": lab.now(), "window": args.window, "commit": commit,
               "component": args.component, "steps": [], "plans": [], "images": {}, "restored": False,
               "release_promoted": False, "completed": False, "client_stream_policy": STREAM_POLICY}
    if previous is not None:
        receipt["reused_gates_from" if args.reuse_gates_from else "reused_kernel_from"] = str(reuse_path)
        receipt["prior_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    done, management_lost = threading.Event(), False
    active_profile, active_label = None, None
    cluster, nodes, upstreams = spark.configuration(argparse.Namespace(cluster_config=config))
    def save():
        lab.write_json_atomic(out / "receipt.json", receipt)
    def guard():
        if not owns() or lab.window_should_close(lab.read_hold()):
            raise RuntimeError("window changed or closure requested; retain partial observations")
    def heartbeat():
        while not done.wait(30):
            if owns():
                lab.beat("lossless " + args.component + ": " + receipt.get("phase", "preparation"))
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def run(argv, name, *, input_text=None):
        nonlocal management_lost
        guard()
        receipt["phase"] = name
        row = {"argv": argv, "started": lab.now(), "log": name + ".log"}
        receipt["steps"].append(row)
        save()
        with (out / row["log"]).open("x") as log:
            process = subprocess.run(argv, cwd=ROOT, input=input_text, text=True,
                                     stdout=log, stderr=subprocess.STDOUT)
        row.update(exit=process.returncode, ended=lab.now())
        text = (out / row["log"]).read_text(errors="replace").lower()
        management_lost |= any(marker in text for marker in MANAGEMENT_ERRORS)
        save()
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}")
    def plan(profile, action):
        argv = [sys.executable, "bin/spark", "--cluster-config", profile, "cluster", *action]
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        path = out / ("plan-" + str(len(receipt["plans"])) + "-" + action[0] + ".json")
        value = {"argv": argv, "exit": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        with path.open("x") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
        receipt["plans"].append(str(path.relative_to(ROOT)))
        save()
        print(process.stdout, flush=True)
        if (process.returncode or "--apply would refuse" in process.stdout
                or "shape:   CHANGED" in process.stdout):
            raise RuntimeError("cluster plan refused; no shape bypass")
        return argv
    def save_logs(stage, *, strict=True):
        nonlocal management_lost
        records, failed = [], False
        for node in nodes["nodes"]:
            relative = out.relative_to(ROOT) / "native-engine-logs" / f"{active_label}-{stage}-{node['name']}.log"
            native_path = str(Path(spark.repository_path(cluster)) / relative)
            command = ("umask 077; mkdir -p " + shlex.quote(str(Path(native_path).parent))
                       + "; set -C; docker logs --timestamps " + shlex.quote(cluster["container"]["name"])
                       + " > " + shlex.quote(native_path) + " 2>&1")
            exported = spark.run_ssh(nodes, node, "bash", "-c", command)
            process = spark.run_ssh(nodes, node, "head", "-c", "33554433", native_path)
            raw = (process.stdout + process.stderr).encode()
            path = out / f"engine-{active_label}-{stage}-{node['name']}.log"
            with path.open("xb") as file:
                file.write(raw)
            management_lost |= exported.returncode == 255 or process.returncode == 255 or any(
                m in (raw.decode(errors="replace") + exported.stderr).lower() for m in MANAGEMENT_ERRORS)
            failed |= exported.returncode != 0 or process.returncode != 0 or len(raw) > 32 * 1024**2
            records.append({"node": node["name"], "export_exit": exported.returncode,
                            "export_stderr": exported.stderr, "exit": process.returncode,
                            "native_full_log": native_path,
                            "path": str(path.relative_to(ROOT)), "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()})
        receipt.setdefault("engine_logs", []).append({"label": active_label, "stage": stage, "files": records})
        save()
        if strict and failed:
            raise RuntimeError("engine evidence unavailable or exceeds its 32 MiB/rank bound")
    def save_io(stage, *, strict=True):
        nonlocal management_lost
        records, failed = [], False
        for node in nodes["nodes"]:
            if not owns():
                raise RuntimeError("window changed before I/O observation")
            process = spark.run_ssh(nodes, node, "python3", "-c", IO_PROGRAM, cluster["container"]["name"])
            raw = process.stdout + process.stderr
            record = {"node": node["name"], "exit": process.returncode, "stdout": process.stdout,
                      "stderr": process.stderr, "program_sha256": hashlib.sha256(IO_PROGRAM.encode()).hexdigest()}
            management_lost |= process.returncode == 255 or any(marker in raw.lower() for marker in MANAGEMENT_ERRORS)
            try:
                if process.returncode or len(raw.encode()) > 65536:
                    raise ValueError("I/O observation failed or exceeded its output bound")
                record["observation"] = json.loads(process.stdout)
                io_gate(record["observation"], cluster["container"]["name"])
            except (ValueError, KeyError, TypeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
                failed = True
            records.append(record)
        path = out / f"io-{active_label}-{stage}.json"
        with path.open("x") as file:
            json.dump(records, file, indent=2)
            file.write("\n")
        receipt.setdefault("io_observations", []).append(str(path.relative_to(ROOT)))
        save()
        if strict and failed:
            raise RuntimeError("incomplete serving-process or NVMe I/O observation")
    code = 1
    try:
        guard()
        problems = preflight(arm, args.through)
        receipt["preflight_problems"] = problems
        save()
        if problems:
            raise RuntimeError("invalid measurement plans: " + "; ".join(problems))
        problems = lab.published_problems()
        if problems:
            raise RuntimeError("publish guard: " + "; ".join(problems))
        if args.through == "release":
            receipt["model_reference"] = recorded_model_reference()
            functional_reference = json.loads((HERE / "functional-reference.json").read_text())
            for check in functional_reference["checks"]:
                if file_sha256(ROOT / check["script"]) != check["script_sha256"]:
                    raise ValueError("deterministic functional workload source changed")
            receipt["functional_reference_sha256"] = file_sha256(HERE / "functional-reference.json")
            save()
        plan(RESTORE, ["stop"])
        if previous is None:
            run([sys.executable, "scripts/lab.py", "run", str((arm / "image.json").relative_to(ROOT)),
                 "--production-config", RESTORE, "--keep-open"], "image")
        run([sys.executable, "bin/spark", "--cluster-config", config, "build", "check"], "build-check")
        for node in nodes["nodes"]:
            process = spark.run_ssh(nodes, node, "docker", "image", "inspect", cluster["container"]["image"],
                "--format", '{"Id":{{json .Id}},"Labels":{{json .Config.Labels}},"Layers":{{json .RootFS.Layers}}}')
            if process.returncode:
                raise RuntimeError("candidate image inspection failed")
            image = json.loads(process.stdout)
            for key, value in cluster["container"]["expected_labels"].items():
                if image["Labels"].get(key) != value:
                    raise ValueError("candidate source label differs: " + key)
            receipt["images"][node["name"]] = image
        if (len({v["Id"] for v in receipt["images"].values()}) != 1
                or len({tuple(v["Layers"]) for v in receipt["images"].values()}) != 1):
            raise ValueError("candidate image identity or layers differ between ranks")
        builder = next(n["name"] for n in nodes["nodes"] if n["head"])
        image_id = receipt["images"][builder]["Id"]
        build_directory = spark.build_directory(spark.build_inputs(upstreams))
        built = json.loads((build_directory / "images" / (image_id.removeprefix("sha256:")[:16] + ".json")).read_text())
        if (built.get("image_id") != image_id or built.get("tag") != cluster["container"]["image"]
                or not built.get("built_utc") or built.get("compile_jobs", 0) < 1
                or built.get("min_available_gib", 0) < 8
                or built.get("deployment_commit") != receipt["images"][builder]["Labels"].get("org.opencontainers.image.revision")):
            raise ValueError("missing original successful build/import-smoke receipt")
        receipt["build_receipt"] = built
        if previous is not None and previous["images"] != receipt["images"]:
            raise ValueError("image changed since the retained native gates")
        save()
        for name in ("kernel", "full-gate"):
            spec = json.loads((arm / (name + ".json")).read_text())
            bundle = ROOT / spec["jobs"][0]["bundles"][0]["bundle"] / "candidate.json"
            bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
            inputs = bundle_hashes(bundle.parent)
            reusing = previous is not None and (name == "kernel" or bool(args.reuse_gates_from))
            external = retained_external_sources(bundle.parent, previous) if reusing else {}
            if not reusing:
                run([sys.executable, "scripts/lab.py", "run", str((arm / (name + ".json")).relative_to(ROOT)),
                     "--production-config", RESTORE, "--keep-open"], name)
            path = ROOT / "results/private/lab" / spec["run"] / "verdicts.json"
            native = json.loads(path.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if reusing and (previous[name]["sha256"] != digest
                            or previous[name]["bundle_sha256"] != bundle_sha):
                raise ValueError("retained native gate or its test bundle changed")
            management_lost |= any(marker in json.dumps(native).lower() for marker in MANAGEMENT_ERRORS)
            management_lost |= any(v.get("exit") == 255 for v in native.get("verdicts", []))
            receipt[name] = {"native": native, "path": str(path.relative_to(ROOT)),
                             "sha256": digest, "bundle_sha256": bundle_sha,
                             "inputs_sha256": inputs, "retained_external_sources_sha256": external}
            save()
            native_kernel_gate(native, receipt["images"], inputs=inputs)
        receipt["gates_passed"] = True
        if args.through == "release":
            previous_starts = None
            job = release["jobs"][0]
            for number, candidate in enumerate(job["arms"]):
                guard()
                label, profile = candidate["label"], candidate["config"]
                argv = plan(RESTORE if number == 0 else profile, ["stop", "--parallel"])
                run([*argv, "--apply"], "stop-" + label)
                argv = plan(profile, ["start", "--replace"])
                active_profile, active_label = profile, label
                memory_out = ROOT / "results/private/determinism" / release["run"] / f"startup-memory-{label}"
                samplers = lab.start_memory_samplers(memory_out)
                try:
                    run([*argv, "--apply"], "boot-" + label)
                finally:
                    startup = lab.stop_memory_samplers(samplers)
                    lab.write_json_atomic(memory_out.with_suffix(".json"), startup)
                if set(startup) != set(receipt["images"]) or any(v.get("min_available_gib", 0) < 5 for v in startup.values()):
                    raise ValueError("startup memory observation missing or below floor")
                save_logs("startup")
                if args.component == "engram":
                    run([sys.executable, str((arm / "check_live_cache.py").relative_to(ROOT)), lab.head_url(),
                         "--config", profile, "--run", release["run"], "--label", label], "cache-" + label)
                    save_io("before-suites")
                starts = None
                for i, name in enumerate(job["suite"]):
                    suffix = "" if i == 0 else "-" + name
                    directory = ROOT / "results/private/suites" / (release["run"] + suffix) / label
                    argv = [sys.executable, "bin/spark", "--cluster-config", profile, "bench", "--suite", name,
                            "--compare", "none", "--label", label, "--output", str(directory.relative_to(ROOT))]
                    if number and i == 0:
                        first = ROOT / "results/private/suites" / release["run"] / job["arms"][0]["label"] / "model.json.gz"
                        argv += ["--corpus", str(first.relative_to(ROOT))]
                    run(["env", "SPARK_DS41F_BENCH_STREAM_POLICY=" + STREAM_POLICY, *argv], f"suite-{label}-{i}")
                    if args.component == "engram":
                        save_io("after-suite-" + str(i))
                    observed = native_suite_gate(directory, sonar.load_suite(name), profile, commit, receipt["images"])
                    if starts is not None and starts != observed:
                        raise ValueError("companion suites changed boot")
                    starts = observed
                    if i == 0:
                        comparison = out / ("model-identity-" + label)
                        comparison.mkdir()
                        candidate_model = directory / "model.json.gz"
                        argv = model_gate_command(candidate_model, comparison, image_id, receipt["model_reference"])
                        try:
                            run(argv, "model-identity-" + label)
                        finally:
                            path = comparison / "report.json"
                            receipt.setdefault("model_identity", {})[label] = {
                                "path": str(path.relative_to(ROOT)),
                                "sha256": file_sha256(path) if path.exists() else None}
                            save()
                        native_model_gate(json.loads(path.read_text()), candidate_model, receipt["model_reference"])
                        receipt["model_identity"][label]["passed"] = True
                        save()
                if previous_starts and any(starts[node] == previous_starts[node] for node in starts):
                    raise ValueError("repeat did not boot every rank")
                previous_starts = starts
                save_logs("suites")
            for i, name in enumerate(job["suite"]):
                suffix = "" if i == 0 else "-" + name
                reference = job["reference"] + ("" if i == 0 else "-" + name)
                run([sys.executable, "scripts/sonar_table.py", release["run"] + suffix,
                     job["arms"][0]["label"], reference, job["arms"][1]["label"]], f"table-{i}")
            receipt["two_serving_boots_complete"] = True
            receipt["recorded_model_identity_passed"] = True
            functional = ROOT / "results/private/bench" / (release["run"] + "-functional")
            run([sys.executable, "bin/spark", "--cluster-config", profile, "bench", "--compare", "none",
                 "--suites", "quality,prefix,admission", "--prefix-tokens", "32768",
                 "--admission-tokens", "485000", "--admission-output-tokens", "8192", "--admission-force-length",
                 "--output", str(functional.relative_to(ROOT))], "functional-bench")
            report = json.loads((functional / "bench.json").read_text())
            if (report.get("aborted") or not report["suites"]["quality"]["ok"]
                    or not report["suites"]["admission"]["ok"]
                    or len(report["suites"]["prefix"]["runs"]) != 3
                    or not all(r["ok"] for r in report["suites"]["prefix"]["runs"])):
                raise ValueError("quality, prefix reuse or four-long-context admission failed")
            checks = []
            for mode in ("logprobs", "greedy"):
                argv = [sys.executable, "experiments/2026-10-12-r6g/token_determinism_text.py", lab.head_url(),
                        "--tokens", "192", "--long-lines", "700", "--wide"]
                if mode == "greedy":
                    argv += ["--no-logprobs"]
                run(argv, "determinism-" + mode)
                checks.append(determinism_gate((out / ("determinism-" + mode + ".log")).read_text(),
                                               logprobs=mode == "logprobs"))
            for key in ("reference_text_sha256", "long_text_sha256"):
                if checks[0][key] != checks[1][key]:
                    raise ValueError("greedy and gathered paths produced different text")
            receipt["functional_digests"] = {}
            for check in functional_reference["checks"]:
                name = "functional-digest-" + check["name"]
                run([sys.executable, check["script"], lab.head_url(), *check["args"]], name)
                receipt["functional_digests"][check["name"]] = functional_digest_gate(
                    (out / (name + ".log")).read_text(), check)
                save()
            run([sys.executable, "experiments/2026-10-23-next-release/acceptance/copy_probe.py",
                 lab.head_url(), "--prompts", "12"], "ordinary-agent-workload")
            receipt["ordinary_agent_workload"] = agent_workload_gate((out / "ordinary-agent-workload.log").read_text())
            receipt["functional_checks_complete"] = True
        receipt["completed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        try:
            if owns():
                if active_profile is not None:
                    save_logs("cleanup", strict=False)
                    if args.component == "engram":
                        save_io("cleanup", strict=False)
                unreachable = [n["name"] for n in nodes["nodes"] if spark.run_ssh(nodes, n, "true").returncode]
                if management_lost or unreachable:
                    hold = lab.read_hold()
                    hold.update(holder="lossless-memory-hardware-review",
                                rule="management unavailable; no launch until isolated and explicitly authorized")
                    lab.write_json_atomic(lab.HOLD, hold)
                    receipt["management_unreachable"] = unreachable
                    receipt["management_loss_latched"] = management_lost
                    raise RuntimeError("management loss; reservation kept without a restoration launch")
                if not owns():
                    raise RuntimeError("window changed during cleanup; no restoration launch")
                plan(RESTORE, ["start", "--replace"])
                if owns():
                    receipt["restored"] = lab.window_close("lossless component measurement finished or deferred")
            elif lab.read_hold() is None:
                receipt["restored"] = lab.production_live()
            else:
                receipt["restore_deferred"] = "foreign reservation; no serving action"
        except BaseException as error:
            receipt["restore_error"] = f"{type(error).__name__}: {error}"
            if owns():
                hold = lab.read_hold()
                hold.update(holder="lossless-memory-restoration-review",
                            rule="restoration guard failed; retain reservation without another launch")
                lab.write_json_atomic(lab.HOLD, hold)
        done.set()
        thread.join(timeout=5)
        if not receipt["restored"]:
            code = 1
        receipt["ended"] = lab.now()
        save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
