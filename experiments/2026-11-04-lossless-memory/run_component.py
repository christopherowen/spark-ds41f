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
from servebench import suite

HOLDER = "spark-lab: owner-authorized lossless memory qualification"
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
MANAGEMENT_ERRORS = ("ssh: connect", "cannot reach", "kex_exchange_identification",
                     "connection timed out during banner exchange", "memory telemetry stopped reporting",
                     "memory telemetry did not start")


def preflight(arm, through):
    kinds = ["image", "kernel", "full-gate"]
    if through == "release":
        kinds.append("release-tp3")
    problems = []
    for kind in kinds:
        spec = json.loads((arm / (kind + ".json")).read_text())
        problems.extend(kind + ": " + problem for problem in lab.plan_problems(lab.plan(spec)))
    return problems


def native_kernel_gate(value, images):
    rows = value.get("verdicts", [])
    if (value.get("agree") is not True or len(rows) != len(images)
            or {r.get("node") for r in rows} != set(images)):
        raise ValueError("kernel verdict omits a rank or disagrees")
    for row in rows:
        if (row.get("passed") is not True or row.get("exit") != 0
                or row.get("errors") or row.get("failed")
                or row.get("image") != images[row["node"]]["Id"]):
            raise ValueError("kernel gate failed or image differs")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("expert-scales", "engram"), required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--through", choices=("gate", "release"), default="release")
    parser.add_argument("--reuse-gates-from", type=Path,
                        help="reuse completed native image gates before any serving measurement")
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
    if args.reuse_gates_from:
        path = (ROOT / args.reuse_gates_from).resolve()
        if not path.is_relative_to(ROOT / "results/private"):
            parser.error("gate receipts must be retained native results")
        previous = json.loads(path.read_text())
        if (previous.get("component") != args.component or previous.get("gates_passed") is not True
                or previous.get("restored") is not True
                or any(s.get("log", "").startswith("boot-") for s in previous.get("steps", []))):
            raise SystemExit("prior execution lacks safely completed gates or already started serving measurement")
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
               "release_promoted": False, "completed": False}
    if previous is not None:
        receipt["reused_gates_from"] = str(args.reuse_gates_from)
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
        if len({tuple(v["Layers"]) for v in receipt["images"].values()}) != 1:
            raise ValueError("candidate layers differ between ranks")
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
            if previous is None:
                run([sys.executable, "scripts/lab.py", "run", str((arm / (name + ".json")).relative_to(ROOT)),
                     "--production-config", RESTORE, "--keep-open"], name)
            path = ROOT / "results/private/lab" / spec["run"] / "verdicts.json"
            native = json.loads(path.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if previous is not None and (previous[name]["sha256"] != digest
                                         or previous[name]["bundle_sha256"] != bundle_sha):
                raise ValueError("retained native gate or its test bundle changed")
            management_lost |= any(marker in json.dumps(native).lower() for marker in MANAGEMENT_ERRORS)
            management_lost |= any(v.get("exit") == 255 for v in native.get("verdicts", []))
            receipt[name] = {"native": native, "path": str(path.relative_to(ROOT)),
                             "sha256": digest, "bundle_sha256": bundle_sha}
            save()
            native_kernel_gate(native, receipt["images"])
        receipt["gates_passed"] = True
        if args.through == "release":
            previous_starts = None
            job = release["jobs"][0]
            for number, candidate in enumerate(job["arms"]):
                guard()
                label, profile = candidate["label"], candidate["config"]
                plan(RESTORE if number == 0 else profile, ["stop"])
                if not lab.stop_cluster():
                    raise RuntimeError("coordinated stop failed")
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
                starts = None
                for i, name in enumerate(job["suite"]):
                    suffix = "" if i == 0 else "-" + name
                    directory = ROOT / "results/private/suites" / (release["run"] + suffix) / label
                    argv = [sys.executable, "bin/spark", "--cluster-config", profile, "bench", "--suite", name,
                            "--compare", "none", "--label", label, "--output", str(directory.relative_to(ROOT))]
                    if number and i == 0:
                        first = ROOT / "results/private/suites" / release["run"] / job["arms"][0]["label"] / "model.json.gz"
                        argv += ["--corpus", str(first.relative_to(ROOT))]
                    run(argv, f"suite-{label}-{i}")
                    observed = native_suite_gate(directory, suite.load(name), profile, commit, receipt["images"])
                    if starts is not None and starts != observed:
                        raise ValueError("companion suites changed boot")
                    starts = observed
                if previous_starts and any(starts[node] == previous_starts[node] for node in starts):
                    raise ValueError("repeat did not boot every rank")
                previous_starts = starts
                save_logs("suites")
            for i, name in enumerate(job["suite"]):
                suffix = "" if i == 0 else "-" + name
                reference = job["reference"] + ("" if i == 0 else "-" + name)
                run([sys.executable, "scripts/servebench_suite_table.py", release["run"] + suffix,
                     job["arms"][0]["label"], reference, job["arms"][1]["label"]], f"table-{i}")
            receipt["two_serving_boots_complete"] = True
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
