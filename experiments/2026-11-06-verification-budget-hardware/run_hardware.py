#!/usr/bin/env python3
"""Run only inside this experiment's already-opened, exclusively owned window.

Open the window from the original clean main checkout, then use coordinated
cluster sync --revision to deploy this published branch. This runner neither
weakens the window publication guard nor takes over an existing foreign hold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab, bench

EXPERIMENT = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
HOLDER = "spark-lab: verification budget hardware"
IMAGE = "sha256:7ae96e98e361601d8232cd99e6af1c6044b7c282bf6904c989e6ad033c147f82"
PARENT = "experiments/2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-2/performance/candidate-40"
GATES = [
    ("greedy", "experiments/2026-10-12-r6g/token_determinism_text.py", ["--tokens", "192", "--long-lines", "700", "--wide"]),
    ("sampled", "experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"]),
    ("controls", "experiments/2026-10-24-tilelang-completion/controls_digest.py", ["--tokens", "96"]),
]


def gate_summary(text, name):
    records = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "summary" in value:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"{name}: missing or ambiguous gate summary")
    result = records[0]
    if name in ("greedy", "sampled"):
        if result.get("identical") is not True or not re.fullmatch(r"[0-9a-f]{64}", result.get("reference_sha256", "")):
            raise ValueError(f"{name}: batch/seed invariance failed")
        return {k: result[k] for k in ("reference_sha256", "long_sha256") if k in result}
    if result.get("failed") != [] or not result.get("digests"):
        raise ValueError("controls: failed or absent results")
    digests = result["digests"]
    for key, value in digests.items():
        if key.endswith("/alone") and digests.get(key.removesuffix("/alone") + "/c8") != value:
            raise ValueError(f"controls: batch mismatch for {key}")
    return digests


def owns(hold, window):
    return bool(hold and hold.get("holder") == HOLDER and hold.get("window") == window
                and hold.get("production_config") == RESTORE)


def validate_plan(text, operation):
    if "--apply would refuse" in text or re.search(r"shape:\s+CHANGE[DS]", text):
        raise ValueError("cluster plan refused; no shape bypass")
    # Stop plans report live state and covered nodes, but have no shape line.
    if operation in ("start", "stop") and "live:" not in text:
        raise ValueError("cluster plan lacks live evidence")
    if operation == "start" and "shape:" not in text:
        raise ValueError("start plan lacks shape evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "qualify010", "qualify020", "restore"), default="screen")
    parser.add_argument("--window")
    parser.add_argument("--restore-commit", help="exact original deployment SHA captured before branch sync")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / EXPERIMENT / "screen.json").read_text())
    arms = spec["jobs"][0]["arms"]
    suites = [f"{EXPERIMENT}/screen-suite.json"]
    if args.stage.startswith("qualify"):
        row = "row" + args.stage[-3:]
        arms = [{"config": f"{row}.json", "label": f"{row}-{boot}"} for boot in ("a", "b")]
        suites = ["release-tp3", "single-stream", "sampled"]
    if args.stage == "restore":
        arms = []
    if not args.apply:
        print(json.dumps({"stage": args.stage, "arms": arms, "suites": suites,
                          "restore_profile": RESTORE, "image": IMAGE,
                          "requires_owned_window": HOLDER, "shape_change": False}, indent=2))
        return 0
    if os.uname().nodename != "dgx1" or not owns(lab.read_hold(), args.window):
        raise SystemExit("wrong host or foreign/missing window; no operation performed")
    if not args.restore_commit or not re.fullmatch(r"[0-9a-f]{40}", args.restore_commit):
        raise SystemExit("an exact captured restore commit is required")
    if subprocess.run(["git", "merge-base", "--is-ancestor", args.restore_commit, "origin/main"], cwd=ROOT).returncode:
        raise SystemExit("restore commit is not on the fetched origin/main")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("dirty deployment checkout")
    lab.PRODUCTION_CONFIG = RESTORE
    out = ROOT / f"results/private/lab/verification-budget-hardware/{args.window}/{args.stage}"
    if out.exists():
        raise SystemExit("prior stage receipts exist; preserve and review before retrying")
    out.mkdir(parents=True)
    receipt = {"started": lab.now(), "window": args.window, "stage": args.stage,
               "commit": lab.checkout_commit(), "restore_commit": args.restore_commit,
               "restore_profile": RESTORE, "steps": [], "restored": False,
               "gate_sources": {name: {"path": script, "sha256": hashlib.sha256((ROOT / script).read_bytes()).hexdigest()}
                                for name, script, _ in GATES}}
    done = threading.Event()
    def save():
        lab.write_json_atomic(out / "receipt.json", receipt)
    def heartbeat():
        while not done.wait(30):
            if owns(lab.read_hold(), args.window):
                lab.beat(f"verification budget {args.stage}")
    threading.Thread(target=heartbeat, daemon=True).start()
    save()
    active_arm = None
    def command(argv, name, restoring=False):
        if not owns(lab.read_hold(), args.window):
            raise RuntimeError("window ownership changed")
        if not restoring and active_arm is None and lab.window_should_close(lab.read_hold()):
            raise RuntimeError("handoff requested")
        path = out / f"{len(receipt['steps']):03d}-{name}.log"
        entry = {"argv": argv, "started": lab.now(), "log": path.name}
        receipt["steps"].append(entry); save()
        with path.open("w") as stream:
            code = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        entry.update(exit=code, ended=lab.now()); save()
        if code:
            raise RuntimeError(f"{name} failed ({code}); retained {path.name}")
        return path
    def action(config, operation, *extra, restoring=False):
        argv = [sys.executable, "bin/spark", "--cluster-config", config, "cluster", operation, *extra]
        plan = command(argv, operation + "-plan", restoring)
        text = plan.read_text()
        print(text, flush=True)
        validate_plan(text, operation)
        command([*argv, "--apply"], operation + "-apply", restoring)
    last_config = None
    success = False
    try:
        if args.stage.startswith("qualify") and lab.window_should_close(lab.read_hold()):
            raise RuntimeError("handoff requested before qualification")
        nodes = lab.nodes_config()
        identities = {}
        for node in nodes["nodes"]:
            result = lab.spark.run_ssh(nodes, node, "docker", "image", "inspect", IMAGE, "--format", "{{.Id}}")
            if result.returncode or result.stdout.strip() != IMAGE:
                raise RuntimeError(f"{node['name']}: pinned image unavailable")
            identities[node["name"]] = result.stdout.strip()
        receipt["images"] = identities; save()
        signatures = None
        if args.stage.startswith("qualify"):
            screen = out.parent / "screen"
            if json.loads((screen / "receipt.json").read_text()).get("completed") is not True:
                raise RuntimeError("qualification needs this window's completed causal screen")
            signatures = json.loads((screen / "control035-a-signatures.json").read_text())
        for index, arm in enumerate(arms):
            # A window's first arm includes its preceding coordinated sync.
            # Finish that job on a new request, then hand off before the next arm.
            if index and lab.window_should_close(lab.read_hold()):
                raise RuntimeError("handoff requested after completed arm")
            config, label = f"{EXPERIMENT}/{arm['config']}", arm["label"]
            active_arm = label
            # Record the profile before a partial start can fail, for coordinated cleanup.
            last_config = config
            samplers = lab.start_memory_samplers(out / f"startup-{label}")
            try:
                action(config, "start", "--replace")
            finally:
                lab.write_json_atomic(out / f"startup-{label}.json", lab.stop_memory_samplers(samplers))
            command([sys.executable, "bin/spark", "--cluster-config", config, "doctor", "--live"], label + "-doctor")
            # Native benchmark guards also cover functional traffic below.
            cluster = lab.spark.read_cluster(config)
            halted = []
            child = {}
            interrupted = []
            def halt(reason):
                if not halted:
                    halted.append(reason)
                    interrupted.append(time.monotonic())
                    if "process" in child and child["process"].poll() is None:
                        child["process"].send_signal(signal.SIGINT)
            watch = bench.MemoryWatch(nodes, float(cluster["host"]["memguard_gib"]) + 1, halt)
            conditions = {}
            current = {}
            try:
                watch.wait_ready()
                conditions["cooling"] = bench.cool_nodes(nodes, nodes["nodes"], 55, 600,
                                                          fan_control=bench.fan_control_enabled(cluster))
                conditions["fan_curve"] = bench.start_fan_curve(nodes, nodes["nodes"],
                                                                 fan_control=bench.fan_control_enabled(cluster))
                if bench.fan_curve_problems(conditions["fan_curve"]):
                    raise RuntimeError("functional gate requires the existing fan curve on all nodes")
                for name, script, extra in GATES:
                    if not owns(lab.read_hold(), args.window):
                        raise RuntimeError("window ownership changed")
                    log = out / f"{label}-{name}.jsonl"
                    argv = [sys.executable, script, lab.head_url(), *extra]
                    with log.open("w") as stream:
                        child["process"] = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        while child["process"].poll() is None:
                            try:
                                watch.check()
                            except Exception as error:
                                halt(str(error))
                            if interrupted and time.monotonic() - interrupted[0] > 10:
                                child["process"].terminate()
                            if interrupted and time.monotonic() - interrupted[0] > 20:
                                child["process"].kill()
                            time.sleep(1)
                        code = child["process"].returncode
                    watch.check()
                    if code or halted:
                        raise RuntimeError(f"{label}/{name}: failed or guard halted")
                    current[name] = gate_summary(log.read_text(), name)
                if signatures is not None and current != signatures:
                    raise RuntimeError("functional signatures changed between arms/boots")
                signatures = current
                lab.write_json_atomic(out / f"{label}-signatures.json", current)
            finally:
                if "process" in child and child["process"].poll() is None:
                    child["process"].terminate()
                    try:
                        child["process"].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child["process"].kill()
                        child["process"].wait()
                conditions["halted"] = halted
                conditions["thermal"] = watch.thermal_summary()
                conditions["memory"] = watch.close()
                conditions["fan_stop_failures"] = bench.stop_fan_curve(nodes, nodes["nodes"], conditions.get("fan_curve", []))
                lab.write_json_atomic(out / f"{label}-gate-conditions.json", conditions)
            for suite in suites:
                stem = Path(suite).stem
                output = out / label / stem
                argv = [sys.executable, "bin/spark", "--cluster-config", config, "bench", "--suite", suite,
                        "--compare", "none", "--label", label, "--output", str(output)]
                corpus = (out / arms[0]["label"] / stem / "model.json.gz" if args.stage == "screen"
                          else ROOT / PARENT / suite / "model.json.gz")
                if corpus.exists() and output != corpus.parent:
                    argv += ["--corpus", str(corpus)]
                with lab.Telemetry(str(output) + "-telemetry", config):
                    command(argv, label + "-" + stem)
            receipt.setdefault("completed_arms", []).append({"label": label, "config": config, "ended": lab.now()})
            save()
            active_arm = None
        success = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        receipt["completed"] = success
        if not success or not args.keep_open or args.stage == "restore" or lab.window_should_close(lab.read_hold() or {}):
            try:
                if not owns(lab.read_hold(), args.window):
                    raise RuntimeError("ownership changed; leave foreign hold alone")
                # Stop the currently labelled arm before its committed profile disappears.
                if last_config:
                    action(last_config, "stop", "--parallel", restoring=True)
                else:
                    action(RESTORE, "stop", "--parallel", restoring=True)
                action(RESTORE, "sync", "--revision", args.restore_commit, restoring=True)
                if not lab.window_close("verification budget finished"):
                    raise RuntimeError("service restoration failed")
                receipt["restored"] = True
            except BaseException as error:
                receipt["restore_error"] = f"{type(error).__name__}: {error}"
                success = False
        receipt["ended"] = lab.now(); save(); done.set()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
