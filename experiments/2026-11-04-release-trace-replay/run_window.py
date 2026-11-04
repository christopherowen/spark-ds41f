#!/usr/bin/env python3
"""Run the approved four-point replay with the lab reservation and watchdog."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab
from spark_ds41f import bench

EXPERIMENT = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
RESTORE = f"{EXPERIMENT}/candidate-37.json"
CANDIDATE = f"{EXPERIMENT}/candidate-38.json"
OUTPUT = ROOT / "results/private/lab/c38-spaceport-tp4-window"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
MANAGEMENT_LOST = False


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    lab.write_json_atomic(path, value)


def cli(config, operation, name, *extra, mutate=False):
    global MANAGEMENT_LOST
    argv = [sys.executable, "bin/spark", "--cluster-config", config, operation, name, *extra]
    plan = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
    print(plan.stdout, flush=True)
    stem = f"{int(time.time())}-{operation}-{name}"
    save(OUTPUT / f"{stem}-plan.json", {"argv": argv, "exit": plan.returncode,
                                      "stdout": plan.stdout, "stderr": plan.stderr})
    if plan.returncode or "--apply would refuse" in plan.stdout or "shape:   CHANGED" in plan.stdout:
        raise RuntimeError(f"{operation} {name} plan refused")
    if mutate:
        result = subprocess.run([*argv, "--apply"], cwd=ROOT, text=True, capture_output=True)
        print(result.stdout, flush=True)
        print(result.stderr, file=sys.stderr, flush=True)
        save(OUTPUT / f"{stem}-apply.json", {"argv": [*argv, "--apply"],
             "exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            detail = (result.stdout + result.stderr).lower()
            MANAGEMENT_LOST = any(text in detail for text in
                                  ("ssh: connect", "cannot reach", "connection reset", "broken pipe"))
            raise RuntimeError(f"{operation} {name} failed ({result.returncode})")


def checked_run(command, log):
    code = lab.run_logged(command, str(log.relative_to(ROOT)))
    if code:
        raise RuntimeError(f"command exited {code}; see {log}")


def measured_point(step, nodes):
    out = ROOT / step["out"]
    out.mkdir(parents=True, exist_ok=True)
    conditions = {"halted": [], "fan_policy": "existing site policy; missing services remain recorded"}
    process = None
    interrupted_at = None

    def halt(reason):
        nonlocal interrupted_at
        if not conditions["halted"]:
            conditions["halted"].append(reason)
            interrupted_at = time.monotonic()
            print(f"HALT: {reason}", flush=True)
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)

    watch = bench.MemoryWatch(nodes, 4, halt)
    try:
        watch.wait_ready()
        conditions["cooling"] = bench.cool_nodes(nodes, nodes["nodes"], 55, 600)
        # All four boards lack the curve: preserve that fact as bench does.
        conditions["fan_curve"] = bench.start_fan_curve(nodes, nodes["nodes"])
        problems = bench.fan_curve_problems(conditions["fan_curve"])
        if problems and not all(row["service"] == "absent" for row in conditions["fan_curve"]):
            raise RuntimeError("inconsistent fan policy: " + "; ".join(problems))
        head = next(node for node in nodes["nodes"] if node["head"])
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))) / 1048576
        conditions["client_admission_gib"] = available
        if available < 20:
            raise RuntimeError(f"{head['name']}: {available:.2f} GiB available; client needs 20 GiB")
        clean = {**step, "seconds": 15, "out": f"{step['out']}/clean.json"}
        checked_run(lab.clean_command(clean), out / "clean.log")
        problems = lab.clean_problems(json.loads((out / "clean.json").read_text()))
        if problems:
            raise RuntimeError("foreign traffic: " + "; ".join(problems))
        watch.check()
        if conditions["halted"]:
            raise RuntimeError(conditions["halted"][0])
        command = lab.agentx_command(step)
        save(out / "command.json", command)
        with lab.Telemetry(f"{step['out']}/point", CANDIDATE), (out / "lab.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    watch.check()
                except Exception as error:
                    halt(str(error))
                if conditions["halted"]:
                    if interrupted_at is None:
                        interrupted_at = time.monotonic()
                    process.send_signal(signal.SIGINT) if time.monotonic() - interrupted_at < 1 else None
                    if time.monotonic() - interrupted_at > 120:
                        process.terminate()
                time.sleep(1)
        if process.returncode or conditions["halted"]:
            raise RuntimeError(f"{step['label']}: exit {process.returncode}; {conditions['halted']}")
        result = json.loads((out / "agentx.json").read_text())
        if not result.get("harness", {}).get("submission_valid"):
            raise RuntimeError(f"{step['label']}: submission invalid")
        return result
    finally:
        conditions["thermal"] = watch.thermal_summary()
        conditions["memory"] = watch.close()
        conditions["fan_stop_failures"] = bench.stop_fan_curve(nodes, nodes["nodes"], conditions.get("fan_curve", []))
        save(out / "conditions.json", conditions)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--attempt", type=int, default=1, help="distinct window receipts; never overwrite a failed attempt")
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be positive")
    if args.attempt > 1:
        OUTPUT = OUTPUT.with_name(f"{OUTPUT.name}-attempt-{args.attempt}")
    lab.PRODUCTION_CONFIG = RESTORE
    specs = [json.loads((ROOT / EXPERIMENT / f"lab-boot-{boot}.json").read_text()) for boot in "ab"]
    points = [step for spec in specs for job in spec["jobs"]
              for step in lab.agentx_steps(spec, job) if step["kind"] == "agentx"]
    if not args.apply:
        print(json.dumps({"restore": RESTORE, "image": IMAGE, "points": points,
                          "output": str(OUTPUT), "attempt": args.attempt,
                          "memory_floor_gib": 4, "client_admission_gib": 20,
                          "cool_below_c": 55, "apply": False}, indent=2))
        return 0
    if os.uname().nodename != "spark-0009":
        raise SystemExit("this window is authorized only on sp1-sp4, controlled from sp1")
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists; preserve prior evidence and choose a new run")
    for point in points:
        if (ROOT / point["out"]).exists():
            raise SystemExit(f"{point['out']} already exists")
    problems = lab.agentx_problems(points[0]) + lab.arm_problems(CANDIDATE) + lab.arm_problems(RESTORE)
    if problems:
        raise SystemExit("; ".join(problems))
    nodes = lab.nodes_config()
    for node in nodes["nodes"]:
        image = lab.spark.run_ssh(nodes, node, "docker", "image", "inspect", "spark-ds41f:candidate-38", "--format", "{{.Id}}")
        if image.returncode or image.stdout.strip() != IMAGE:
            raise SystemExit(f"{node['name']}: candidate image mismatch")
    OUTPUT.mkdir(parents=True)
    receipt = {"started": lab.now(), "points": [], "restored": False,
               "attempt": args.attempt,
               "deployment_commit": lab.checkout_commit(), "authorization": "owner approved full TP4 test window"}
    lab.window_open(0, "candidate-38 TP4 trace replay")
    done = threading.Event()
    current = {"note": "preparing pinned corpus"}

    def heartbeat():
        while not done.wait(30):
            lab.beat(current["note"])

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    code = 1
    try:
        if not lab.agentx_prepared(points[0]):
            cli(RESTORE, "cluster", "stop", mutate=True)
            checked_run(lab.agentx_prepare_command(points[0]), OUTPUT / "prepare.log")
            if not lab.agentx_prepared(points[0]):
                raise RuntimeError("corpus preparation produced no valid marker")
        for point in points:
            hold = lab.read_hold()
            if not lab.hold_is_ours(hold) or lab.window_should_close(hold):
                raise RuntimeError("reservation lost or another user requested the cluster")
            current["note"] = point["label"]
            lab.beat(current["note"])
            samplers = lab.start_memory_samplers(OUTPUT / f"boot-{point['label']}")
            try:
                cli(CANDIDATE, "cluster", "start", "--replace", mutate=True)
            finally:
                save(OUTPUT / f"boot-{point['label']}-memory.json", lab.stop_memory_samplers(samplers))
            result = measured_point(point, nodes)
            receipt["points"].append({"label": point["label"], "result": result})
            save(OUTPUT / "receipt.json", receipt)
            checked_run([sys.executable, "bin/spark", "--cluster-config", CANDIDATE, "doctor", "--live"],
                        OUTPUT / f"{point['label']}-doctor.log")
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
        lab.save_engine_logs(str((OUTPUT / "failure").relative_to(ROOT)), config=CANDIDATE)
    finally:
        current["note"] = "restoring candidate-37"
        lab.beat(current["note"])
        try:
            unreachable = [node["name"] for node in nodes["nodes"]
                           if lab.spark.run_ssh(nodes, node, "true").returncode]
            if MANAGEMENT_LOST or unreachable:
                # A lost management path requires bounded isolation and renewed
                # hardware authorization, even if a later SSH happens to work.
                hold = lab.read_hold()
                hold["holder"] = "tp4-trace-replay-hardware-review"
                hold["rule"] = "management path lost; no launch until isolated and explicitly authorized"
                lab.write_json_atomic(lab.HOLD, hold)
                receipt["management_unreachable"] = unreachable
                raise RuntimeError("management path lost; reservation retained, watchdog launch disabled")
            cli(RESTORE, "cluster", "start", "--replace", mutate=True)
            if not lab.production_live():
                raise RuntimeError("candidate-37 live verification failed")
            receipt["restored"] = lab.window_close("approved trace replay completed")
        except BaseException as error:
            receipt["restore_error"] = f"{type(error).__name__}: {error}"
            print(receipt["restore_error"], flush=True)
            code = 1
        finally:
            done.set()
            thread.join(timeout=2)
            receipt["ended"] = lab.now()
            save(OUTPUT / "receipt.json", receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
