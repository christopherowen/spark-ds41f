#!/usr/bin/env python3
"""Build and gate candidate-39 on idle DGX4 inside the head's owned window."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import bench, lab

PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-39/image-dgx4.json"
IMAGE = "spark-ds41f:candidate-39"
OUT = ROOT / "results/private/candidate-rounds/candidate-39-image"
NODES = {"ssh_user": "swank", "nodes": [{"name": "dgx4", "management_ip": "10.0.1.74", "head": True}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--window", required=True)
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"profile": PROFILE, "image": IMAGE, "node": "dgx4", "output": str(OUT)}))
        return 0
    if os.uname().nodename != "dgx4" or OUT.exists():
        raise SystemExit("wrong host or prior output exists; preserve evidence")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("build checkout is dirty")
    hold = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1", "cat ~/spark-hold.json"], text=True)
    if json.loads(hold).get("window") != args.window:
        raise SystemExit("head reservation changed")
    if subprocess.check_output(["docker", "ps", "-q"]).strip():
        raise SystemExit("DGX4 has active containers")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise SystemExit("DGX4 has active compute processes")
    gpu_lock = open("/tmp/spark-lab-gpu.lock", "w")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(parents=True)
    receipt = {"started": lab.now(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "steps": [], "conditions": {"halted": []}}
    proc = None
    def save():
        lab.write_json_atomic(OUT / "receipt.json", receipt)
    def halt(reason):
        if reason not in receipt["conditions"]["halted"]:
            receipt["conditions"]["halted"].append(reason)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
    def run(argv, name):
        nonlocal proc
        record = {"argv": argv, "started": lab.now(), "log": name + ".log"}
        receipt["steps"].append(record); save()
        with (OUT / record["log"]).open("w") as log:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            interrupted = None
            while proc.poll() is None:
                try:
                    watch.check()
                except Exception as error:
                    halt(str(error))
                if receipt["conditions"]["halted"]:
                    interrupted = interrupted or time.monotonic()
                    if time.monotonic() - interrupted > 60:
                        subprocess.run(["docker", "stop", "copy-default-c39-gate"], capture_output=True)
                        os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(1)
        record.update(exit=proc.returncode, ended=lab.now()); save()
        if proc.returncode or receipt["conditions"]["halted"]:
            raise RuntimeError(f"{name} exited {proc.returncode}; {receipt['conditions']['halted']}")
    watch = bench.MemoryWatch(NODES, 4, halt)
    code = 1
    try:
        watch.wait_ready()
        receipt["conditions"]["cooling"] = bench.cool_nodes(NODES, NODES["nodes"], 55, 600)
        receipt["conditions"]["fan_curve"] = bench.start_fan_curve(NODES, NODES["nodes"])
        problems = bench.fan_curve_problems(receipt["conditions"]["fan_curve"])
        if problems and not all(r["service"] == "absent" for r in receipt["conditions"]["fan_curve"]):
            raise RuntimeError(str(problems))
        for operation, extra in [("prepare", []), ("check", []), ("image", []), ("image", ["--apply"])]:
            run([sys.executable, "bin/spark", "--cluster-config", PROFILE, "build", operation, *extra], "build-" + operation + ("-apply" if extra else ""))
        receipt["image"] = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        expected = "31412339aafdd2c1392c7294975355dbc523c5e1"
        sources = [p for p in (ROOT / ".work/build").glob("*/src/vllm") if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == expected]
        if not sources:
            raise RuntimeError("canonical source checkout absent")
        manifest = OUT / "gate-manifest.json"
        bundle = OUT / "bundle"
        run([sys.executable, "scripts/release_tests.py", "gate", str(sources[0]), expected, str(manifest)], "gate-manifest")
        run([sys.executable, "scripts/release_tests.py", "bundle", str(manifest), IMAGE, str(bundle), "candidate-39"], "gate-bundle")
        candidate = json.loads((bundle / "candidate.json").read_text())
        lab.compile_cache(candidate).mkdir(parents=True, exist_ok=True)
        command = lab.bundle_command(bundle, candidate, OUT / "gate")
        command[2:2] = ["--name", "copy-default-c39-gate"]
        run(command, "release-tests")
        verdict = lab.verdict_from_lines((OUT / "release-tests.log").read_text().splitlines(), candidate)
        receipt["gate"] = verdict
        if not verdict["passed"]:
            raise RuntimeError("release-test verdict failed")
        receipt["passed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        receipt["conditions"]["thermal"] = watch.thermal_summary()
        receipt["conditions"]["memory"] = watch.close()
        receipt["conditions"]["fan_stop_failures"] = bench.stop_fan_curve(NODES, NODES["nodes"], receipt["conditions"].get("fan_curve", []))
        if receipt["conditions"]["fan_stop_failures"]:
            receipt["passed"] = False
            receipt["error"] = "fan policy restoration failed"
            code = 1
        receipt["ended"] = lab.now(); save()
        gpu_lock.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
