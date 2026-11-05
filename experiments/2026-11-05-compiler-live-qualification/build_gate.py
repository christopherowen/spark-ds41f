#!/usr/bin/env python3
"""Build and gate candidate-42 on idle DGX4 inside the head's owned window."""
import argparse
import hashlib
import importlib.util
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import bench, lab, config as spark_config

PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-42/image-dgx4.json"
IMAGE = "spark-ds41f:candidate-42"
OUT = ROOT / "results/private/candidate-rounds/candidate-42-image"
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
    existing = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if existing.returncode == 0:
        raise SystemExit("candidate image tag already exists; preserve it and review before a build")
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
        if receipt["conditions"]["halted"]:
            raise RuntimeError("guard halted before " + name)
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
                        subprocess.run(["docker", "stop", "compiler-c42-gate", "compiler-c42-initcheck", "compiler-c42-features-1", "compiler-c42-features-2"], capture_output=True)
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
        configured = spark_config.read_cluster(PROFILE)
        receipt["source"] = {"vllm_head": "5f926df6d3126266a9c88c82c06544060008c33c",
                             "expected_labels": configured["container"]["expected_labels"],
                             "tilelang_head": "454202b2ba8d0cf10b8caf434a956755e039483f"}
        receipt["image"] = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        expected = "5f926df6d3126266a9c88c82c06544060008c33c"
        sources = [p for p in (ROOT / ".work/build").glob("*/src/vllm") if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == expected]
        if not sources:
            raise RuntimeError("canonical source checkout absent")
        manifest = OUT / "gate-manifest.json"
        bundle = OUT / "bundle"
        run([sys.executable, "scripts/release_tests.py", "gate", str(sources[0]), expected, str(manifest)], "gate-manifest")
        run([sys.executable, "scripts/release_tests.py", "bundle", str(manifest), IMAGE, str(bundle), "candidate-42"], "gate-bundle")
        candidate = json.loads((bundle / "candidate.json").read_text())
        lab.compile_cache(candidate).mkdir(parents=True, exist_ok=True)
        command = lab.bundle_command(bundle, candidate, OUT / "gate")
        command[2:2] = ["--name", "compiler-c42-gate"]
        run(command, "release-tests")
        verdict = lab.verdict_from_lines((OUT / "release-tests.log").read_text().splitlines(), candidate)
        receipt["gate"] = verdict
        receipt["gate_native_results"] = str(lab.compile_cache(candidate) / "release-tests/candidate-42")
        shutil.copytree(receipt["gate_native_results"], OUT / "gate-results")
        if not verdict["passed"]:
            raise RuntimeError("release-test verdict failed")
        qualification = ROOT / "experiments/2026-11-05-compiler-live-qualification"
        source_hash = json.loads((qualification / "runs/preflight/topk-source.json").read_text())["sha256"]
        repro = ROOT / "experiments/2026-11-04-topk-live-reads/bundle/topk_reads.py"
        sanitizer = Path("/usr/local/cuda-13.0/compute-sanitizer")
        if not (sanitizer / "compute-sanitizer").is_file():
            raise RuntimeError("recorded Compute Sanitizer installation is absent")
        initcheck = ["docker", "run", "--name", "compiler-c42-initcheck", "--rm", "--gpus", "all",
                     "--ipc=host", "--ulimit=core=0", "--security-opt=seccomp=unconfined",
                     "--ulimit=memlock=-1:-1", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1",
                     "-e", "DO_NOT_TRACK=1", "-e", "PYTORCH_NO_CUDA_MEMORY_CACHING=1",
                     "-e", "TILELANG_CACHE_DIR=/c/tilelang-compiler-c42",
                     "-e", "EXPECTED_TOPK_SOURCE_SHA256=" + source_hash,
                     "-v", str(sanitizer) + ":/opt/compute-sanitizer:ro",
                     "-v", str(repro) + ":/qualification/reproducer.py:ro",
                     "-v", str(qualification / "topk_initcheck.py") + ":/qualification/check.py:ro",
                     "-v", str(lab.compile_cache(candidate)) + ":/c",
                     "-w", "/opt/spark-ds41f/candidate/vllm",
                     "--entrypoint", "/opt/compute-sanitizer/compute-sanitizer", receipt["image"],
                     "--tool", "initcheck", "--error-exitcode", "1", "--print-limit", "4",
                     "python3", "/qualification/check.py"]
        run(initcheck, "topk-initcheck")
        output = (OUT / "topk-initcheck.log").read_text()
        if "ERROR SUMMARY: 0 errors" not in output or "exact top-k output" not in output:
            raise RuntimeError("strict top-k initcheck lacks both zero-error and exact-output evidence")
        receipt["topk_initcheck"] = {"passed": True, "source_sha256": source_hash,
                                     "log_sha256": hashlib.sha256(output.encode()).hexdigest()}
        compiler_head = "454202b2ba8d0cf10b8caf434a956755e039483f"
        compiler_sources = [p for p in (ROOT / ".work/build").glob("*/src/tilelang")
                            if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip() == compiler_head]
        if not compiler_sources:
            raise RuntimeError("canonical combined compiler source is absent")
        run([sys.executable, str(qualification / "compiler_gate.py"), "--image", receipt["image"],
             "--source", str(compiler_sources[0]), "--output", str(OUT / "compiler-features")], "compiler-features")
        receipt["compiler_features"] = json.loads((OUT / "compiler-features/receipt.json").read_text())
        if receipt["compiler_features"].get("passed") is not True:
            raise RuntimeError("combined installed compiler feature gates failed")
        receipt["passed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        # Independent final cleanup also covers interruption of the gate driver itself.
        for ownership in (OUT / "compiler-features").glob("boot*/ownership.json"):
            try:
                owned = json.loads(ownership.read_text())
                if (owned["name"] not in ("compiler-c42-features-1", "compiler-c42-features-2")
                        or owned["image"] != receipt.get("image")):
                    raise RuntimeError("owned feature-container identity differs")
                spec = importlib.util.spec_from_file_location("compiler_cleanup", qualification / "compiler_gate.py")
                cleanup = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cleanup)
                cleanup.cleanup_owned_container(ownership.parent, owned["image"], owned["token"],
                                                owned["name"], suffix="-builder-final")
            except BaseException as error:
                receipt["passed"] = False
                receipt["conditions"]["halted"].append("owned gate cleanup unverified")
                receipt["error"] = "cannot reach or validate owned gate container cleanup: " + str(error)
                print(receipt["error"], flush=True)
                code = 1
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
