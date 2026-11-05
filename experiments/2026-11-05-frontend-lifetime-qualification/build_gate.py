#!/usr/bin/env python3
"""Build and gate candidate-43 on idle DGX4 inside the head's owned window."""
import argparse
import hashlib
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

PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-43/image-dgx4.json"
IMAGE = "spark-ds41f:candidate-43"
OUT = ROOT / "results/private/candidate-rounds/candidate-43-image"
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
    def run(argv, name, retain_failed_verdict=False):
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
                        subprocess.run(["docker", "stop", "frontend-c43-gate", "frontend-c43-initcheck"], capture_output=True)
                        os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(1)
        record.update(exit=proc.returncode, ended=lab.now()); save()
        if receipt["conditions"]["halted"] or (proc.returncode and not retain_failed_verdict):
            raise RuntimeError(f"{name} exited {proc.returncode}; {receipt['conditions']['halted']}")
        return proc.returncode
    watch = bench.MemoryWatch(NODES, 4, halt)
    code = 1
    try:
        watch.wait_ready()
        receipt["conditions"]["cooling"] = bench.cool_nodes(NODES, NODES["nodes"], 55, 600)
        receipt["conditions"]["fan_curve"] = bench.start_fan_curve(NODES, NODES["nodes"])
        problems = bench.fan_curve_problems(receipt["conditions"]["fan_curve"])
        if problems:
            raise RuntimeError(str(problems))
        for operation, extra in [("prepare", []), ("check", []), ("image", []), ("image", ["--apply"])]:
            run([sys.executable, "bin/spark", "--cluster-config", PROFILE, "build", operation, *extra], "build-" + operation + ("-apply" if extra else ""))
        configured = spark_config.read_cluster(PROFILE)
        receipt["source"] = {"vllm_head": "99029e6cbdb966ff198dd13902a0825e8d507e27",
                             "expected_labels": configured["container"]["expected_labels"],
                             "tilelang_head": "a0e66d71fe7617216f2de5d86c3b088595367387"}
        receipt["image"] = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        expected = "99029e6cbdb966ff198dd13902a0825e8d507e27"
        sources = [p for p in (ROOT / ".work/build").glob("*/src/vllm") if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == expected]
        if not sources:
            raise RuntimeError("canonical source checkout absent")
        manifest = OUT / "gate-manifest.json"
        bundle = OUT / "bundle"
        run([sys.executable, "scripts/release_tests.py", "gate", str(sources[0]), expected, str(manifest), str(ROOT / "experiments/2026-11-05-frontend-lifetime-qualification/release-tests.json")], "gate-manifest")
        run([sys.executable, "scripts/release_tests.py", "bundle", str(manifest), IMAGE, str(bundle), "candidate-43"], "gate-bundle")
        manifest_data = json.loads(manifest.read_text())
        required = {"tests/v1/engine/test_request_submission_cleanup.py", "tests/v1/engine/test_utility_call_cleanup.py"}
        if len(manifest_data["files"]) != 138 or not required <= set(manifest_data["files"]):
            raise RuntimeError("full legacy and frontend regression gate manifest is incomplete")
        receipt["gate_manifest"] = {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                     "files": len(manifest_data["files"]), "required_frontend_files": sorted(required)}
        candidate = json.loads((bundle / "candidate.json").read_text())
        # Isolate the entire gate cache, including pytest temporary caches and
        # loader fixtures; preserve C42's shared-cache failure artifacts intact.
        cache = Path.home() / ".cache/spark-lab/qualification" / ("candidate-43-" + receipt["image"].split(":", 1)[1])
        if cache.exists():
            raise RuntimeError("new-image qualification cache already exists; preserve it and review")
        generated = (bundle / "candidate.json").read_bytes()
        (OUT / "gate-candidate-generated.json").write_bytes(generated)
        candidate["cache"] = str(cache)
        lab.write_json_atomic(bundle / "candidate.json", candidate)
        receipt["release_cache"] = {"host_root": str(cache), "absent_before": True,
                                    "image": receipt["image"],
                                    "generated_bundle_sha256": hashlib.sha256(generated).hexdigest(),
                                    "actual_bundle_sha256": hashlib.sha256((bundle / "candidate.json").read_bytes()).hexdigest()}
        save()
        lab.compile_cache(candidate).mkdir(parents=True, exist_ok=False)
        command = lab.bundle_command(bundle, candidate, OUT / "gate")
        command[2:2] = ["--name", "frontend-c43-gate"]
        gate_exit = run(command, "release-tests", retain_failed_verdict=True)
        verdict = lab.verdict_from_lines((OUT / "release-tests.log").read_text().splitlines(), candidate)
        receipt["gate"] = verdict
        receipt["gate_native_results"] = str(lab.compile_cache(candidate) / "release-tests/candidate-43")
        if Path(receipt["gate_native_results"]).is_dir():
            shutil.copytree(receipt["gate_native_results"], OUT / "gate-results")
        save()
        if gate_exit or not verdict["passed"] or not (OUT / "gate-results").is_dir():
            raise RuntimeError(f"release-test gate failed; exit={gate_exit}, verdict={verdict}")
        qualification = ROOT / "experiments/2026-11-05-frontend-lifetime-qualification"
        source_hash = json.loads((qualification / "runs/preflight/topk-source.json").read_text())["sha256"]
        repro = ROOT / "experiments/2026-11-04-topk-live-reads/bundle/topk_reads.py"
        sanitizer = Path("/usr/local/cuda-13.0/compute-sanitizer")
        if not (sanitizer / "compute-sanitizer").is_file():
            raise RuntimeError("recorded Compute Sanitizer installation is absent")
        initcheck = ["docker", "run", "--name", "frontend-c43-initcheck", "--rm", "--gpus", "all",
                     "--ipc=host", "--ulimit=core=0", "--security-opt=seccomp=unconfined",
                     "--ulimit=memlock=-1:-1", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1",
                     "-e", "DO_NOT_TRACK=1", "-e", "PYTORCH_NO_CUDA_MEMORY_CACHING=1",
                     "-e", "TILELANG_CACHE_DIR=/c/tilelang-frontend-c43",
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
