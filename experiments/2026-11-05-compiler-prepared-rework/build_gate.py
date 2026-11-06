#!/usr/bin/env python3
"""Draft C45 build/gate control; apply is blocked pending the named integrations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_ds41f import bench, lab, config as spark_config
import c45_policy as policy
import c45_native
from c45_build_control import BuildControl
from c45_management import ManagementLatch
import compiler_gate

PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-45/image-dgx4.json"
IMAGE = "spark-ds41f:candidate-45"
OUT = ROOT / "results/private/candidate-rounds/candidate-45-image"
NODES = {"ssh_user": "swank", "nodes": [{"name": "dgx4", "management_ip": "10.0.1.74", "head": True}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--window", required=True)
    parser.add_argument("--holder")
    parser.add_argument("--owner-token")
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"profile": PROFILE, "image": IMAGE, "node": "dgx4", "output": str(OUT),
                          "launch_blockers": policy.launch_blockers(),
                          "source_prepare_limits": policy.PREP_LIMITS,
                          "release_gate_limits": policy.GATE_LIMITS,
                          "initcheck_limits": policy.INITCHECK_LIMITS}))
        return 0
    policy.require_launch_ready()
    if os.uname().nodename != "dgx4" or OUT.exists():
        raise SystemExit("wrong host or prior output exists; preserve evidence")
    os.environ["SPARK_DS41F_STRICT_TELEMETRY"] = "1"
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise SystemExit("build checkout is dirty")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    management = ManagementLatch()
    def check_claims():
        head = management.ssh("dgx1", ["cat", "~/spark-hold.json"], "build-head-lease", timeout=15)
        policy.assert_owned(json.loads(head), args.window, args.holder, args.owner_token)
        local = subprocess.check_output([sys.executable, "-c", policy.CLAIM_PROBE], text=True, timeout=10)
        records = policy.validate_local_records(json.loads(local), args.owner_token, args.window)
        return {"passed": True, "commit": commit, "window": args.window,
                "holder": args.holder, "owner_token": args.owner_token, "local": records}
    initial_claims = check_claims()
    if subprocess.check_output(["docker", "ps", "-q"]).strip():
        raise SystemExit("DGX4 has active containers")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise SystemExit("DGX4 has active compute processes")
    existing = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True)
    if existing.returncode == 0:
        raise SystemExit("candidate image tag already exists; preserve it and review before a build")
    gpu_lock = open("/tmp/spark-lab-gpu.lock", "a+")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpu_locks = []
    try:
        for path in (Path.home() / "spark-cpu-work.lock", Path.home() / "spark-lab/offline-cpu.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            cpu_locks.append(handle)
    except BaseException:
        for handle in cpu_locks:
            handle.close()
        gpu_lock.close()
        raise
    OUT.mkdir(parents=True)
    receipt = {"started": lab.now(), "commit": commit,
        "window": args.window, "holder": args.holder, "owner_token": args.owner_token,
        "steps": [], "conditions": {"halted": []}, "initial_local_claims": initial_claims,
        "resource_limits": {"source_prepare": policy.PREP_LIMITS, "release_gate": policy.GATE_LIMITS,
                            "topk_initcheck": policy.INITCHECK_LIMITS}}
    worker = None
    control = None
    def save():
        receipt.update(management.snapshot())
        lab.write_json_atomic(OUT / "receipt.json", receipt)
    management.changed = save
    def halt(reason):
        if control is not None:
            control.halt(reason)
        elif str(reason) not in receipt["conditions"]["halted"]:
            receipt["conditions"]["halted"].append(str(reason))
            save()
    watch = bench.MemoryWatch(NODES, 4, halt)
    locks = {"gpu": gpu_lock, "cpu-work": cpu_locks[0], "cpu-offline": cpu_locks[1]}
    control = BuildControl(commit=commit, window=args.window, holder=args.holder, token=args.owner_token,
                           locks=locks, watch=watch, scope_check=check_claims, management=management,
                           receipt=receipt, output=OUT, save=save)
    code = 1
    try:
        watch.wait_ready()
        receipt["conditions"]["cooling"] = bench.cool_nodes(NODES, NODES["nodes"], 55, 600)
        receipt["conditions"]["fan_curve"] = bench.start_fan_curve(NODES, NODES["nodes"])
        problems = bench.fan_curve_problems(receipt["conditions"]["fan_curve"])
        if problems:
            raise RuntimeError(str(problems))
        for operation in ("prepare", "check"):
            source_command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                "-p", "CPUQuota=100%", "-p", "MemoryMax=8G", "-p", "MemorySwapMax=0", "-p", "TasksMax=64",
                sys.executable, str(Path(__file__).with_name("build_resources.py")), "source",
                "--profile", PROFILE, "--operation", operation, "--apply"]
            control.run(source_command, "build-" + operation)
        # Fresh worker creation/inspection belongs inside the owned window,
        # after existing idle/cold guards, never as a prerequisite fresh lease.
        worker = policy.create_validate_owned_worker(policy.load_plan()["builder_worker"],
            OUT, receipt["commit"], args.window, args.holder, args.owner_token)
        receipt["builder_worker"] = worker
        if worker.get("passed") is not True:
            raise RuntimeError("owned immutable BuildKit worker did not validate")
        image_env = ["env", "BUILDX_BUILDER=" + worker["builder"], "OMP_NUM_THREADS=1",
                     "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "MAX_JOBS=2"]
        for extra in ([], ["--apply"]):
            control.run([*image_env, sys.executable, "bin/spark", "--cluster-config", PROFILE,
                         "build", "image", "--max-compile-jobs", "2", *extra],
                        "build-image" + ("-apply" if extra else ""))
        receipt["builder_worker_stop"] = policy.stop_owned_worker(worker, OUT, args.owner_token)
        if receipt["builder_worker_stop"].get("passed") is not True:
            raise RuntimeError("own BuildKit worker did not stop before GPU gates")
        worker = None
        configured = spark_config.read_cluster(PROFILE)
        receipt["source"] = {"vllm_head": compiler_gate.VLLM_HEAD,
                             "vllm_tree": compiler_gate.VLLM_TREE,
                             "expected_labels": configured["container"]["expected_labels"],
                             "patch_count": 574,
                             "patchset_sha256": "c7cdd12575661dc07642c9b4207a224f2b39bd913a259336dd28ee2af945390b",
                             "tilelang_head": compiler_gate.HEAD,
                             "tilelang_tree": compiler_gate.TREE,
                             "tilelang_version": compiler_gate.VERSION,
                             "tilelang_patchset_sha256": compiler_gate.PATCHSET}
        receipt["image"] = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        expected = compiler_gate.VLLM_HEAD
        sources = [p for p in (ROOT / ".work/build").glob("*/src/vllm") if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == expected]
        if not sources:
            raise RuntimeError("canonical source checkout absent")
        manifest = OUT / "gate-manifest.json"
        bundle = OUT / "bundle"
        control.run([sys.executable, "scripts/release_tests.py", "gate", str(sources[0]), expected, str(manifest)], "gate-manifest")
        control.run([sys.executable, "scripts/release_tests.py", "bundle", str(manifest), IMAGE, str(bundle), "candidate-45"], "gate-bundle")
        manifest_data = json.loads(manifest.read_text())
        required_iket = "tests/kernels/test_deepseek_v41_tilelang_iket.py"
        if len(manifest_data["files"]) != compiler_gate.RELEASE_FILES or required_iket not in manifest_data["files"]:
            raise RuntimeError("full 136-file C40 release and IKET regression gate manifest is incomplete")
        receipt["gate_manifest"] = {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                                     "files": len(manifest_data["files"]), "required_iket_file": required_iket}
        candidate = json.loads((bundle / "candidate.json").read_text())
        # Isolate the entire gate cache, including pytest temporary caches and
        # loader fixtures; preserve C42's shared-cache failure artifacts intact.
        cache = Path.home() / ".cache/spark-lab/qualification" / ("candidate-45-" + receipt["image"].split(":", 1)[1])
        if cache.exists():
            raise RuntimeError("new-image qualification cache already exists; preserve it and review")
        generated = (bundle / "candidate.json").read_bytes()
        (OUT / "gate-candidate-generated.json").write_bytes(generated)
        candidate["cache"] = str(cache)
        lab.write_json_atomic(bundle / "candidate.json", candidate)
        lab.compile_cache(candidate).mkdir(parents=True, exist_ok=False)
        host_tilelang_root = lab.compile_cache(candidate) / "tilelang-release-tests"
        namespace_absent = not os.path.lexists(host_tilelang_root / compiler_gate.NAMESPACE)
        if not namespace_absent:
            raise RuntimeError("builder release-gate compiler namespace already exists; preserve and refuse")
        receipt["release_cache"] = {"host_root": str(cache), "host_tilelang_root": str(host_tilelang_root),
                                    "container_root": candidate["env"]["TILELANG_CACHE_DIR"],
                                    "version": compiler_gate.VERSION, "namespace": compiler_gate.NAMESPACE,
                                    "absent_before": True, "namespace_absent_before_import": namespace_absent,
                                    "image": receipt["image"],
                                    "generated_bundle_sha256": hashlib.sha256(generated).hexdigest(),
                                    "actual_bundle_sha256": hashlib.sha256((bundle / "candidate.json").read_bytes()).hexdigest()}
        save()
        gate_token = uuid.uuid4().hex
        control.register_container("compiler-c45-gate", gate_token, receipt["image"])
        command = lab.bundle_command(bundle, candidate, OUT / "gate")
        command[2:2] = ["--name", "compiler-c45-gate", "--label", "spark.qualification-token=" + gate_token,
                        *policy.container_limits("release")]
        gate_exit = control.run(command, "release-tests", retain_failed_verdict=True).returncode
        verdict = lab.verdict_from_lines((OUT / "release-tests.log").read_text().splitlines(), candidate)
        receipt["gate"] = verdict
        receipt["gate_native_results"] = str(lab.compile_cache(candidate) / "release-tests/candidate-45")
        if Path(receipt["gate_native_results"]).is_dir():
            shutil.copytree(receipt["gate_native_results"], OUT / "gate-results")
        native = OUT / "gate-results/results.jsonl"
        receipt["native_gate_audit"] = (policy.audit_native_files(manifest_data, native.read_text())
            if native.is_file() else {"passed": False, "error": "native per-file outcomes are missing"})
        receipt["native_regression_audit"] = (c45_native.audit(OUT / "gate-results")
            if native.is_file() else {"passed": False, "error": "native regression outcomes missing"})
        save()
        if (gate_exit or not verdict["passed"] or not receipt["native_gate_audit"]["passed"]
                or not receipt["native_regression_audit"]["passed"]):
            raise RuntimeError(f"release-test gate failed; exit={gate_exit}, verdict={verdict}")
        qualification = ROOT / "experiments/2026-11-05-compiler-prepared-rework"
        source_hash = compiler_gate.TOPK_SHA
        repro = ROOT / "experiments/2026-11-04-topk-live-reads/bundle/topk_reads.py"
        sanitizer = Path("/usr/local/cuda-13.0/compute-sanitizer")
        if not (sanitizer / "compute-sanitizer").is_file():
            raise RuntimeError("recorded Compute Sanitizer installation is absent")
        initcheck_token = uuid.uuid4().hex
        control.register_container("compiler-c45-initcheck", initcheck_token, receipt["image"])
        initcheck = ["docker", "run", "--name", "compiler-c45-initcheck",
                     "--label", "spark.qualification-token=" + initcheck_token, "--rm", "--gpus", "all",
                     *policy.container_limits("initcheck"),
                     "--ipc=host", "--ulimit=core=0", "--security-opt=seccomp=unconfined",
                     "--ulimit=memlock=-1:-1", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1",
                     "-e", "DO_NOT_TRACK=1", "-e", "PYTORCH_NO_CUDA_MEMORY_CACHING=1",
                     "-e", "TILELANG_CACHE_DIR=/c/tilelang-compiler-c45",
                     "-e", "EXPECTED_TOPK_SOURCE_SHA256=" + source_hash,
                     "-v", str(sanitizer) + ":/opt/compute-sanitizer:ro",
                     "-v", str(repro) + ":/qualification/reproducer.py:ro",
                     "-v", str(qualification / "topk_initcheck.py") + ":/qualification/check.py:ro",
                     "-v", str(lab.compile_cache(candidate)) + ":/c",
                     "-w", "/opt/spark-ds41f/candidate/vllm",
                     "--entrypoint", "/opt/compute-sanitizer/compute-sanitizer", receipt["image"],
                     "--tool", "initcheck", "--error-exitcode", "1", "--print-limit", "4",
                     "python3", "/qualification/check.py"]
        control.run(initcheck, "topk-initcheck")
        output = (OUT / "topk-initcheck.log").read_text()
        if "ERROR SUMMARY: 0 errors" not in output or "exact top-k output" not in output:
            raise RuntimeError("strict top-k initcheck lacks both zero-error and exact-output evidence")
        receipt["topk_initcheck"] = {"passed": True, "source_sha256": source_hash,
                                     "log_sha256": hashlib.sha256(output.encode()).hexdigest()}
        compiler_sources = [p for p in (ROOT / ".work/build").glob("*/src/tilelang")
                            if subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip() == compiler_gate.HEAD]
        if not compiler_sources:
            raise RuntimeError("canonical C45 TileLang source checkout is absent")
        binding = compiler_gate.source_binding(compiler_sources[0], compiler_gate.load(qualification / "gpu-gate-plan.json"))
        compiler_gate.save(OUT / "source-binding.json", binding)
        receipt["source_binding"] = binding
        identity_token = uuid.uuid4().hex
        control.register_container("compiler-c45-identity", identity_token, receipt["image"])
        identity_cmd = ["docker", "run", "--name", "compiler-c45-identity",
                        "--label", "spark.qualification-token=" + identity_token, "--rm", "--gpus", "all",
                        "--network", "none", "--ipc=host", *policy.container_limits("identity"),
                        "-e", "TILELANG_CACHE_DIR=" + receipt["release_cache"]["container_root"],
                        "-e", "DO_NOT_TRACK=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                        "-v", str((OUT / "source-binding.json").resolve()) + ":/packet/source-binding.json:ro",
                        "-v", str(qualification / "compiler_gate.py") + ":/packet/compiler_gate.py:ro",
                        "-v", str(qualification / "gpu-gate-plan.json") + ":/packet/gpu-gate-plan.json:ro",
                        "-v", str(OUT.resolve()) + ":/out", "-w", "/out", "--entrypoint", "python3",
                        receipt["image"], "/packet/compiler_gate.py", "--identity-only",
                        "--binding", "/packet/source-binding.json",
                        "--output", "/out/installed-compiler-identity.json"]
        control.run(identity_cmd, "compiler-identity")
        receipt["installed_compiler_identity"] = json.loads((OUT / "installed-compiler-identity.json").read_text())
        authority_path = OUT / "pre-feature-authority.json"
        authority_payload = {**receipt, "schema": "c45-compiler-feature-authority/1",
                             "candidate": 45, "stage": "release-and-topk-passed-features-pending",
                             "passed": True, "error": None}
        compiler_gate.save(authority_path, authority_payload)
        authority_sha = compiler_gate.sha(authority_path)
        receipt["pre_feature_authority"] = {"path": str(authority_path), "sha256": authority_sha}
        save()
        compiler_gate.host(OUT / "compiler-features", compiler_sources[0],
                           authority_path, authority_sha, control=control)
        receipt["compiler_features"] = json.loads((OUT / "compiler-features/receipt.json").read_text())
        if receipt["compiler_features"].get("passed") is not True:
            raise RuntimeError("combined installed compiler feature gates failed")
        receipt["conditions"]["thermal"] = watch.thermal_summary()
        with watch.lock:
            receipt["conditions"]["memory"] = {
                name: {"min_available_gib": state.get("min_available_kib", 0) / 1048576,
                       "swap_growth_gib": (state.get("swap_max_kib", 0) - state.get("swap_start_kib", 0)) / 1048576}
                for name, state in watch.state.items()}
        # Fan cleanup is checked in finally; all other complete guard evidence
        # must already exist before the completed image can be accepted.
        snapshot = {**receipt["conditions"], "fan_stop_failures": []}
        if not c45_native.conditions_pass(snapshot, ["dgx4"], 4):
            raise RuntimeError("DGX4 memory/thermal guard evidence is incomplete or adverse: "
                               + json.dumps(receipt["conditions"], sort_keys=True))
        receipt["passed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        if control is not None:
            control.active = False
        if worker is not None:
            try:
                receipt["builder_worker_stop_after_failure"] = policy.stop_owned_worker(worker, OUT, args.owner_token)
                if receipt["builder_worker_stop_after_failure"].get("passed") is not True:
                    raise RuntimeError("owned worker stop did not pass")
            except BaseException as error:
                receipt["builder_worker_stop_error"] = f"{type(error).__name__}: {error}"
                receipt["passed"] = False
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
        for handle in cpu_locks:
            handle.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
