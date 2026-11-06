"""Gate one built immutable image on idle DGX4 inside this task's window."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from spark_ds41f import bench, build, config, lab

HOLDER = "spark-lab: verification capture policies"
NODES = {
    "ssh_user": "swank",
    "nodes": [{"name": "dgx4", "management_ip": "10.0.1.74", "head": True}],
}
# These files affect only qualification, not the built image or runtime recipe.
POST_BUILD_FILES = {
    str(EXP.relative_to(ROOT) / name)
    for name in (
        "gpu_gate.py",
        "test_gpu_gate.py",
        "test_graph_replay.py",
        "boundary_grid.py",
        "lab-plan.md",
        "README.md",
        "functional.json",
        "performance.json",
        "branch_round.py",
        "test_branch_round.py",
        "preservation.py",
        "test_preservation.py",
        "gate_window.py",
        "stop_gate.py",
        "test_gate_window.py",
    )
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def owner(head, window):
    raw = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            head,
            "cat ~/spark-hold.json",
        ],
        text=True,
        timeout=12,
    )
    hold = json.loads(raw)
    if (hold.get("holder"), hold.get("window")) != (HOLDER, window):
        raise RuntimeError("exact window ownership lost")
    return hold


def build_matches(record, inspection, current, expected, changed):
    if (
        record.get("cpu_build_completed") is not True
        or record.get("error")
        or not record.get("ended")
        or record.get("builder_stop_exit") != 0
        or record.get("image") != inspection.get("Id")
    ):
        raise RuntimeError("CPU build is incomplete or inspection differs")
    if any(
        path not in POST_BUILD_FILES
        and not path.startswith(str(EXP.relative_to(ROOT) / "runs") + "/")
        for path in changed
    ):
        raise RuntimeError("runtime/build inputs changed after the recorded build")
    if inspection["Config"].get("Labels", {}) != current["Config"].get("Labels", {}):
        raise RuntimeError("copied image labels differ")
    if lab.spark.image_content_identity(inspection) != lab.spark.image_content_identity(
        current
    ):
        raise RuntimeError("copied image filesystem/runtime configuration differs")
    labels = current["Config"].get("Labels", {})
    for key, value in expected.items():
        if labels.get(key) != value:
            raise RuntimeError(f"source label differs: {key}")


def validated_screen(path, revision, image):
    record = json.loads(path.read_text())
    checks = {"gpu-smoke", "graph-replay", "memcheck", "initcheck"}
    if (
        record.get("screen_passed") is not True
        or not record.get("ended")
        or record.get("commit") != revision
        or record.get("error")
        or record.get("image_content") != lab.spark.image_content_identity(image)
        or any(
            record.get("checks", {}).get(k, {}).get("passed") is not True
            for k in checks
        )
        or record.get("conditions", {}).get("halted")
        or record.get("conditions", {}).get("fan_stop_failures")
    ):
        raise RuntimeError(
            "prior screen is incomplete or differs from this image/commit"
        )
    steps = []
    for name in ("gpu-smoke", "graph-replay", "memcheck", "initcheck"):
        matching = [x for x in record["steps"] if x["log"] == name + ".log"]
        if len(matching) != 1:
            raise RuntimeError("prior screen step missing or duplicated")
        step = matching[0]
        if step.get("exit") != 0 or hashlib.sha256(
            (path.parent / step["log"]).read_bytes()
        ).hexdigest() != step.get("sha256"):
            raise RuntimeError("prior screen log differs or step failed")
        steps.append(step)
    return record, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--window", required=True)
    parser.add_argument("--head", default="dgx1")
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--built-inspection", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--variant",
        choices=("exact-native", "row-ladder", "shape-union"),
        default="exact-native",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--screen-only", action="store_true")
    modes.add_argument("--screen-receipt", type=Path)
    args = parser.parse_args()
    directory = (
        EXP if args.variant == "exact-native" else EXP / "variants" / args.variant
    )
    profile = str((directory / "tp3.json").relative_to(ROOT))
    configured = config.read_cluster(profile)
    upstreams = json.loads((ROOT / configured["upstreams_config"]).read_text())
    expected_head = upstreams["sources"]["vllm"]["patch_head"]
    if not args.apply:
        print(
            json.dumps(
                {
                    "profile": profile,
                    "source": expected_head,
                    "output": str(args.out),
                    "holder": HOLDER,
                    "resources": "2 CPUs / 32 GiB, DGX4 GPU",
                }
            )
        )
        return 0
    if os.uname().nodename != "dgx4" or git("status", "--porcelain"):
        raise RuntimeError("gate needs a clean published checkout on DGX4")
    owner(args.head, args.window)
    if subprocess.check_output(["docker", "ps", "-q"]).strip():
        raise RuntimeError("DGX4 has active containers")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("DGX4 has active compute processes")
    with open("/tmp/spark-lab-gpu.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = git("rev-parse", "HEAD")
        cpu = json.loads(args.build_receipt.read_text())
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", cpu["commit"], revision],
            cwd=ROOT,
            check=True,
        )
        changed = git("diff", "--name-only", cpu["commit"], revision).splitlines()
        original = json.loads(args.built_inspection.read_text())
        if isinstance(original, list):
            (original,) = original
        image = json.loads(
            subprocess.check_output(
                ["docker", "image", "inspect", configured["container"]["image"]],
                text=True,
            )
        )[0]
        build_matches(
            cpu, original, image, configured["container"]["expected_labels"], changed
        )
        source_head = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            source_head != expected_head
            or subprocess.check_output(
                ["git", "-C", str(args.source), "status", "--porcelain"]
            ).strip()
        ):
            raise RuntimeError("manifest source is not the exact clean prepared head")
        args.out.mkdir(parents=True, exist_ok=False)
        out = args.out.resolve()
        shutil.copyfile(args.build_receipt, out / "cpu-build-receipt.json")
        shutil.copyfile(args.built_inspection, out / "built-inspection.json")
        shutil.copyfile(
            args.build_receipt.parent / "build-image-cpu.log",
            out / "build-image-cpu.log",
        )
        receipt = {
            "commit": revision,
            "started": lab.now(),
            "window": args.window,
            "passed": False,
            "image": image["Id"],
            "image_content": lab.spark.image_content_identity(image),
            "source": {
                "vllm_head": source_head,
                "expected_labels": configured["container"]["expected_labels"],
            },
            "build_commit": cpu["commit"],
            "post_build_files": changed,
            "checks": {},
            "steps": [
                {
                    "log": "build-image-cpu.log",
                    "argv": cpu["command"],
                    "exit": 0,
                    "started": cpu["started"],
                    "ended": cpu["ended"],
                }
            ],
            "conditions": {"halted": []},
        }
        container = "verification-shapes-gate-" + re.sub("[^a-z0-9-]", "-", args.window)
        cache = out / "cache"
        cache.mkdir()
        proc = None
        cancelled = False

        def save():
            lab.write_json_atomic(out / "receipt.json", receipt)

        def halt(reason):
            nonlocal cancelled
            cancelled = True
            if reason not in receipt["conditions"]["halted"]:
                receipt["conditions"]["halted"].append(reason)

        def run(argv, name, timeout=3600):
            nonlocal proc
            owner(args.head, args.window)
            if cancelled:
                raise RuntimeError("gate cancelled")
            step = {
                "argv": list(map(str, argv)),
                "log": name + ".log",
                "started": lab.now(),
            }
            receipt["steps"].append(step)
            save()
            started = time.monotonic()
            with (out / step["log"]).open("w") as output:
                proc = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while proc.poll() is None:
                    try:
                        owner(args.head, args.window)
                        watch.check()
                        if time.monotonic() - started > timeout:
                            raise RuntimeError(f"{name} exceeded {timeout}s")
                    except BaseException as error:  # noqa: BLE001 - stop owned work and retain cancellation evidence
                        halt(f"{type(error).__name__}: {error}")
                    if cancelled:
                        # Only our named container/process can be stopped.
                        subprocess.run(
                            ["docker", "stop", "--time", "10", container],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                        break
                    time.sleep(3)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            step.update(
                exit=proc.returncode,
                ended=lab.now(),
                sha256=hashlib.sha256((out / step["log"]).read_bytes()).hexdigest(),
            )
            save()
            watch.check()
            owner(args.head, args.window)
            if proc.returncode or cancelled:
                raise RuntimeError(f"{name} failed; see its retained log")
            return (out / step["log"]).read_text(errors="replace")

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: halt(f"signal {signum}"))
        watch = bench.MemoryWatch(NODES, 8, halt)
        save()
        code = 1
        try:
            watch.wait_ready()
            receipt["conditions"]["cooling"] = bench.cool_nodes(
                NODES, NODES["nodes"], 55, 600
            )
            receipt["conditions"]["fan_curve"] = bench.start_fan_curve(
                NODES, NODES["nodes"]
            )
            problems = bench.fan_curve_problems(receipt["conditions"]["fan_curve"])
            if problems and not all(
                x["service"] == "absent" for x in receipt["conditions"]["fan_curve"]
            ):
                raise RuntimeError(str(problems))
            root = lab.image_source_root(image["Id"])
            common = [
                "docker",
                "run",
                "--rm",
                "--name",
                container,
                "--gpus",
                "all",
                "--cpus=2",
                "--memory=32g",
                "--memory-swap=32g",
                "--ipc=host",
                "--ulimit=core=0",
                "--security-opt=seccomp=unconfined",
                "--ulimit=memlock=-1:-1",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "VLLM_NO_USAGE_STATS=1",
                "-e",
                "DO_NOT_TRACK=1",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "MAX_JOBS=1",
                "-e",
                "TILELANG_CACHE_DIR=/c/tilelang",
                "-v",
                f"{cache}:/c",
                "-w",
                root + "/vllm",
            ]
            if args.screen_receipt:
                prior, screened_steps = validated_screen(
                    args.screen_receipt, revision, image
                )
                receipt["checks"] = prior["checks"]
                receipt["screen_receipt"] = str(args.screen_receipt.resolve())
                receipt["screen_receipt_sha256"] = hashlib.sha256(
                    args.screen_receipt.read_bytes()
                ).hexdigest()
                for step in screened_steps:
                    shutil.copyfile(
                        args.screen_receipt.parent / step["log"], out / step["log"]
                    )
                    receipt["steps"].append(step)
                receipt["screen_passed"] = True
            else:
                smoke = out / "smoke.py"
                smoke.write_text(build.smoke_script(upstreams))
                run(
                    [
                        *common,
                        "-v",
                        f"{smoke}:/gate/smoke.py:ro",
                        "--entrypoint",
                        "python3",
                        image["Id"],
                        "/gate/smoke.py",
                    ],
                    "gpu-smoke",
                    120,
                )
                receipt["checks"]["gpu-smoke"] = {"passed": True}
                replay = [
                    *common,
                    "-v",
                    f"{EXP / 'test_graph_replay.py'}:/gate/test_graph_replay.py:ro",
                ]
                result = run(
                    [
                        *replay,
                        "--entrypoint",
                        "python3",
                        image["Id"],
                        "-m",
                        "pytest",
                        "-q",
                        "/gate/test_graph_replay.py",
                    ],
                    "graph-replay",
                    600,
                )
                if not lab.suite_verdict(result.splitlines())["passed"]:
                    raise RuntimeError("graph replay did not pass")
                receipt["checks"]["graph-replay"] = {"passed": True}
                sanitizer = Path("/usr/local/cuda-13.0/compute-sanitizer")
                if not (sanitizer / "compute-sanitizer").is_file():
                    raise RuntimeError("Compute Sanitizer installation absent")
                for name in ("memcheck", "initcheck"):
                    result = run(
                        [
                            *replay,
                            "-e",
                            "PYTORCH_NO_CUDA_MEMORY_CACHING=1",
                            "-v",
                            f"{sanitizer}:/opt/compute-sanitizer:ro",
                            "--entrypoint",
                            "/opt/compute-sanitizer/compute-sanitizer",
                            image["Id"],
                            "--tool",
                            name,
                            "--error-exitcode",
                            "97",
                            "--print-limit",
                            "12",
                            "python3",
                            "-m",
                            "pytest",
                            "-q",
                            "/gate/test_graph_replay.py",
                        ],
                        name,
                        600,
                    )
                    if (
                        "ERROR SUMMARY: 0 errors" not in result
                        or not lab.suite_verdict(result.splitlines())["passed"]
                    ):
                        raise RuntimeError(
                            name + " lacks zero-error and exact-output evidence"
                        )
                    receipt["checks"][name] = {
                        "passed": True,
                        "log_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    }
                receipt["screen_passed"] = True
            if args.screen_only:
                receipt["qualification"] = (
                    "bounded kernel screen only; release and serving gates pending"
                )
                code = 0
            else:
                manifest, bundle = out / "manifest.json", out / "bundle"
                run(
                    [
                        sys.executable,
                        "scripts/release_tests.py",
                        "gate",
                        str(args.source),
                        expected_head,
                        str(manifest),
                    ],
                    "gate-manifest",
                    60,
                )
                run(
                    [
                        sys.executable,
                        "scripts/release_tests.py",
                        "bundle",
                        str(manifest),
                        image["Id"],
                        str(bundle),
                        "verification-shapes",
                    ],
                    "gate-bundle",
                    60,
                )
                candidate = json.loads((bundle / "candidate.json").read_text())
                candidate["cache"] = str(cache)
                command = lab.bundle_command(bundle, candidate, out)
                command[2:2] = [
                    "--name",
                    container,
                    "--cpus=2",
                    "--memory=32g",
                    "--memory-swap=32g",
                    "-e",
                    "OMP_NUM_THREADS=1",
                    "-e",
                    "MAX_JOBS=1",
                ]
                result = run(command, "release-tests", 10800)
                receipt["gate"] = lab.verdict_from_lines(result.splitlines(), candidate)
                receipt["gate_native_results"] = str(
                    cache / "release-tests/verification-shapes"
                )
                if not receipt["gate"]["passed"]:
                    raise RuntimeError("full applicable release gate failed")
                receipt["checks"]["focused-tests"] = {"passed": True}
                receipt["passed"] = True
                code = 0
        except BaseException as error:  # noqa: BLE001 - stop owned work and retain cancellation evidence
            receipt["error"] = f"{type(error).__name__}: {error}"
            print(receipt["error"], flush=True)
        finally:
            subprocess.run(
                ["docker", "stop", "--time", "10", container],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            receipt["conditions"]["thermal"] = watch.thermal_summary()
            receipt["conditions"]["memory"] = watch.close()
            receipt["conditions"]["fan_stop_failures"] = bench.stop_fan_curve(
                NODES, NODES["nodes"], receipt["conditions"].get("fan_curve", [])
            )
            if receipt["conditions"]["fan_stop_failures"] or cancelled:
                receipt["passed"] = False
                receipt["screen_passed"] = False
                code = 1
            receipt["ended"] = lab.now()
            save()
            lock.close()
        return code


if __name__ == "__main__":
    raise SystemExit(main())
