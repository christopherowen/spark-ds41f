#!/usr/bin/env python3
"""Plan C45 build limits and validate saved worker observations; no Docker dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("buildkitd.toml")
MEMORY = 32 * 1024**3
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS")


def validate_name_image(name, image):
    if not re.fullmatch(r"c45-[a-z0-9][a-z0-9_-]{0,55}", name):
        raise ValueError("use a unique C45 builder name; never modify the shared builder")
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
        raise ValueError("BuildKit image must be a recorded immutable local ID or repository digest")


def builder_plan(name, image, config=CONFIG):
    validate_name_image(name, image)
    if tomllib.loads(config.read_text()).get("worker", {}).get("oci", {}).get("max-parallelism") != 1:
        raise ValueError("BuildKit must serialize build steps")
    create = ["docker", "buildx", "create", "--name", name, "--node", name + "0", "--driver", "docker-container",
              "--buildkitd-config", str(config),
              "--buildkitd-flags", "--allow-insecure-entitlement=network.host"]
    for option in ("image=" + image, "memory=32g", "memory-swap=32g", "cpu-quota=200000",
                   "cpu-period=100000", "network=host", "restart-policy=no",
                   "env.NVIDIA_VISIBLE_DEVICES=void"):
        create += ["--driver-opt", option]
    return {"create_argv": create, "selected_environment": {"BUILDX_BUILDER": name},
            "bootstrap_argv": ["docker", "buildx", "inspect", name, "--bootstrap"],
            "stop_argv": ["docker", "buildx", "stop", name],
            "image_command_suffix": ["build", "image", "--max-compile-jobs", "2"],
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "limits": {"compile_jobs": 2, "worker_cpu_quota": 200000, "worker_cpu_period": 100000,
                       "worker_memory_bytes": MEMORY, "worker_memory_plus_swap_bytes": MEMORY,
                       "parallel_build_steps": 1},
            "scope": "Owned DGX4 window only; preserve publish/idle/memory/thermal guards and named plans. "
                     "No global builder selection, shared builder update, GPU build device or cache pruning. "
                     "The ordinary image command still owns its separate GPU import smoke."}


def validate_worker(name, image, buildx, inspection, actual_config, cgroup, config=CONFIG):
    plan = builder_plan(name, image, config)
    if buildx.get("Name") != name or buildx.get("Driver") != "docker-container" or len(buildx.get("Nodes", [])) != 1:
        raise ValueError("wrong builder name, driver or node count")
    if not isinstance(inspection, list) or len(inspection) != 1:
        raise ValueError("one inspected builder container is required")
    container = inspection[0]
    host = container.get("HostConfig", {})
    expected = {"CpuQuota": 200000, "CpuPeriod": 100000, "Memory": MEMORY, "MemorySwap": MEMORY}
    if any(host.get(k) != v for k, v in expected.items()):
        raise ValueError("actual BuildKit CPU/memory/swap limits differ")
    if (container.get("Name") != "/buildx_buildkit_" + name + "0"
            or container.get("State", {}).get("Running") is not True
            or container.get("Config", {}).get("Image") != image
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", container.get("Image", ""))):
        raise ValueError("builder container identity, image or running state differs")
    if image.startswith("sha256:") and container["Image"] != image:
        raise ValueError("actual local BuildKit image ID differs from its immutable pin")
    if (host.get("DeviceRequests") or host.get("Devices") or host.get("Runtime") == "nvidia"
            or "NVIDIA_VISIBLE_DEVICES=void" not in container.get("Config", {}).get("Env", [])):
        raise ValueError("BuildKit must not request GPUs or force the NVIDIA runtime")
    if (tomllib.loads(actual_config.decode()) != tomllib.loads(config.read_text())
            or str(cgroup.get("cpu.max", "")).split() != ["200000", "100000"]
            or str(cgroup.get("memory.max", "")).strip() != str(MEMORY)
            or str(cgroup.get("memory.swap.max", "")).strip() != "0"):
        raise ValueError("actual daemon configuration or cgroup bounds differ")
    args = container.get("Args", [])
    if ("--allow-insecure-entitlement=network.host" not in args
            or any("security.insecure" in str(arg) or "device" in str(arg) for arg in args)):
        raise ValueError("unexpected BuildKit entitlements")
    return {"status": "saved_BuildKit_worker_resource_observations_validated", "passed": True,
            "builder": name, "container_id": container["Id"], "image_id": container["Image"],
            "effective_host_config": expected, "cgroup": cgroup, "plan": plan,
            "gpu_assignment": {"device_requests": host.get("DeviceRequests"), "devices": host.get("Devices"),
                               "runtime": host.get("Runtime"), "privileged": host.get("Privileged"),
                               "limit": "The standard BuildKit worker is privileged; empty GPU requests do not prove device paths invisible. No GPU workload is dispatched to this worker."},
            "limits": "Worker compilation bound only. Docker --load/image-store work and the standard GPU "
                      "smoke are separate processes. Keep host memory guards and record an actual compiler "
                      "PID/cgroup ancestor during the new build; do not claim client-only limits cover them."}


def source_plan(profile, operation):
    path = Path(profile)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("profile must be named relative to the repository")
    return {"argv": [sys.executable, str(ROOT / "bin/spark"), "--cluster-config", profile,
                     "build", operation],
            "limits": {"cpu_affinity_count": 1, "address_space_bytes": 8 * 1024**3,
                       "environment": {key: "1" for key in THREAD_ENV}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    validate = commands.add_parser("validate-worker")
    for command in (plan, validate):
        command.add_argument("--name", required=True)
        command.add_argument("--image", required=True)
    validate.add_argument("--buildx-inspection", type=Path, required=True)
    validate.add_argument("--container-inspection", type=Path, required=True)
    validate.add_argument("--actual-config", type=Path, required=True)
    validate.add_argument("--cgroup-observation", type=Path, required=True)
    source = commands.add_parser("source")
    source.add_argument("--profile", required=True)
    source.add_argument("--operation", choices=("prepare", "check"), required=True)
    source.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        result = builder_plan(args.name, args.image)
    elif args.command == "validate-worker":
        result = validate_worker(args.name, args.image, json.loads(args.buildx_inspection.read_text()),
                                 json.loads(args.container_inspection.read_text()),
                                 args.actual_config.read_bytes(), json.loads(args.cgroup_observation.read_text()))
    else:
        result = source_plan(args.profile, args.operation)
        if args.apply:
            if sys.platform != "linux":
                raise SystemExit("source preparation is remote Linux only")
            resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
            environment = os.environ.copy()
            environment.update(result["limits"]["environment"])
            os.chdir(ROOT)
            os.execvpe(result["argv"][0], result["argv"], environment)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
