"""Build only the CPU image stages on SP8; GPU smoke and qualification stay pending.

Uses the exact command printed by the normal build CLI, its host-memory guard,
and an isolated 2-CPU/24-GiB BuildKit worker with no device entitlement/CDI.
No image from this script is a passing qualification receipt or serving input.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import build

EXP = Path(__file__).resolve().parent
BUILDER = "verification-shapes-cpu"
BUILDKIT = "docker.io/moby/buildkit@sha256:92f93f25966c544549b88144406ce8cd9dcdf5df128250711031a4acd9149a8e"
MEMORY = 24 * 1024**3


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, **kwargs)


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def builder_identity():
    data = json.loads(
        output(["docker", "inspect", "buildx_buildkit_" + BUILDER + "0"])
    )[0]
    host = data["HostConfig"]
    if (
        host["Memory"] != MEMORY
        or host["MemorySwap"] != MEMORY
        or host["CpuQuota"] != 200000
        or host["CpuPeriod"] != 100000
        or host.get("DeviceRequests")
        or host.get("Devices")
        or host["RestartPolicy"]["Name"] != "no"
    ):
        raise RuntimeError("BuildKit resource/device settings differ")
    if data["Config"]["Image"] != BUILDKIT:
        raise RuntimeError("BuildKit image differs from the pinned ARM64 manifest")
    configured = output(
        [
            "docker",
            "exec",
            "buildx_buildkit_" + BUILDER + "0",
            "cat",
            "/etc/buildkit/buildkitd.toml",
        ]
    )
    if configured.strip() != (EXP / "buildkitd.toml").read_text().strip():
        raise RuntimeError("BuildKit configuration differs")
    return {
        "image": data["Image"],
        "reference": data["Config"]["Image"],
        "memory": host["Memory"],
        "memory_swap": host["MemorySwap"],
        "cpu_quota": host["CpuQuota"],
        "cpu_period": host["CpuPeriod"],
        "config_sha256": hashlib.sha256(configured.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("exact-native", "row-ladder", "shape-union"),
        default="exact-native",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    directory = (
        EXP if args.variant == "exact-native" else EXP / "variants" / args.variant
    )
    profile = str((directory / "tp3.json").relative_to(ROOT))
    cli = [
        sys.executable,
        str(ROOT / "bin/spark"),
        "--cluster-config",
        profile,
        "build",
        "image",
    ]
    plan = output(cli)
    print(plan, flush=True)
    if not args.apply:
        print(
            "CPU phase only; worker 2 CPUs / 24 GiB, no GPU devices. GPU gates remain required."
        )
        return 0
    if os.uname().nodename != "spark-0016" or args.attempt < 1:
        raise RuntimeError("CPU builds are restricted to the reserved SP8 host")
    reservation = json.loads(
        (Path.home() / "spark-work-reservations/verification-shapes.json").read_text()
    )
    if (
        reservation.get("holder") != "verification-shapes"
        or reservation.get("build_cpu_quota_percent") != 200
        or reservation.get("build_memory_max_gib") != 24
        or reservation.get("gpu") is not False
    ):
        raise RuntimeError("CPU-only build reservation is absent or differs")
    cluster, _, upstreams = build.configuration(
        argparse.Namespace(cluster_config=profile)
    )
    inputs = build.build_inputs(upstreams)
    build.require_idle_build_host(cluster)
    if build.available_memory_gib() < build.BUILD_MIN_AVAILABLE_GIB:
        raise RuntimeError("normal build start memory guard refused")
    tag = cluster["container"]["image"]
    if build.image_labels(tag) is not None:
        raise RuntimeError("immutable image tag already exists; preserve it")
    out = (
        ROOT
        / "results/private/verification-shapes"
        / f"build-{args.variant}-attempt-{args.attempt}"
    )
    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "commit": build.repository_revision(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "profile": profile,
        "tag": tag,
        "cpu_only": True,
        "passed": False,
        "qualification": "GPU smoke, release tests, sanitizers and serving qualification are pending",
    }

    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    with build.BUILD_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            present = (
                subprocess.run(
                    ["docker", "buildx", "inspect", BUILDER],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if not present:
                run(
                    [
                        "docker",
                        "buildx",
                        "create",
                        "--name",
                        BUILDER,
                        "--driver",
                        "docker-container",
                        "--buildkitd-config",
                        str(EXP / "buildkitd.toml"),
                        "--buildkitd-flags",
                        "--allow-insecure-entitlement network.host",
                        "--driver-opt",
                        f"image={BUILDKIT},memory=24g,memory-swap=24g,cpu-quota=200000,cpu-period=100000,restart-policy=no",
                    ]
                )
            run(["docker", "buildx", "inspect", BUILDER, "--bootstrap"])
            receipt["builder"] = builder_identity()
            # Check the actual worker RUN environment without importing a CUDA
            # library or issuing any GPU operation. Device entitlement is disabled.
            base = inputs["base_image"]
            probe = f"FROM {base}\nRUN python3 -c \"import glob; p=glob.glob('/dev/nvidia*')+glob.glob('/dev/nvhost*')+glob.glob('/dev/dri/*'); print('GPU device paths:',p); assert not p\"\n"
            (out / "device-probe.Dockerfile").write_text(probe)
            probe_context = out / "probe-context"
            probe_context.mkdir()
            with (out / "device-probe.log").open("w") as log:
                run(
                    [
                        "docker",
                        "buildx",
                        "build",
                        "--builder",
                        BUILDER,
                        "--platform",
                        "linux/arm64",
                        "--progress",
                        "plain",
                        "--file",
                        "-",
                        str(probe_context),
                    ],
                    input=probe,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            receipt["worker_gpu_device_paths"] = []
            command = shlex.split(plan.splitlines()[0])
            if command[:3] != ["docker", "buildx", "build"]:
                raise RuntimeError("normal build CLI did not print a build command")
            command = [
                item.replace("=<sized from MemAvailable>", "=1") for item in command
            ]
            command[-1:-1] = [
                "--builder",
                BUILDER,
                "--progress",
                "plain",
                "--build-arg",
                "NCCL_COMPILE_JOBS=1",
            ]
            receipt["command"] = command
            save()
            # The controller's affinity also documents the single compile job;
            # the separate Docker worker enforces the real CPU/memory ceilings.
            os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
            receipt["compile_jobs"] = build.compile_jobs(build.available_memory_gib())
            if receipt["compile_jobs"] != 1:
                raise RuntimeError("compile job sizing differs from the bounded plan")
            with (out / "build-image-cpu.log").open("w") as log:
                saved_out, saved_err = os.dup(1), os.dup(2)
                try:
                    os.dup2(log.fileno(), 1)
                    os.dup2(log.fileno(), 2)
                    receipt["min_available_gib"] = build.run_memory_guarded(
                        command, build.BUILD_RESERVE_GIB
                    )
                finally:
                    os.dup2(saved_out, 1)
                    os.dup2(saved_err, 2)
                    os.close(saved_out)
                    os.close(saved_err)
            image = build.image_labels(tag)
            if image is None:
                raise RuntimeError("build returned without the image")
            for name in build.build_projects(upstreams["sources"]):
                expected = inputs[name]["expected_tree"]
                if build.source_labels.get(image["labels"], name + ".tree") != expected:
                    raise RuntimeError(f"built {name} source label differs")
            receipt.update(image=image["id"], cpu_build_completed=True)
        except BaseException as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            receipt["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save()
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
