"""Owner-authorized Spaceport-only metadata screen; preserves other reservations."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/chris/spark-lab/checkouts/vllm-drafter-orchestration-20261105")
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
HOLDER = "drafter-metadata-replay"
SOURCE_HEAD = "a2b0ea500dc5022e23c4d6fc667254a806c984dc"


def utc():
    return datetime.now(timezone.utc).isoformat()


def output(command):
    return subprocess.check_output(command, text=True).strip()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-commit", required=True)
    args = parser.parse_args()
    assert socket.gethostname() == "spark-0016"
    assert (
        output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) == args.deployment_commit
    )
    assert not output(["git", "-C", str(ROOT), "status", "--porcelain"])
    assert output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]) == SOURCE_HEAD
    assert not output(["git", "-C", str(SOURCE), "status", "--porcelain"])
    args.output.mkdir(parents=True, exist_ok=False)
    request = Path("/home/chris/spark-request.json")
    hold = Path("/home/chris/spark-hold.json")
    token = uuid.uuid4().hex
    name = "drafter-metadata-" + token[:12]
    lease = dict(
        holder=HOLDER,
        token=token,
        since=utc(),
        heartbeat=utc(),
        expected_end=None,
        scope=["sp8"],
        rule="exclusive sp8 GPU screen; no serving changes",
        authorization="Owner explicitly allowed this Spaceport-only screen as an exception to dgx1 hold, after current GPU work.",
    )
    identity = dict(
        started=utc(),
        deployment_commit=args.deployment_commit,
        source_commit=SOURCE_HEAD,
        image=IMAGE,
        authorization=lease["authorization"],
        limits=dict(cpus=2, memory_bytes=8 * 1024**3, build_workers=2),
        jobs=[],
    )
    write(args.output / "identity.json", identity)
    gpu_lock = open("/tmp/spark-lab-gpu.lock", "a")
    cpu_lock = open("/home/chris/spark-cpu-work.lock", "a")
    stop = threading.Event()
    acquired = False
    own_request = False

    def heartbeat():
        while not stop.wait(10):
            if not hold.exists() or json.loads(hold.read_text()).get("token") != token:
                return
            lease["heartbeat"] = utc()
            write(hold, lease)

    def abort(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, abort)
    try:
        print("Waiting for the Spaceport reservation and existing request.", flush=True)
        while True:
            if not request.exists():
                try:
                    with request.open("x") as handle:
                        json.dump(
                            dict(
                                who=HOLDER,
                                why="Owner-approved isolated metadata placement screen",
                                how_long="about five minutes, two fresh runs; yield after current boot",
                                since=utc(),
                            ),
                            handle,
                        )
                    own_request = True
                    print("Spaceport request queued.", flush=True)
                except FileExistsError:
                    pass
            if (
                request.exists()
                and json.loads(request.read_text()).get("who") == HOLDER
            ):
                own_request = True
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if (
                        not hold.exists()
                        and not Path("/home/chris/spark3-hold.json").exists()
                    ):
                        acquired = True
                        break
                    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
            time.sleep(5)
        # The GPU lock protects the idle check and reservation from another runner.
        for cid in output(["docker", "ps", "-q"]).split():
            config = json.loads(output(["docker", "inspect", cid]))[0]["HostConfig"]
            assert config["Runtime"] == "runc" and not any(
                config.get(k) for k in ("DeviceRequests", "Devices", "Privileged")
            ), "GPU-capable container still running"
        assert not output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader",
            ]
        ), "GPU busy"
        with hold.open("x") as handle:
            json.dump(lease, handle, indent=2)
        request.unlink()
        own_request = False
        threading.Thread(target=heartbeat, daemon=True).start()
        fcntl.flock(cpu_lock, fcntl.LOCK_EX)
        write(
            args.output / "gpu-before.json",
            json.loads(output(["docker", "image", "inspect", IMAGE])),
        )
        for boot in (1, 2):
            if (
                request.exists()
                and json.loads(request.read_text()).get("who") != HOLDER
            ):
                identity["yielded"] = json.loads(request.read_text())
                break
            dest = args.output / f"boot-{boot}"
            dest.mkdir()
            lease["job"] = f"metadata-screen-boot-{boot}"
            lease["heartbeat"] = utc()
            write(hold, lease)
            telemetry = output(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,clocks.sm",
                    "--format=csv",
                ]
            )
            (dest / "before.csv").write_text(telemetry + "\n")
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--gpus=all",
                "--network=none",
                "--cpuset-cpus=0,1",
                "--cpus=2",
                "--memory=8g",
                "--memory-swap=8g",
                "--pids-limit=256",
                "-v",
                f"{ROOT}:/experiment:ro",
                "-v",
                f"{SOURCE}:/source:ro",
                "-v",
                f"{dest}:/out",
            ]
            for relative in (
                "models/deepseek_v4_1/sparse_mla.py",
                "v1/worker/gpu/spec_decode/dflash/cudagraph.py",
                "v1/worker/gpu/spec_decode/dflash/speculator.py",
            ):
                command += [
                    "-v",
                    f"{SOURCE}/vllm/{relative}:/opt/spark-ds41f/candidate/vllm/vllm/{relative}:ro",
                ]
            for setting in (
                "OMP_NUM_THREADS=2",
                "OPENBLAS_NUM_THREADS=2",
                "MAX_JOBS=2",
                "PYTHONUNBUFFERED=1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                "VLLM_PLUGINS=",
                "VLLM_USE_V2_MODEL_RUNNER=1",
                "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1",
                "DO_NOT_TRACK=1",
                "VLLM_NO_USAGE_STATS=1",
                "TILELANG_CACHE_DIR=/out/tilelang-cache",
                "VLLM_CACHE_ROOT=/out/vllm-cache",
                "PYTHONPATH=/opt/spark-ds41f/candidate/vllm",
            ):
                command += ["-e", setting]
            script = f"""set -eu
uv venv --python /usr/bin/python3 --system-site-packages /tmp/drafter-venv
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q /source/tests/v1/attention/test_b12x_v41_ced_metadata.py -k draft_metadata_graph
/tmp/drafter-venv/bin/python /experiment/experiments/2026-11-05-drafter-metadata-replay/measure_metadata.py --boot {boot} --output /out/measurements.json
cat /sys/fs/cgroup/memory.peak > /out/container-memory-peak.txt
"""
            command += ["--entrypoint", "bash", IMAGE, "-c", script]
            write(dest / "command.json", command)
            print(f"Starting boot {boot}.", flush=True)
            started = utc()
            with (dest / "output.log").open("w") as log:
                code = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT
                ).returncode
            identity["jobs"].append(
                dict(boot=boot, started=started, ended=utc(), exit=code)
            )
            write(args.output / "identity.json", identity)
            print(f"Boot {boot} exited {code}.", flush=True)
            if code:
                raise SystemExit(code)
        identity["completed"] = len(identity["jobs"]) == 2 and all(
            j["exit"] == 0 for j in identity["jobs"]
        )
    finally:
        stop.set()
        # Only this runner's disposable test container may be stopped.
        if acquired:
            subprocess.run(
                ["docker", "stop", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if hold.exists() and json.loads(hold.read_text()).get("token") == token:
            hold.unlink()
        if (
            own_request
            and request.exists()
            and json.loads(request.read_text()).get("who") == HOLDER
        ):
            request.unlink()
        identity["ended"] = utc()
        write(args.output / "identity.json", identity)
        gpu_lock.close()
        cpu_lock.close()


if __name__ == "__main__":
    main()
