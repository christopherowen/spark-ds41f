"""Owner-authorized Spaceport-only broader GPU suites and CUDA graph replay trace."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

ROOT = Path("/home/chris/spark-lab/checkouts/drafter-orchestration-20261105")
SOURCE = Path("/home/chris/spark-lab/checkouts/vllm-drafter-orchestration-20261105")
GATES_DIR = Path("/home/chris/spark-lab/results/drafter-gates-20261106")
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
    parser.add_argument(
        "--boots", type=int, nargs="+", choices=(1, 2), default=[1, 2]
    )
    args = parser.parse_args()
    assert socket.gethostname() == "spark-0016"
    assert (
        output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        == args.deployment_commit
    )
    assert not output(["git", "-C", str(ROOT), "status", "--porcelain"])
    assert output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]) == SOURCE_HEAD
    assert not output(["git", "-C", str(SOURCE), "status", "--porcelain"])
    args.output.mkdir(parents=True, exist_ok=False)
    request = Path("/home/chris/spark-request.json")
    hold = Path("/home/chris/spark-hold.json")
    token = uuid.uuid4().hex
    name = "drafter-gates-" + token[:12]
    lease = dict(
        holder=HOLDER,
        token=token,
        since=utc(),
        heartbeat=utc(),
        expected_end=None,
        scope=["sp8"],
        rule="exclusive sp8 GPU qualification gates and graph trace; no serving changes",
        authorization="Owner explicitly authorized broader GPU suites and CUDA graph replay trace on sp8 (Pool A).",
    )
    identity = dict(
        started=utc(),
        deployment_commit=args.deployment_commit,
        source_commit=SOURCE_HEAD,
        image=IMAGE,
        authorization=lease["authorization"],
        limits=dict(cpus=2, memory_bytes=8 * 1024**3, build_workers=2),
        trace_script_sha256=hashlib.sha256(
            (GATES_DIR / "trace_and_gates.py").read_bytes()
        ).hexdigest(),
        runner_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requested_boots=args.boots,
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
        print("Checking sp8 reservation and request state.", flush=True)
        while True:
            if not request.exists():
                try:
                    with request.open("x") as handle:
                        json.dump(
                            dict(
                                who=HOLDER,
                                why="Owner-approved broader GPU suites and CUDA graph replay trace",
                                how_long="about five minutes, two fresh boots",
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
        image = json.loads(output(["docker", "image", "inspect", IMAGE]))[0]
        write(
            args.output / "image.json",
            {"id": image["Id"], "labels": image["Config"]["Labels"]},
        )
        for boot in args.boots:
            if (
                request.exists()
                and json.loads(request.read_text()).get("who") != HOLDER
            ):
                identity["yielded"] = json.loads(request.read_text())
                break
            dest = args.output / f"boot-{boot}"
            dest.mkdir()
            lease["job"] = f"metadata-gates-boot-{boot}"
            lease["heartbeat"] = utc()
            write(hold, lease)
            until = time.monotonic() + 600
            while (
                float(
                    output(
                        [
                            "nvidia-smi",
                            "--query-gpu=temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                )
                >= 55
            ):
                assert time.monotonic() < until, "cooling timeout"
                time.sleep(2)
            telemetry = output(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,clocks.sm,fan.speed,power.draw,power.limit",
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
                f"{GATES_DIR}:/gates:ro",
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
echo "=== [1/4] Cold-cache compilation, before/after graph memory, and CUDA graph replay trace ==="
/tmp/drafter-venv/bin/python /gates/trace_and_gates.py --boot {boot} --output /out/trace.json --dot-dir /out/dots --hashes /experiment/experiments/2026-11-05-drafter-metadata-replay/micro-source-hashes.json
echo "=== [2/4] Focused draft_metadata_graph GPU regression ==="
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q /source/tests/v1/attention/test_b12x_v41_ced_metadata.py -k draft_metadata_graph
echo "=== [3/4] Full CED metadata, DFlash input, causality, DCP, cudagraph, and DSpark Markov/sampling suites ==="
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q \\
  /source/tests/v1/attention/test_b12x_v41_ced_metadata.py \\
  /source/tests/v1/spec_decode/test_dflash_prepare_inputs.py \\
  /source/tests/v1/spec_decode/test_dflash_causality.py \\
  /source/tests/v1/spec_decode/test_dflash_dcp.py \\
  /source/tests/v1/spec_decode/test_dspark_v41_gathered_markov.py \\
  /source/tests/v1/spec_decode/test_dspark_vocab_parallel.py \\
  /source/tests/v1/worker/test_dspark_greedy_markov.py \\
  /source/tests/v1/worker/test_dspark_utils.py \\
  /source/tests/v1/cudagraph/test_cudagraph_manager.py \\
  /source/tests/kernels/test_deepseek_v41_tilelang_dspark_context.py
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q \\
  /source/tests/v1/spec_decode/test_dspark_topk.py -k "not [True]"
echo "=== [4/4] Focused TileLang DFlash inputs and greedy Markov sampling kernels ==="
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q \\
  /source/tests/v1/worker/test_gpu_tilelang_inputs.py -k "prepare_dflash_inputs or padding_rows"
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -p no:cacheprovider -q \\
  /source/tests/v1/worker/test_gpu_tilelang_sampling.py -k "greedy or markov"
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
            after = output(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,clocks.sm,fan.speed,power.draw,power.limit",
                    "--format=csv",
                ]
            )
            (dest / "after.csv").write_text(after + "\n")
            identity["jobs"].append(
                dict(boot=boot, started=started, ended=utc(), exit=code)
            )
            write(args.output / "identity.json", identity)
            print(f"Boot {boot} exited {code}.", flush=True)
            if code:
                raise SystemExit(code)
        identity["completed"] = len(identity["jobs"]) == len(args.boots) and all(
            j["exit"] == 0 for j in identity["jobs"]
        )
    finally:
        stop.set()
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
