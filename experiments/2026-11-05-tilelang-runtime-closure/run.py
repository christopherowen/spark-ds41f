#!/usr/bin/env python3
"""Published-source isolated sp8 tests; one reservation for back-to-back jobs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
TREE = "c061e5f6931306fa4655ba531428f639c365bf10"
WRITE_LOCK = threading.RLock()


def capture(argv):
    return subprocess.check_output(argv, text=True).strip()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def memory():
    return 1024 * int(
        next(
            x.split()[1]
            for x in Path("/proc/meminfo").read_text().splitlines()
            if x.startswith("MemAvailable:")
        )
    )


def temperature():
    return max(
        int(p.read_text()) / 1000
        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")
    )


def write(path, obj):
    with WRITE_LOCK:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=["tests", "bench", "pair", "iket", "identity", "copy_screen", "wire", "transfer", "activity", "embedding_bench", "draft_bench"],
        default=["tests", "pair", "iket"],
    )
    parser.add_argument("--paired-rounds", type=int, choices=(120, 480, 2048), default=120)
    parser.add_argument("--case-filter", type=Path)
    parser.add_argument("--boots", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--family", choices=("sampler", "reduction", "gather", "prompt", "wire", "draft", "embedding", "all"), default="sampler")
    args = parser.parse_args()
    if args.case_filter:
        assert args.operations in (["pair"], ["activity"], ["embedding_bench"]), "filtered runs contain only paired jobs"
        case_filter = (ROOT / args.case_filter).resolve()
        assert case_filter.is_file() and case_filter.is_relative_to(
            ROOT / "experiments"
        )

    if args.paired_rounds == 2048:
        assert args.operations == ["embedding_bench"] and args.case_filter

    assert socket.gethostname() == "spark-0016", "authorized only on idle sp8"
    assert not capture(["git", "-C", str(ROOT), "status", "--porcelain"]), (
        "dirty checkout"
    )
    for cid in capture(["docker", "ps", "-q"]).splitlines():
        info = json.loads(capture(["docker", "inspect", cid]))[0]
        host = info["HostConfig"]
        assert host["Runtime"] == "runc" and not any(
            host.get(k) for k in ("DeviceRequests", "Devices", "Privileged")
        ), "GPU-capable container running"
    assert not capture(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]
    ), "GPU busy"
    assert memory() >= 40 * (1 << 30), "insufficient host memory"
    image = json.loads(capture(["docker", "image", "inspect", IMAGE]))[0]
    assert image["Config"]["Labels"]["local.spark-ds41f.vllm.tree"] == TREE
    lock = open("/tmp/spark-lab-gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hold = Path.home() / "spark-hold.json"
    assert not hold.exists() and not (Path.home() / "spark3-hold.json").exists(), (
        "reserved"
    )
    token = uuid.uuid4().hex
    container = "runtime-check-" + token[:12]
    state = dict(
        holder="tilelang-runtime-closure",
        token=token,
        since=utc(),
        expected_end=None,
        heartbeat=utc(),
        scope=["sp8"],
        rule="exclusive sp8 GPU checks; no serving changes",
    )
    request = Path.home() / "spark-request.json"
    if request.exists():
        assert json.loads(request.read_text()).get("who") == state["holder"], "another check is queued"
    out = Path.home() / "spark-lab/tilelang-runtime-closure-20261105" / args.run
    out.mkdir(parents=True, exist_ok=False)
    folders = [ROOT / name for name in args.experiments]
    assert all(
        p.is_dir() and p.resolve().is_relative_to(ROOT / "experiments") for p in folders
    )
    jobs = [
        dict(experiment=str(p.relative_to(ROOT)), operation=op, boot=boot)
        for p in folders
        for op in args.operations
        for boot in args.boots
    ]
    write(
        out / "identity.json",
        dict(
            commit=capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            image=IMAGE,
            labels=image["Config"]["Labels"],
            started=utc(),
            jobs=jobs,
        ),
    )
    with hold.open("x") as handle:
        json.dump(state, handle, indent=2)
    request = Path.home() / "spark-request.json"
    if (
        request.exists()
        and json.loads(request.read_text()).get("who") == state["holder"]
    ):
        request.unlink()
    stop, active, failed = threading.Event(), threading.Event(), threading.Event()

    def monitor():
        tick = 0
        with (out / "memory.jsonl").open("w", buffering=1) as handle:
            while not stop.is_set():
                available = memory()
                handle.write(
                    json.dumps(
                        dict(
                            utc=utc(),
                            available_bytes=available,
                            hottest_c=temperature(),
                            active=active.is_set(),
                            job=state.get("job"),
                        )
                    )
                    + "\n"
                )
                if active.is_set() and available < 8 * (1 << 30):
                    failed.set()
                    subprocess.run(["docker", "kill", container], capture_output=True)
                if tick % 20 == 0:
                    if (
                        not hold.exists()
                        or json.loads(hold.read_text()).get("token") != token
                    ):
                        failed.set()
                        subprocess.run(
                            ["docker", "kill", container], capture_output=True
                        )
                        return
                    state["heartbeat"] = utc()
                    write(hold, state)
                tick += 1
                stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results, invalid = [], set()
    try:
        for job in jobs:
            label = (
                Path(job["experiment"]).name
                + "-"
                + job["operation"]
                + "-"
                + str(job["boot"])
            )
            if job["experiment"] in invalid:
                results.append(
                    dict(job=job, label=label, skipped="earlier variant failure")
                )
                write(out / "results.json", results)
                continue
            dest = out / label
            dest.mkdir()
            until = time.monotonic() + 600
            while temperature() >= 55:
                assert time.monotonic() < until, "cooling timeout"
                stop.wait(2)
            assert not failed.is_set(), "memory/reservation guard failed"
            if request.exists() and len(results) < len(jobs):
                queued = json.loads(request.read_text())
                if queued.get("who") != state["holder"]:
                    write(out / "yielded.json", dict(ended=utc(), request=queued, pending=jobs[len(results):]))
                    return 75
            state["job"] = label
            write(hold, state)
            cmd = [
                "docker",
                "run",
                "--rm",
                "--name",
                container,
                "--gpus=all",
                "--ipc=host",
                "--memory=32g",
                "--memory-swap=32g",
                "--ulimit=core=0",
                "--cpuset-cpus=5-6,15-16",
                "-v",
                str(ROOT) + ":/r:ro",
                "-v",
                str(out) + ":/out",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                "-e",
                "VLLM_PLUGINS=",
                "-e",
                "DO_NOT_TRACK=1",
                "-e",
                "VLLM_NO_USAGE_STATS=1",
                "-e",
                "VLLM_USE_V2_MODEL_RUNNER=1",
                "-e",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                "-e",
                "TILELANG_CACHE_DIR=/out/tilelang-cache",
                "-e",
                "OMP_NUM_THREADS=2",
                "-e",
                "MAX_JOBS=2",
                "-e",
                "PYTHONUNBUFFERED=1",
                "--entrypoint",
                "sh",
                IMAGE,
                "-c",
                'uv venv --system-site-packages /tmp/.venv >&2 && exec /tmp/.venv/bin/python "$@"',
                "sh",
                "/r/" + job["experiment"] + "/measure.py",
                job["operation"],
                "--family", args.family,
                "--out",
                "/out/" + label,
            ]
            if job["operation"] in ("pair", "activity", "embedding_bench", "draft_bench"):
                cmd += ["--rounds", str(args.paired_rounds)]
                if args.case_filter:
                    cmd += ["--case-filter", "/r/" + str(case_filter.relative_to(ROOT))]
            write(dest / "command.json", cmd)
            print(utc(), "START", label, flush=True)
            active.set()
            begin = time.monotonic()
            with (dest / "output.txt").open("w") as handle:
                process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "kill", container], capture_output=True)
                    code = 124
            active.clear()
            row = dict(
                job=job,
                label=label,
                exit=code,
                seconds=time.monotonic() - begin,
                memory_guard=failed.is_set(),
            )
            if (dest / "result.json").exists():
                row["result"] = json.loads((dest / "result.json").read_text())
            results.append(row)
            write(out / "results.json", results)
            print(utc(), "END", label, code, round(row["seconds"], 2), flush=True)
            if code:
                invalid.add(job["experiment"])
            assert not failed.is_set(), "memory/reservation guard failed"
        write(
            out / "completed.json",
            dict(ended=utc(), jobs=len(results), failed=sorted(invalid)),
        )
    finally:
        stop.set()
        thread.join(5)
        subprocess.run(["docker", "kill", container], capture_output=True)
        if hold.exists() and json.loads(hold.read_text()).get("token") == token:
            hold.unlink()
        write(
            out / "closure.json",
            dict(ended=utc(), reservation_released=True, container=container),
        )
        print(utc(), "sp8 released", flush=True)
    return bool(invalid)


if __name__ == "__main__":
    raise SystemExit(main())
