#!/usr/bin/env python3
"""Isolated sp8 diagnostic jobs from a clean published checkout."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
TREE = "c061e5f6931306fa4655ba531428f639c365bf10"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / "experiments/2026-10-31-iket-full-43l-and-runtime-opt/harness"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def capture(argv):
    return subprocess.check_output(argv, text=True).strip()


def memory():
    return int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines()
                    if l.startswith("MemAvailable:"))) * 1024


def temperature():
    return max(int(p.read_text()) / 1000 for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    jobs = json.loads(Path(args.matrix).read_text())
    if socket.gethostname() != "spark-0016":
        raise SystemExit("authorized only on idle sp8")
    if capture(["docker", "ps", "-q"]):
        raise SystemExit("sp8 is not idle")
    if capture(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise SystemExit("diagnostic checkout must be clean")
    if memory() < 40 * (1 << 30):
        raise SystemExit("less than 40 GiB available")
    info = json.loads(capture(["docker", "image", "inspect", IMAGE]))[0]
    if info["Config"]["Labels"]["local.spark-ds41f.vllm.tree"] != TREE:
        raise SystemExit("candidate-38 image identity mismatch")
    hold = Path.home() / "spark-hold.json"
    if hold.exists() or (Path.home() / "spark3-hold.json").exists():
        raise SystemExit("sp8 is reserved")
    lock = open("/tmp/spark-lab-gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = Path.home() / "spark-lab/iket-repair-20261104" / args.run
    out.mkdir(parents=True, exist_ok=False)
    state = {"holder": "iket-repair-" + args.run, "since": utc(), "expected_end": None,
             "heartbeat": utc(), "nodes": ["sp8"],
             "rule": "sp8-only diagnostics; no other GPU jobs or builds on sp8"}
    mutex = threading.RLock()

    def write(path, value):
        with mutex:
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(json.dumps(value, indent=2) + "\n")
            os.replace(temp, path)

    write(out / "identity.json", {"image": IMAGE, "labels": info["Config"]["Labels"],
                                   "commit": capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                                   "matrix": jobs, "started": utc()})
    with hold.open("x") as handle:
        json.dump(state, handle)
    stop, active, failed_guard = threading.Event(), threading.Event(), threading.Event()
    container = "iket-repair-sp8"

    def monitor():
        n = 0
        with (out / "memory.jsonl").open("w", buffering=1) as handle:
            while not stop.is_set():
                free = memory()
                handle.write(json.dumps({"utc": utc(), "available_bytes": free,
                                         "hottest_c": temperature(), "active": active.is_set(),
                                         "job": state.get("current")}) + "\n")
                if active.is_set() and free < 8 * (1 << 30):
                    failed_guard.set()
                    subprocess.run(["docker", "kill", container], capture_output=True)
                if n % 20 == 0:
                    state["heartbeat"] = utc()
                    write(hold, state)
                n += 1
                stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = []
    try:
        for job in jobs:
            label = job["label"]
            dest = out / label
            dest.mkdir()
            deadline = time.monotonic() + 600
            while temperature() >= 55:
                if time.monotonic() >= deadline:
                    raise RuntimeError("sp8 failed to cool below 55 C")
                time.sleep(2)
            state["current"] = label
            write(hold, state)
            script = "/r/" + job.get("script", "reproduce.py")
            workload = ["python3", script, job["operation"], "--out", "/out/" + label]
            workload += job.get("args", [])
            if job.get("debugger"):
                workload += ["--wait-file", "/out/" + label + "/go"]
            if job["mode"] != "plain":
                workload = ["python3", "/h/iket_profile.py", "/out/" + label + "/profile",
                            "--timeout", str(job.get("timeout", 180)), "--log-level", "debug", "--warp-trace",
                            "--pool-gb", str(job.get("pool_gb", 0.25)),
                            *(["--skip-raw-data-dump"] if job["mode"] == "patched" else []),
                            "--", *workload]
            cap = str(job.get("memory_gb", 32)) + "g"
            argv = ["docker", "run", "--rm", "--name", container, "--gpus=all", "--ipc=host",
                    "--memory=" + cap, "--memory-swap=" + cap, "--ulimit=core=0",
                    "--cpuset-cpus=5-9,15-19", "-v", f"{HERE}:/r:ro", "-v", f"{OLD}:/h:ro",
                    "-v", f"{out}:/out", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                    "-e", "VLLM_PLUGINS=", "-e", "DO_NOT_TRACK=1", "-e", "VLLM_NO_USAGE_STATS=1",
                    "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONFAULTHANDLER=1"]
            if job.get("script") == "full_step.py":
                argv += ["-v", f"{Path.home() / '.cache/huggingface'}:/hf:ro",
                         "-v", f"{Path.home() / 'spark-lab/spaceport-iket-c38-20261104/cache'}:/c"]
            for key, value in sorted(job.get("env", {}).items()):
                argv += ["-e", f"{key}={value}"]
            argv += ["--entrypoint", workload[0], IMAGE, *workload[1:]]
            write(out / (label + ".command.json"), argv)
            print(utc(), "START", label, flush=True)
            active.set()
            start = time.monotonic()
            with (out / (label + ".log")).open("w") as handle:
                try:
                    process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT)
                    if job.get("debugger"):
                        until = time.monotonic() + 90
                        while not (dest / "ready").exists() and process.poll() is None:
                            if time.monotonic() >= until:
                                raise subprocess.TimeoutExpired(argv, 90)
                            time.sleep(.1)
                        if (dest / "ready").exists():
                            top = capture(["docker", "top", container, "-eo", "pid,args"])
                            candidates = []
                            for line in top.splitlines()[1:]:
                                pid, command = line.strip().split(None, 1)
                                parts = command.split()
                                if len(parts) > 1 and parts[1] == script:
                                    candidates.append(pid)
                            if len(candidates) != 1:
                                raise RuntimeError("could not identify debugger target: " + top)
                            pid = candidates[0]
                            debugger = ["sudo", "-n", "gdb", "-q", "--batch",
                                        "-ex", "set pagination off", "-ex", "set confirm off",
                                        "-ex", "set print thread-events off",
                                        "-ex", f"set sysroot /proc/{pid}/root", "-p", pid,
                                        "-ex", "handle SIGABRT stop print nopass"]
                            if job.get("catch_throw"):
                                debugger += ["-ex", "source " + str(HERE / "catch_throw.py")]
                            debugger += ["-ex", "shell touch " + str(dest / "go"),
                                         "-ex", "continue", "-ex", "thread apply all bt 32", "-ex", "kill"]
                            write(out / (label + ".debugger-command.json"), debugger)
                            with (out / (label + ".gdb.log")).open("w") as debug_log:
                                subprocess.run(debugger, stdout=debug_log, stderr=subprocess.STDOUT,
                                               timeout=job.get("timeout", 180) + 30)
                    code = process.wait(timeout=max(1, job.get("timeout", 180) + 60 - (time.monotonic() - start)))
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "kill", container], capture_output=True)
                    code = 124
            active.clear()
            result = {"job": job, "exit": code, "seconds": time.monotonic() - start,
                      "memory_guard": failed_guard.is_set()}
            if (dest / "result.json").exists():
                result["result"] = json.loads((dest / "result.json").read_text())
            results.append(result)
            write(out / "results.json", results)
            print(utc(), "END", json.dumps(result), flush=True)
            if failed_guard.is_set():
                raise RuntimeError("memory guard stopped this window")
        write(out / "completed.json", {"ended": utc(), "jobs": len(results)})
    finally:
        stop.set()
        thread.join(5)
        subprocess.run(["docker", "kill", container], capture_output=True)
        if hold.exists() and json.loads(hold.read_text()).get("holder") == state["holder"]:
            hold.unlink()
        print(utc(), "sp8 reservation released", flush=True)


if __name__ == "__main__":
    main()
