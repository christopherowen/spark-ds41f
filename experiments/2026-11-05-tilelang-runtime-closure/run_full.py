#!/usr/bin/env python3
"""Isolated full-model CUDA inventory with a verified, published Python overlay.

Run on sp8 with stdin from this file. This controller is an experiment input,
not a serving overlay. It uses the already-published 43-layer harness on sp8.
"""
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
HOLDER = "tilelang-runtime-full-20261105" + os.environ.get("IKET_RUN_SUFFIX", "")
HOME_DIR = Path.home()
REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "experiments/2026-10-31-iket-full-43l-and-runtime-opt"
OUT = HOME_DIR / "spark-lab" / HOLDER
CACHE = HOME_DIR / "spark-lab/tilelang-runtime-full-20261105/cache"
HOLD = HOME_DIR / "spark-hold.json"
CONTAINER = "tilelang-runtime-full"
JSON_LOCK = threading.RLock()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def command(args):
    return subprocess.check_output(args, text=True).strip()


def write_json(path, data):
    with JSON_LOCK:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, path)


def available():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:"))) * 1024


def hottest():
    return max(int(p.read_text()) / 1000 for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"))


def main():
    if socket.gethostname() != "spark-0016":
        raise SystemExit("this experiment is authorized only on sp8 (spark-0016)")
    if command(["docker", "ps", "-q"]):
        raise SystemExit("sp8 has a running container; refusing GPU work")
    if command(["git", "-C", str(REPO), "status", "--porcelain"]):
        raise SystemExit("sp8 source checkout is dirty")
    info = json.loads(command(["docker", "image", "inspect", IMAGE]))[0]
    if info["Config"]["Labels"]["local.spark-ds41f.vllm.tree"] != TREE:
        raise SystemExit("image source tree mismatch")
    if command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]):
        raise SystemExit("sp8 GPU is busy")
    if available() < 105 * (1 << 30):
        raise SystemExit("less than 105 GiB available before isolated profiling")
    if HOLD.exists() or (HOME_DIR / "spark3-hold.json").exists():
        raise SystemExit("sp8 already has a reservation")
    request = HOME_DIR / "spark-request.json"
    if request.exists() and json.loads(request.read_text()).get("who") not in (HOLDER, "tilelang-runtime-closure"):
        raise SystemExit("another check is queued on sp8")
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise SystemExit("experiment output is not empty; refusing to overwrite evidence")
    CACHE.mkdir(parents=True, exist_ok=True)
    identity = {
        "started": stamp(), "node": socket.gethostname(), "image": IMAGE,
        "image_labels": info["Config"]["Labels"],
        "deployment_commit": command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "harness_sha256": {str(p.relative_to(HARNESS)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in HARNESS.rglob("*.py") if "__pycache__" not in str(p)},
        "kernel": os.uname().release,
        "initial_mem_available_bytes": available(), "initial_hottest_c": hottest(),
        "scope": "combined TileLang candidates; one GPU, dummy weights, 40 target and 3 drafter layers, no Engram or real peers",
    }
    write_json(OUT / "identity.json", identity)
    held = {"holder": HOLDER, "since": stamp(), "expected_end": None, "heartbeat": stamp(),
            "nodes": ["sp8"], "rule": "sp8-only GPU profiling; no other GPU jobs or builds on sp8",
            "authorization": "owner explicitly authorized idle sp8 and candidate-38; other clusters stay running"}
    lock = open("/tmp/spark-lab-gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with HOLD.open("x") as handle:
        json.dump(held, handle, indent=2)
    if request.exists() and json.loads(request.read_text()).get("who") in (HOLDER, "tilelang-runtime-closure"):
        request.unlink()
    stopped = threading.Event()
    active = threading.Event()
    guard_failure = threading.Event()
    memory_path = OUT / "memory.jsonl"

    def monitor():
        n = 0
        with memory_path.open("a", buffering=1) as handle:
            while not stopped.is_set():
                free = available()
                handle.write(json.dumps({"utc": stamp(), "mem_available_bytes": free,
                                         "hottest_c": hottest(), "container_active": active.is_set()}) + "\n")
                if active.is_set() and free < 8 * (1 << 30):
                    guard_failure.set()
                    subprocess.run(["docker", "kill", CONTAINER], capture_output=True)
                if n % 20 == 0:
                    held["heartbeat"] = stamp()
                    write_json(HOLD, held)
                n += 1
                stopped.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    env = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": "",
        "DO_NOT_TRACK": "1", "VLLM_NO_USAGE_STATS": "1", "MAX_JOBS": "4",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_DS41_HEAD_DTYPE": "float32",
        "VLLM_DS41_PACKED_BF16_LM_HEAD": "1", "VLLM_L2_PREFETCH_KERNELS": "tilelang",
        "VLLM_L2_PREFETCH_GRID": "2", "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2",
        "SPARK_DS41F_DSPARK_MAIN_PROJ_TP": "1", "SPARK_DS41F_DSPARK_VOCAB_PARALLEL": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0", "VLLM_LATE_ADMISSION_LAYER": "30",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32", "MALLOC_ARENA_MAX": "2",
        "MALLOC_MMAP_THRESHOLD_": "1048576", "OMP_NUM_THREADS": "8",
        "SAFETENSORS_FAST_GPU": "1", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "1800", "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
        "PYTHONPATH": "/b/harness:/b/scripts:/opt/spark-ds41f/candidate/vllm",
        "TILELANG_CACHE_DIR": "/c/tilelang-off", "TRITON_CACHE_DIR": "/c/triton",
        "VLLM_CACHE_ROOT": "/c/vllm",
    }
    if os.environ.get("IKET_CALL_ORIGINS") == "1":
        env["IKET_CALL_ORIGINS"] = "1"
    results = []
    try:
        # Every plain and instrumented point boots twice. Warp-trace is an
        # attribution control, never a replacement for uninstrumented timings.
        jobs = [(int(shape), mode, int(repeat)) for shape, mode, repeat in
                (entry.split(":") for entry in os.environ.get(
                    "IKET_JOBS", "4:nsys:1,4:nsys:2,3:nsys:1,3:nsys:2").split(","))]
        for shape, mode, repeat in jobs:
            deadline = time.monotonic() + 600
            while hottest() >= 55:
                if time.monotonic() >= deadline:
                    raise RuntimeError("sp8 did not cool below 55 C")
                time.sleep(2)
            label = f"tp{shape}-c1-{mode}-{repeat}"
            held["current"] = label
            write_json(HOLD, held)
            target = "/out/" + label
            step = ["/tmp/.venv/bin/python", "/r/experiments/2026-11-05-tilelang-runtime-closure/full_step.py", "on" if mode == "on" else "off",
                    "--full-layers", "--tp-shape", str(shape), "--requests", "1", "--tokens", "8",
                    "--grid", os.environ.get("IKET_GRID", "4"), "--max-num-seqs", "16" if shape == 4 else "8",
                    "--max-model-len", "8192", "--kv-bytes", os.environ.get("IKET_KV_BYTES", str(2 << 30)), "--prefix-caching",
                    "--lean-capture", "--no-warmup-request", "--out", target + "/result"]
            if mode == "on":
                step += ["--iket", target + "/sources-run", "--ctas", "1", "--iters", "1",
                         "--warps", "1", "--granularity", "phase", "--payloads",
                         "--iket-cache-dir", f"/c/tilelang-on-tp{shape}", "--skip", "^(vision_|topk_topp)"]
                step = ["python3", "/b/harness/iket_profile.py", target + "/profile",
                        "--timeout", "1800", "--log-level", os.environ.get("IKET_LOG_LEVEL", "error"), "--single-pass-max-ts", os.environ.get("IKET_TS", "64"),
                        "--skip-raw-data-dump", "--pool-gb", os.environ.get("IKET_POOL_GB", "14"), "--", *step]
            elif mode == "warp":
                step = ["python3", "/b/harness/iket_profile.py", target + "/profile",
                        "--timeout", "1800", "--log-level", os.environ.get("IKET_WARP_LOG_LEVEL", "error"),
                        "--warp-trace", "--skip-raw-data-dump",
                        "--pool-gb", os.environ.get("IKET_WARP_POOL_GB", "5"), "--", *step]
            elif mode == "nsys":
                (OUT / label).mkdir()
                step = ["/opt/nsys/bin/nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                        "--cuda-graph-trace=node", "--export=sqlite", "--force-overwrite=true",
                        "--output", target + "/timeline", *step]
            memory = os.environ.get("IKET_MEMORY_GB", "100") + "g"
            argv = ["docker", "run", "--rm", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
                    "--memory=" + memory, "--memory-swap=" + memory, "--ulimit=core=0",
                    "--cpuset-cpus=5-9,15-19", "-v", f"{HARNESS}:/b:ro", "-v", f"{REPO}:/r:ro",
                    "-v", f"{HOME_DIR / '.cache/huggingface'}:/hf:ro",
                    "-v", f"{OUT}:/out", "-v", f"{CACHE}:/c"]
            if mode == "nsys":
                argv += ["-v", "/opt/nvidia/nsight-systems/2025.3.2:/opt/nsys:ro"]
            for key, value in env.items():
                argv += ["-e", f"{key}={value}"]
            argv += ["--entrypoint", "sh", IMAGE, "-c",
                     'uv venv --system-site-packages /tmp/.venv >&2 && exec "$@"', "sh", *step]
            write_json(OUT / f"{label}.command.json", argv)
            started = time.monotonic()
            print(stamp(), "START", label, flush=True)
            active.set()
            with (OUT / f"{label}.log").open("w") as handle:
                try:
                    proc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, timeout=2100)
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    subprocess.run(["docker", "kill", CONTAINER], capture_output=True)
                    code = 124
            active.clear()
            item = {"label": label, "exit": code, "wall_s": round(time.monotonic() - started, 3),
                    "memory_guard_triggered": guard_failure.is_set()}
            result_path = OUT / label / "result/result.json"
            if result_path.exists():
                item["result"] = json.loads(result_path.read_text())
            results.append(item)
            write_json(OUT / "runs.json", results)
            print(stamp(), "END", json.dumps(item), flush=True)
            if code or guard_failure.is_set():
                raise RuntimeError(f"{label} failed; receipt retained")
            if request.exists() and len(results) < len(jobs):
                queued = json.loads(request.read_text())
                if queued.get("who") not in (HOLDER, "tilelang-runtime-closure"):
                    write_json(OUT / "yielded.json", dict(ended=stamp(), request=queued, pending=jobs[len(results):]))
                    return 75
        write_json(OUT / "completed.json", {"finished": stamp(), "jobs": len(results)})
    finally:
        stopped.set()
        thread.join(5)
        subprocess.run(["docker", "kill", CONTAINER], capture_output=True)
        if HOLD.exists() and json.loads(HOLD.read_text()).get("holder") == HOLDER:
            HOLD.unlink()
        print(stamp(), "reservation released; serving clusters untouched", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
