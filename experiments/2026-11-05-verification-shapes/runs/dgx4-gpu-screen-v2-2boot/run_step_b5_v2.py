"""Detached Step B5 v2 runner on dgx4 for PR #921 — 2-boot 0-error initcheck screen + Sparse-MLA benchmark."""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HOLD_PATH = pathlib.Path.home() / "spark-hold.json"
HOLDER = "spark-lab: verification capture policies"
WINDOW = "dgx4-verification-shapes-20261106"
TOKEN = "dgx4-verification-shapes-20261106"
WORK_DIR = pathlib.Path("/home/swank/spark-work/verification-shapes/dgx4-step-b5")
ROOT = pathlib.Path("/home/swank/spark-work/verification-shapes")
DEP = ROOT / "deployment"
TRANSFER = ROOT / "image-transfer-exact-native-1"
VLLM = ROOT / "vllm"
RESULTS_ROOT = DEP / "results/private/verification-shapes"
IMAGE_ID = "sha256:3e03ac089d50d1408272aac0dc32ab134109e228a1d8dc8f8310688ccdb16209"
COMMIT = '167481c083307694962a24c0ce6556272faee94c'

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    print(f"[{utc_now()}] {msg}", flush=True)

def acquire_or_adopt_hold():
    while True:
        if HOLD_PATH.exists():
            try:
                existing = json.loads(HOLD_PATH.read_text())
                if existing.get("token") == TOKEN and existing.get("holder") == HOLDER:
                    existing["heartbeat"] = utc_now()
                    tmp = HOLD_PATH.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(existing, indent=2) + "\n")
                    os.replace(tmp, HOLD_PATH)
                    log(f"Adopted our existing {HOLD_PATH} with token {TOKEN}")
                    return
            except Exception:
                pass
        else:
            containers = subprocess.check_output(["docker", "ps", "-q"], text=True).strip()
            gpu_procs = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip()
            if not containers and not gpu_procs:
                now = utc_now()
                exp = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
                payload = {
                    "holder": HOLDER,
                    "window": WINDOW,
                    "token": TOKEN,
                    "pool": "dgx4-single",
                    "nodes": ["dgx4"],
                    "controller": f"dgx4:{DEP}",
                    "controller_commit": COMMIT,
                    "restore_profile": "dgx4-idle-standalone",
                    "restore_commit": COMMIT,
                    "since": now,
                    "expected_end": exp,
                    "heartbeat": now,
                    "watchdog": "15-minute stale-heartbeat watchdog on dgx4",
                    "rule": "node-scoped hold on dgx4 (dgx4-single); no other GPU experiments on dgx4 while this file exists",
                    "current": "Step B5 v2: PR #921 2-boot 0-error full initcheck screen + Sparse-MLA verification shape benchmark on dgx4",
                }
                try:
                    fd = os.open(str(HOLD_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    with os.fdopen(fd, "w") as f:
                        f.write(json.dumps(payload, indent=2) + "\n")
                    log(f"Acquired {HOLD_PATH} with token {TOKEN}")
                    return
                except FileExistsError:
                    pass
        log("Waiting for dgx4:~/spark-hold.json and GPU idle...")
        time.sleep(10)

def start_watchdog():
    subprocess.Popen(
        [sys.executable, str(WORK_DIR / "watchdog.py")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

def main():
    acquire_or_adopt_hold()
    stop_ev = threading.Event()
    current_state = ["starting Boot 1 gpu_gate.py --screen-only"]
    hb_thread = None
    status = 1
    summary = {
        "started": utc_now(),
        "token": TOKEN,
        "window": WINDOW,
        "deployment_commit": COMMIT,
        "source_head": "8dea43f62a149a3aef2be4b24fa1c68e9b33407d",
        "paired_patch": "0003-bound-mxfp8-live-rows.patch (2541c9a1372f7f35589b0aafc343b00711999de9)",
        "image": IMAGE_ID,
        "boots": {},
    }
    try:
        start_watchdog()

        def beat_loop():
            while not stop_ev.wait(20):
                try:
                    if not HOLD_PATH.exists():
                        return
                    data = json.loads(HOLD_PATH.read_text())
                    if data.get("token") != TOKEN:
                        return
                    data["heartbeat"] = utc_now()
                    data["current"] = current_state[0]
                    tmp = HOLD_PATH.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(data, indent=2) + "\n")
                    os.replace(tmp, HOLD_PATH)
                except Exception as e:
                    log(f"Heartbeat warning: {e}")

        hb_thread = threading.Thread(target=beat_loop, daemon=True)
        hb_thread.start()

        sys.path.insert(0, str(DEP))
        from spark_ds41f import lab
        src_root = lab.image_source_root(IMAGE_ID)
        paired_gemm = DEP / "experiments/2026-11-05-verification-shapes/runs/paired-mxfp8-live-rows/gemm.py"
        shared_bench_cache = RESULTS_ROOT / "native-capture-screen-v2-boot-1/cache"

        for boot in (1, 2):
            run_id = f"native-capture-screen-v2-boot-{boot}"
            out_dir = RESULTS_ROOT / run_id
            current_state[0] = f"PR #921 Step B5 v2: running gpu_gate.py --screen-only boot {boot}"
            log(f"=== Starting gpu_gate.py --screen-only Boot {boot} -> {out_dir} ===")
            cmd = [
                sys.executable,
                str(DEP / "experiments/2026-11-05-verification-shapes/gpu_gate.py"),
                "--apply",
                "--head", "dgx4",
                "--window", WINDOW,
                "--source", str(VLLM),
                "--build-receipt", str(TRANSFER / "receipt.json"),
                "--built-inspection", str(TRANSFER / "built-inspection.json"),
                "--out", str(out_dir),
                "--screen-only",
            ]
            subprocess.run(cmd, cwd=DEP, check=True)
            log(f"=== Completed gpu_gate.py --screen-only Boot {boot} (screen_passed=True) ===")

            bench_out = out_dir / f"shape-benchmark-boot-{boot}.json"
            current_state[0] = f"PR #921 Step B5 v2: running Sparse-MLA + shape replay benchmark boot {boot}"
            log(f"=== Starting Sparse-MLA + shape replay benchmark Boot {boot} -> {bench_out} ===")
            docker_cmd = [
                "docker", "run", "--rm",
                "--name", f"verification-shapes-bench-boot-{boot}",
                "--gpus", "all", "--cpus=2", "--memory=32g", "--memory-swap=32g",
                "--ipc=host", "--ulimit=core=0", "--security-opt=seccomp=unconfined", "--ulimit=memlock=-1:-1",
                "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
                "-e", "OMP_NUM_THREADS=1", "-e", "MAX_JOBS=1", "-e", "TILELANG_CACHE_DIR=/c/tilelang",
                "-v", f"{shared_bench_cache}:/c",
                "-v", f"{paired_gemm}:{src_root}/vllm/vllm/models/deepseek_v4_1/tilelang/gemm.py:ro",
                "-v", f"{WORK_DIR / 'bench_shapes.py'}:/gate/bench_shapes.py:ro",
                "-v", f"{out_dir}:/out",
                "-w", src_root + "/vllm",
                "--entrypoint", "python3",
                IMAGE_ID,
                "/gate/bench_shapes.py", str(boot), f"/out/shape-benchmark-boot-{boot}.json",
            ]
            subprocess.run(docker_cmd, cwd=DEP, check=True)
            log(f"=== Completed Sparse-MLA + shape replay benchmark Boot {boot} ===")
            summary["boots"][f"boot_{boot}"] = {
                "screen_receipt": str(out_dir / "receipt.json"),
                "benchmark_receipt": str(bench_out),
            }

        summary["passed"] = True
        status = 0
    except BaseException as exc:
        summary["passed"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        log(f"ERROR: {summary['error']}")
        raise
    finally:
        summary["ended"] = utc_now()
        (WORK_DIR / "step-b5-v2-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stop_ev.set()
        if hb_thread is not None:
            hb_thread.join(timeout=5)
        for name in (
            f"verification-shapes-gate-{WINDOW}",
            "verification-shapes-bench-boot-1",
            "verification-shapes-bench-boot-2",
        ):
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if HOLD_PATH.exists():
                h = json.loads(HOLD_PATH.read_text())
                if h.get("token") == TOKEN:
                    HOLD_PATH.unlink()
                    log(f"Released {HOLD_PATH}")
        except Exception as e:
            log(f"Failed to release hold: {e}")
    return status

if __name__ == "__main__":
    raise SystemExit(main())
