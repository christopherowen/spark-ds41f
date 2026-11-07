#!/usr/bin/env python3
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HOLD_PATH = Path("/home/chris/spark-hold.json")
OUT_DIR = Path("/home/chris/spark-lab/rework-0741-lsa")
HOLDER = "optional-lsa-0741-polish"
TOKEN = str(uuid.uuid4())


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_idle() -> bool:
    res = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return res.returncode == 0 and not res.stdout.strip()


def acquire_hold() -> dict:
    while True:
        if not HOLD_PATH.exists() and gpu_idle():
            payload = {
                "holder": HOLDER,
                "token": TOKEN,
                "nodes": ["sp8"],
                "scope": ["sp8"],
                "pool": "sp8-single",
                "since": utc_now(),
                "expected_end": None,
                "heartbeat": utc_now(),
                "rule": "Scoped single-node 2-boot GPU unit test verification on sp8 for 0741 VLLM_DS41_LSA polish",
            }
            try:
                fd = os.open(str(HOLD_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w") as f:
                    json.dump(payload, f, indent=2)
                    f.write("\n")
                print(f"[{utc_now()}] Acquired {HOLD_PATH}", flush=True)
                return payload
            except FileExistsError:
                pass
        time.sleep(3)


def run_boot(boot_idx: int) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--entrypoint",
        "python3",
        "-v",
        "/home/chris/spark-lab/rework-0741-lsa/vllm/vllm/models/deepseek_v4_1:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1:ro",
        "-v",
        "/home/chris/spark-lab/rework-0741-lsa/vllm/tests/models/test_deepseek_v4_1_lsa.py:/opt/spark-ds41f/candidate/vllm/tests/models/test_deepseek_v4_1_lsa.py:ro",
        "-w",
        "/opt/spark-ds41f/candidate/vllm",
        "spark-ds41f:candidate-38",
        "-c",
        (
            "import pytest, sys; "
            "sys.exit(pytest.main(["
            "\"-v\", \"--noconftest\", "
            "\"-p\", \"tests.kernels.deepseek_v41_pdl_sources\", "
            "\"tests/models/test_deepseek_v4_1_lsa.py\", "
            "\"tests/models/test_deepseek_v4_1_attention.py\""
            "]))"
        ),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_s = round(time.monotonic() - t0, 3)
    log_file = OUT_DIR / f"gpu_pytest_boot{boot_idx}.log"
    log_file.write_text(proc.stdout + "\n" + proc.stderr)
    print(f"[{utc_now()}] Boot {boot_idx} rc={proc.returncode} elapsed={elapsed_s}s", flush=True)
    return {
        "boot": boot_idx,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "log_file": str(log_file),
        "tail": (proc.stdout + "\n" + proc.stderr).splitlines()[-25:],
    }


def main() -> int:
    payload = acquire_hold()
    stop_evt = threading.Event()

    def heartbeat_loop():
        while not stop_evt.wait(30.0):
            try:
                if HOLD_PATH.exists():
                    cur = json.loads(HOLD_PATH.read_text())
                    if cur.get("token") == TOKEN:
                        cur["heartbeat"] = utc_now()
                        tmp = HOLD_PATH.with_suffix(f".tmp.{os.getpid()}")
                        tmp.write_text(json.dumps(cur, indent=2) + "\n")
                        tmp.replace(HOLD_PATH)
            except Exception:
                pass

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    try:
        b1 = run_boot(1)
        if b1["returncode"] != 0:
            (OUT_DIR / "verification_2boot.json").write_text(json.dumps({"boot1": b1}, indent=2) + "\n")
            return b1["returncode"]
        b2 = run_boot(2)
        summary = {"holder": HOLDER, "token": TOKEN, "boot1": b1, "boot2": b2}
        (OUT_DIR / "verification_2boot.json").write_text(json.dumps(summary, indent=2) + "\n")
        return b2["returncode"]
    finally:
        stop_evt.set()
        try:
            if HOLD_PATH.exists():
                cur = json.loads(HOLD_PATH.read_text())
                if cur.get("token") == TOKEN:
                    HOLD_PATH.unlink()
                    print(f"[{utc_now()}] Released {HOLD_PATH}", flush=True)
        except Exception as e:
            print(f"Failed to release hold: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
