#!/usr/bin/env python3
"""Detached 2-boot GPU verification and benchmark runner on sp8."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HOLD_PATH = Path.home() / "spark-hold.json"
REQ_PATH = Path.home() / "spark-request.json"
WORKDIR = Path("/home/chris/spark-lab/rework-cross-topology-parity")
VLLM_DIR = WORKDIR / "vllm"
HOLDER = "rework-cross-topology-parity-agent34"
IMAGE = "spark-ds41f:candidate-38"


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(ts: str) -> float:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def acquire_hold() -> None:
    if not REQ_PATH.exists():
        REQ_PATH.write_text(
            json.dumps(
                {
                    "who": HOLDER,
                    "why": "Scoped 2-boot GPU verification on sp8 for cross-topology parity (0655/0666/0672)",
                    "how_long": "90s",
                    "requested_at": now_iso(),
                },
                indent=2,
            )
            + "\n"
        )
    while True:
        if not HOLD_PATH.exists():
            now = now_iso()
            payload = {
                "holder": HOLDER,
                "scope": ["sp8"],
                "nodes": ["sp8"],
                "since": now,
                "expected_end": None,
                "heartbeat": now,
                "rule": "Scoped single-node 2-boot container verification on sp8 for feat/cross-topology-parity",
            }
            tmp = HOLD_PATH.with_suffix(".tmp.agent34")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            try:
                os.link(tmp, HOLD_PATH)
                tmp.unlink(missing_ok=True)
                REQ_PATH.unlink(missing_ok=True)
                print(f"[{now_iso()}] Acquired {HOLD_PATH}", flush=True)
                return
            except FileExistsError:
                tmp.unlink(missing_ok=True)
        else:
            try:
                data = json.loads(HOLD_PATH.read_text())
                if data.get("holder") == HOLDER:
                    REQ_PATH.unlink(missing_ok=True)
                    return
                hb = data.get("heartbeat") or data.get("since")
                age_s = time.time() - parse_iso(hb) if hb else 0.0
                if age_s > 900.0:
                    containers = subprocess.check_output(
                        ["docker", "ps", "-q"], text=True
                    ).strip()
                    if not containers:
                        print(
                            f"[{now_iso()}] Hold by {data.get('holder')} expired (age {age_s:.0f}s > 900s, 0 containers); removing stale hold.",
                            flush=True,
                        )
                        HOLD_PATH.unlink(missing_ok=True)
                        continue
                print(
                    f"[{now_iso()}] Waiting for hold held by {data.get('holder')} (age {age_s:.0f}s)...",
                    flush=True,
                )
            except Exception as e:
                print(f"[{now_iso()}] Hold read warning: {e}", flush=True)
        time.sleep(3.0)


def refresh_heartbeat() -> None:
    try:
        if HOLD_PATH.exists():
            data = json.loads(HOLD_PATH.read_text())
            if data.get("holder") == HOLDER:
                data["heartbeat"] = now_iso()
                HOLD_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except Exception:
        pass


def release_hold() -> None:
    try:
        if HOLD_PATH.exists():
            data = json.loads(HOLD_PATH.read_text())
            if data.get("holder") == HOLDER:
                HOLD_PATH.unlink(missing_ok=True)
                print(f"[{now_iso()}] Released {HOLD_PATH}", flush=True)
    except Exception as e:
        print(f"[{now_iso()}] Error releasing hold: {e}", flush=True)


def run_boot(boot_num: int) -> dict:
    refresh_heartbeat()
    print(f"\n=== Starting Container Boot {boot_num} using {IMAGE} ===", flush=True)
    t0 = time.monotonic()
    boot_json = f"/out/boot{boot_num}_results.json"
    boot_log = WORKDIR / f"boot{boot_num}.log"
    inner_script = f"""
    set -euo pipefail
    VLLM_PKG=$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)')
    VLLM_ROOT=$(dirname "$VLLM_PKG")
    cp -rf /src/vllm/* "$VLLM_PKG/"
    cp -rf /src/tests/* "$VLLM_ROOT/tests/"
    cd "$VLLM_ROOT"
    pytest --noconftest -q \\
      tests/models/test_deepseek_v4_1_compressor_prepare.py \\
      tests/models/test_deepseek_v4_1_compressor.py \\
      tests/models/test_deepseek_v4_1_attention.py
    pytest --noconftest -q \\
      tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \\
      -k "test_split_linear or test_packed_bf16_compressor_shapes_exact or test_sharded_wqa_wkv_and_strided_attention_front_match_replicated"
    python3 -u /out/bench_cross_topology_parity.py {boot_json}
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "-v",
        f"{VLLM_DIR}:/src:ro",
        "-v",
        f"{WORKDIR}:/out",
        "-v",
        "/tmp/rework_parity_tl_cache:/root/.tilelang",
        "--entrypoint",
        "bash",
        IMAGE,
        "-lc",
        inner_script,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    boot_log.write_text(proc.stdout)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Boot {boot_num} failed with exit code {proc.returncode}")
    elapsed = round(time.monotonic() - t0, 2)
    print(f"=== Completed Container Boot {boot_num} in {elapsed}s ===", flush=True)
    return {
        "boot": boot_num,
        "elapsed_s": elapsed,
        "results_file": str(WORKDIR / f"boot{boot_num}_results.json"),
    }


def main() -> None:
    stop_hb = threading.Event()

    def heartbeat_loop():
        while not stop_hb.wait(30):
            refresh_heartbeat()

    acquire_hold()
    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    status = {"state": "running", "started_at": now_iso()}
    (WORKDIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")

    try:
        b1 = run_boot(1)
        b2 = run_boot(2)
        status.update(
            {
                "state": "completed",
                "completed_at": now_iso(),
                "image": IMAGE,
                "boots": [b1, b2],
            }
        )
        (WORKDIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{now_iso()}] ALL 2 BOOTS SUCCEEDED!", flush=True)
    except Exception as e:
        status.update(
            {
                "state": "failed",
                "failed_at": now_iso(),
                "error": str(e),
            }
        )
        (WORKDIR / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{now_iso()}] FAILED: {e}", flush=True)
        raise
    finally:
        stop_hb.set()
        release_hold()


if __name__ == "__main__":
    main()
