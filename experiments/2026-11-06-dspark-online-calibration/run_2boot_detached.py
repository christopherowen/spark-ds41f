import datetime
import json
import os
import subprocess
import time
from pathlib import Path

WORKDIR = Path("/home/chris/spark-lab/rework-686-693-694")
VLLM_DIR = WORKDIR / "vllm"
HOLD_PATH = Path.home() / "spark-hold.json"
REQ_PATH = Path.home() / "spark-request.json"
HOLDER = "adaptive-verification-rework"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(ts: str) -> float:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def acquire_hold() -> None:
    if not REQ_PATH.exists():
        REQ_PATH.write_text(
            json.dumps(
                {
                    "who": HOLDER,
                    "why": "Scoped 2-boot unit and kernel verification on sp8 (< 60s)",
                    "how_long": "60s",
                    "requested_at": utc_now(),
                },
                indent=2,
            )
            + "\n"
        )
    while True:
        if not HOLD_PATH.exists():
            now = utc_now()
            payload = {
                "holder": HOLDER,
                "scope": ["sp8"],
                "since": now,
                "expected_end": None,
                "heartbeat": now,
                "rule": "Scoped single-node 2-boot container verification on sp8 for feat/adaptive-verification-rework",
            }
            tmp = HOLD_PATH.with_suffix(".tmp.rework26")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            try:
                os.link(tmp, HOLD_PATH)
                tmp.unlink(missing_ok=True)
                REQ_PATH.unlink(missing_ok=True)
                print(f"[{utc_now()}] Acquired ~/spark-hold.json on sp8", flush=True)
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
                    containers = subprocess.check_output(["docker", "ps", "-q"], text=True).strip()
                    if not containers:
                        print(
                            f"[{utc_now()}] Hold by {data.get('holder')} expired (heartbeat age {age_s:.0f}s > 900s, 0 containers); removing stale hold.",
                            flush=True,
                        )
                        HOLD_PATH.unlink(missing_ok=True)
                        continue
                print(
                    f"[{utc_now()}] Waiting for hold held by {data.get('holder')} (heartbeat age {age_s:.0f}s)...",
                    flush=True,
                )
            except Exception as e:
                print(f"[{utc_now()}] Hold check warning: {e}", flush=True)
        time.sleep(3.0)


def refresh_heartbeat() -> None:
    try:
        if HOLD_PATH.exists():
            data = json.loads(HOLD_PATH.read_text())
            if data.get("holder") == HOLDER:
                data["heartbeat"] = utc_now()
                HOLD_PATH.write_text(json.dumps(data, indent=2) + "\n")
    except Exception:
        pass


def release_hold() -> None:
    try:
        if HOLD_PATH.exists():
            data = json.loads(HOLD_PATH.read_text())
            if data.get("holder") == HOLDER:
                HOLD_PATH.unlink(missing_ok=True)
                print(f"[{utc_now()}] Released ~/spark-hold.json on sp8", flush=True)
    except Exception:
        pass


def run_boot(boot_num: int) -> dict:
    refresh_heartbeat()
    log_path = WORKDIR / f"boot{boot_num}.log"
    metrics_path = WORKDIR / f"boot{boot_num}_metrics.json"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--entrypoint",
        "bash",
        "-v",
        f"{VLLM_DIR}:/src:ro",
        "-v",
        f"{WORKDIR}:/work", "-v", "/tmp/rework_tl_cache:/root/.tilelang",
        "spark-ds41f:candidate-38",
        "-lc",
        f"""
        set -euo pipefail
        VLLM_PKG=$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)')
        cp -rf /src/vllm/* "$VLLM_PKG/"
        mkdir -p /tmp/rework_tests
        cp /src/tests/v1/spec_decode/test_adaptive_verification.py /tmp/rework_tests/
        cp /src/tests/v1/spec_decode/test_acceptance_estimator.py /tmp/rework_tests/
        cp /src/tests/v1/worker/test_dspark_greedy_markov.py /tmp/rework_tests/
        cp /src/tests/v1/spec_decode/test_copy_drafts.py /tmp/rework_tests/
        cp /src/tests/v1/worker/test_dead_rows.py /tmp/rework_tests/
        cd /tmp/rework_tests
        pytest --noconftest -q \\
          test_adaptive_verification.py \\
          test_acceptance_estimator.py \\
          test_dspark_greedy_markov.py \\
          test_copy_drafts.py \\
          test_dead_rows.py
        python3 /work/verify_rework_gpu.py > /work/boot{boot_num}_metrics.json
        cat /work/boot{boot_num}_metrics.json
        """,
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed_s = round(time.monotonic() - t0, 2)
    log_path.write_text(proc.stdout)
    print(f"[{utc_now()}] Boot {boot_num} finished with rc={proc.returncode} in {elapsed_s}s", flush=True)
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Boot {boot_num} failed with exit code {proc.returncode}")
    return {
        "boot": boot_num,
        "elapsed_s": elapsed_s,
        "metrics": json.loads(metrics_path.read_text()),
    }


def main() -> None:
    acquire_hold()
    watchdog_code = (
        "import json, os, pathlib, time\n"
        f"pid = {os.getpid()}\n"
        f"p = pathlib.Path({str(HOLD_PATH)!r})\n"
        "while True:\n"
        "    time.sleep(5)\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except OSError:\n"
        "        try:\n"
        "            if p.exists() and json.loads(p.read_text()).get('holder') == 'adaptive-verification-rework':\n"
        "                p.unlink(missing_ok=True)\n"
        "        except Exception:\n"
        "            pass\n"
        "        break\n"
    )
    subprocess.Popen(
        ["python3", "-c", watchdog_code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        b1 = run_boot(1)
        b2 = run_boot(2)
        summary = {
            "completed_at": utc_now(),
            "status": "PASS",
            "boots": [b1, b2],
        }
        (WORKDIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[{utc_now()}] WROTE SUMMARY.JSON SUCCESSFULLY", flush=True)
    finally:
        release_hold()


if __name__ == "__main__":
    main()
