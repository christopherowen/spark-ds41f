"""Coordinated single-GPU sp8 runner for optional-lsa correctness gates and 2-boot benchmark."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT_DIR = EXP / "runs/gpu-gates-01"
HOLD_PATH = Path.home() / "spark-hold.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(ref: str = "HEAD") -> str:
    return (
        subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()
    )


def gpu_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.current.graphics,power.draw,display_active",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def memory_snapshot() -> dict:
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    mem = subprocess.check_output(["free", "-b"], text=True).strip()
    return {"at": now_iso(), "gpu_apps": apps.splitlines() if apps else [], "free_b": mem}


def resolve_mount(src: str) -> str:
    if src.startswith("~/"):
        return str(Path.home() / src[2:])
    if src.startswith("/"):
        return src
    return str(ROOT / src)


def run_stage(stage_name: str, job_dir_name: str, token: str, commit: str, refresh_fn):
    refresh_fn()
    spec = json.loads((EXP / job_dir_name / "candidate.json").read_text())
    cname = f"opt-lsa-{stage_name}-{token[:8]}"
    argv = [
        "docker",
        "run",
        "--name",
        cname,
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "16g",
        "--pids-limit",
        "256",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit=core=0",
    ]
    for k, v in sorted(spec["env"].items()):
        argv.extend(["-e", f"{k}={v}"])
    for src, dst in spec["mounts"]:
        argv.extend(["-v", f"{resolve_mount(src)}:{dst}:ro"])
    cache_dir = Path.home() / ".cache/spark-lab/compile"
    cache_dir.mkdir(parents=True, exist_ok=True)
    argv.extend(
        [
            "-v",
            f"{cache_dir}:/c",
            "-w",
            "/opt/spark-ds41f/candidate/vllm",
            "--entrypoint",
            "python3",
            spec["image"],
            *spec["argv"],
        ]
    )
    cmd_rec = {
        "argv": argv,
        "cwd": str(ROOT),
        "source_commit": commit,
        "at": now_iso(),
    }
    (OUT_DIR / f"{stage_name}-command.json").write_text(json.dumps(cmd_rec, indent=2) + "\n")

    start_gpu = gpu_snapshot()
    start_temp = float(start_gpu.split(",")[0].strip())
    t0 = time.monotonic()
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.monotonic() - t0
    end_gpu = gpu_snapshot()
    refresh_fn()

    combined_out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    (OUT_DIR / f"{stage_name}-output.txt").write_text(combined_out)
    verdict = {
        "exit": proc.returncode,
        "seconds": elapsed,
        "at": now_iso(),
        "start_hottest_c": start_temp,
        "end_gpu": end_gpu,
    }
    (OUT_DIR / f"{stage_name}-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Stage {stage_name} failed with exit {proc.returncode}:\n{combined_out[-2000:]}")

    result_lines = [
        line[len("RESULT ") :]
        for line in proc.stdout.splitlines()
        if line.startswith("RESULT ")
    ]
    assert result_lines, f"No RESULT line in {stage_name} output"
    parsed = json.loads(result_lines[-1])
    assert parsed.get("passed") is True, f"Stage {stage_name} did not report passed=True"
    (OUT_DIR / f"{stage_name}.json").write_text(json.dumps(parsed, indent=2) + "\n")
    print(f"STAGE_OK {stage_name} ({elapsed:.2f}s)", flush=True)
    return parsed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=str, default="")
    args = ap.parse_args()
    selected = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = git_head("HEAD")
    pub_commit = git_head("origin/latent-sparse-attention")
    assert commit == pub_commit, f"HEAD {commit} != origin/latent-sparse-attention {pub_commit}"

    token = f"optional-lsa-{uuid.uuid4().hex[:12]}"
    since = now_iso()
    lease = {
        "holder": "optional-lsa-gpu-gates",
        "token": token,
        "nodes": ["sp8"],
        "pool": "sp8-single",
        "since": since,
        "expected_end": None,
        "heartbeat": since,
        "controller": str(ROOT),
        "controller_commit": commit,
        "restore_profile": "none (standalone sp8 kernel screen; no serving container on sp8)",
        "rule": "Exclusive sp8 single-GPU window for PR #877 optional non-quantized BF16 TileLang LSA qualification and 2-boot benchmark; auto-releases if heartbeat > 15m old.",
    }

    fd = os.open(str(HOLD_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(lease, indent=2) + "\n")

    stop_event = threading.Event()

    def refresh_heartbeat():
        if HOLD_PATH.exists():
            current = json.loads(HOLD_PATH.read_text())
            if current.get("token") == token:
                current["heartbeat"] = now_iso()
                tmp = HOLD_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(current, indent=2) + "\n")
                tmp.replace(HOLD_PATH)

    def watchdog_loop():
        while not stop_event.wait(30.0):
            try:
                refresh_heartbeat()
            except Exception:
                pass

    wd_thread = threading.Thread(target=watchdog_loop, daemon=True)
    wd_thread.start()

    try:
        (OUT_DIR / "lease.json").write_text(json.dumps(lease, indent=2) + "\n")
        pub_rec = {
            "at": now_iso(),
            "source_commit": commit,
            "published_ref": "origin/latent-sparse-attention",
            "published_head": pub_commit,
            "source_reachable": True,
        }
        (OUT_DIR / "publication-check.json").write_text(json.dumps(pub_rec, indent=2) + "\n")
        (OUT_DIR / "memory-before.json").write_text(json.dumps(memory_snapshot()) + "\n")

        stages = [
            ("summary-initcheck", "sp8-summary-initcheck"),
            ("summary-racecheck", "sp8-summary-racecheck"),
            ("summary-timing", "sp8-summary-timing"),
            ("registration", "sp8-registration"),
            ("actual", "sp8-actual"),
            ("initcheck", "sp8-initcheck"),
            ("graph", "sp8-graph"),
            ("bench-boot1", "sp8-bench-boot1"),
            ("bench-boot2", "sp8-bench-boot2"),
        ]
        for s_name, j_dir in stages:
            if selected is None or s_name in selected:
                run_stage(s_name, j_dir, token, commit, refresh_heartbeat)

        (OUT_DIR / "memory-after.json").write_text(json.dumps(memory_snapshot()) + "\n")
    finally:
        stop_event.set()
        if HOLD_PATH.exists():
            try:
                current = json.loads(HOLD_PATH.read_text())
                if current.get("token") == token:
                    HOLD_PATH.unlink()
            except Exception:
                pass

    rem = subprocess.check_output(
        ["docker", "ps", "--filter", f"name=opt-lsa-", "--format", "{{.Names}}"],
        text=True,
    ).strip()
    released = {
        "at": now_iso(),
        "removed": not HOLD_PATH.exists(),
        "remaining_containers": rem.splitlines() if rem else [],
    }
    (OUT_DIR / "released.json").write_text(json.dumps(released, indent=2) + "\n")
    indep = {
        "at": now_iso(),
        "own_lease_and_container_removed": not HOLD_PATH.exists() and not rem,
        "sp8_hold_present": HOLD_PATH.exists(),
    }
    (OUT_DIR / "independent-release.json").write_text(json.dumps(indep, indent=2) + "\n")
    print("ALL_STAGES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
