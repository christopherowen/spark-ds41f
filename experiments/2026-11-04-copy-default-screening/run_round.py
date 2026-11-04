#!/usr/bin/env python3
"""Coordinate the authorized candidate-39 build, gates and two-boot TP3 metrics."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab

EXPERIMENT = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-39/tp3.json"
OUT = ROOT / "results/private/lab/candidate-39-round"
BUILD_ROOT = "/home/swank/projects/spark-ds41f-copy-default"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        print(json.dumps({"restore": RESTORE, "candidate": CANDIDATE, "build_host": "dgx4", "serving_nodes": ["dgx1", "dgx2", "dgx3"], "specs": ["functional.json", "performance.json"], "output": str(OUT), "shape_change": False}, indent=2))
        return 0
    if os.uname().nodename != "dgx1" or OUT.exists() or lab.read_hold():
        raise SystemExit("wrong host, prior receipts or an existing reservation; do not duplicate")
    lab.PRODUCTION_CONFIG = RESTORE
    lab.window_open(0, "owner-authorized candidate-39 defaults; GPU4 build/gate and TP3 repeat metrics")
    window = lab.read_hold()["window"]
    OUT.mkdir(parents=True)
    receipt = {"started": lab.now(), "window": window, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "steps": [], "restored": False}
    done = threading.Event()
    management_lost = False
    def owns():
        hold = lab.read_hold()
        return bool(hold and hold.get("window") == window and lab.hold_is_ours(hold))
    def save():
        lab.write_json_atomic(OUT / "receipt.json", receipt)
    def heartbeat():
        while not done.wait(30):
            if owns():
                lab.beat(receipt.get("phase", "candidate-39"))
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    def run(argv, name, allow_closing=False):
        nonlocal management_lost
        if not owns() or (not allow_closing and lab.window_should_close(lab.read_hold())):
            raise RuntimeError("window changed or closure requested")
        receipt["phase"] = name
        record = {"argv": argv, "started": lab.now(), "log": name + ".log"}
        receipt["steps"].append(record); save()
        with (OUT / record["log"]).open("w") as log:
            process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit=process.returncode, ended=lab.now()); save()
        if process.returncode:
            detail = (OUT / record["log"]).read_text(errors="replace").lower()
            management_lost = management_lost or (argv[0] == "ssh" and process.returncode == 255) or any(
                text in detail for text in ("ssh: connect", "cannot reach", "kex_exchange_identification", "connection timed out during banner exchange"))
            raise RuntimeError(f"{name} exited {process.returncode}")
    def cli(config, operation, action, *extra):
        argv = [sys.executable, "bin/spark", "--cluster-config", config, operation, action, *extra]
        plan = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
        lab.write_json_atomic(OUT / f"{operation}-{action}-plan.json", {"argv": argv, "exit": plan.returncode, "stdout": plan.stdout, "stderr": plan.stderr})
        print(plan.stdout, flush=True)
        if plan.returncode or "--apply would refuse" in plan.stdout or "shape:   CHANGED" in plan.stdout:
            raise RuntimeError("plan refused; no shape bypass")
        run([*argv, "--apply"], operation + "-" + action, allow_closing=config == RESTORE)
    code = 1
    try:
        command = ["python3", f"{EXPERIMENT}/build_gate.py", "--window", window, "--apply"]
        remote = f"cd {shlex.quote(BUILD_ROOT)} && test $(git rev-parse HEAD) = {shlex.quote(receipt['commit'])} && " + shlex.join(command)
        run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx4", remote], "image-and-gate")
        raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "dgx4", f"cat {BUILD_ROOT}/results/private/candidate-rounds/candidate-39-image/receipt.json"], text=True)
        image = json.loads(raw)
        lab.write_json_atomic(OUT / "image-receipt.json", image)
        if not image.get("passed") or not image.get("gate", {}).get("passed"):
            raise RuntimeError("image gate lacks passing receipt")
        run(["bash", "-o", "pipefail", "-c", "ssh -o BatchMode=yes -o ConnectTimeout=8 dgx4 docker save spark-ds41f:candidate-39 | docker load"], "seed-image")
        actual = subprocess.check_output(["docker", "image", "inspect", "spark-ds41f:candidate-39", "--format", "{{.Id}}"], text=True).strip()
        if actual != image["image"]:
            raise RuntimeError("head image digest differs from build receipt")
        cli(CANDIDATE, "build", "distribute")
        for spec in ["functional.json", "performance.json"]:
            run([sys.executable, "scripts/lab.py", "run", f"{EXPERIMENT}/{spec}", "--production-config", RESTORE, "--keep-open"], spec.removesuffix(".json"))
        receipt["completed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        receipt["phase"] = "restoration"
        try:
            unreachable = []
            for host in ["dgx1", "dgx2", "dgx3", "dgx4"]:
                try:
                    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "true"], capture_output=True, timeout=15)
                    failed = p.returncode != 0
                except subprocess.TimeoutExpired:
                    failed = True
                if failed:
                    unreachable.append(host)
            if management_lost or unreachable:
                if owns():
                    hold = lab.read_hold(); hold["holder"] = "candidate-round-hardware-review"
                    hold["rule"] = "management unavailable; no launch until isolated and explicitly authorized"
                    lab.write_json_atomic(lab.HOLD, hold)
                receipt["management_unreachable"] = unreachable
                receipt["management_loss_latched"] = management_lost
                raise RuntimeError("management loss: reservation preserved; no restoration launch")
            if owns():
                if not lab.production_live():
                    cli(RESTORE, "cluster", "start", "--replace")
                receipt["restored"] = lab.window_close("candidate-39 round ended")
            elif not lab.read_hold() and lab.production_live():
                receipt["restored"] = True
            else:
                raise RuntimeError("window ownership changed; no serving action attempted")
            if not receipt["restored"]:
                raise RuntimeError("restoration verification failed; retain reservation")
        except BaseException as error:
            receipt["restore_error"] = f"{type(error).__name__}: {error}"
            code = 1
        done.set(); thread.join(timeout=2)
        receipt["ended"] = lab.now(); save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
