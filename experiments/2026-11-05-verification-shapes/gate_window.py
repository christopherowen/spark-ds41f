"""One DGX4 gate job inside an exclusive window, with independent cleanup."""

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from pathlib import Path

from branch_round import HOLDER, ROOT, Round, clean_revision, lab, require_owner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--restore-checkout", required=True, type=Path)
    p.add_argument(
        "--restore-config",
        default="experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json",
    )
    p.add_argument("--gate-checkout", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--build-receipt", required=True, type=Path)
    p.add_argument("--built-inspection", required=True, type=Path)
    p.add_argument("--screen-receipt", type=Path)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.run_id):
        raise RuntimeError("run ID must be a lowercase filename component")
    if not args.apply:
        print(
            json.dumps(
                {
                    "job": "release gate" if args.screen_receipt else "bounded screen",
                    "host": "dgx4",
                    "restore": args.restore_config,
                    "run": args.run_id,
                    "requires": [
                        "explicit queue handoff",
                        "no foreign hold/request",
                        "published execution commit",
                        "normal live/idle window guards",
                    ],
                }
            )
        )
        return 0
    if (
        os.uname().nodename != "dgx1"
        or lab.read_hold()
        or lab.REQUEST.exists()
        or lab.LEGACY_REQUEST.exists()
    ):
        raise RuntimeError("wrong host, held window or pending request; defer")
    revision = clean_revision(ROOT)
    restore = args.restore_checkout.resolve()
    restore_revision = clean_revision(restore)
    if restore == ROOT:
        raise RuntimeError(
            "restore controller must be an independent clean main checkout"
        )
    query = "import argparse,json,sys; from spark_ds41f import cluster as s; c,n,_=s.configuration(argparse.Namespace(cluster_config=sys.argv[1])); print(json.dumps(n))"
    nodes = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", query, args.restore_config], cwd=restore, text=True
        )
    )
    cluster, ours, _ = lab.spark.configuration(
        argparse.Namespace(cluster_config=args.restore_config)
    )
    if nodes != ours:
        raise RuntimeError("restoration node maps differ")
    for node in nodes["nodes"]:
        head, _, dirty = lab.spark.remote_repository_state(cluster, nodes, node)
        if dirty or head != restore_revision:
            raise RuntimeError("node checkouts differ from the saved clean revision")
    remote_check = shlex.join(
        ["git", "-C", str(args.gate_checkout), "rev-parse", "HEAD"]
    )
    remote_head = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "dgx4", remote_check], text=True
    ).strip()
    if remote_head != revision:
        raise RuntimeError("DGX4 gate checkout differs from the controller")
    out = ROOT / "results/private/verification-shapes" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    lab.PRODUCTION_CONFIG = args.restore_config
    subprocess.run(
        [
            sys.executable,
            str(restore / "scripts/lab.py"),
            "window",
            "open",
            "--minutes",
            "0",
            "--note",
            "verification capture policies",
            "--production-config",
            args.restore_config,
        ],
        cwd=restore,
        check=True,
    )
    opened = lab.read_hold()
    if not opened or opened.get("holder") != HOLDER:
        raise RuntimeError("normal window opening did not establish ownership")
    window = opened["window"]
    unit = "verification-shapes-gate-" + window
    state = {
        "started": lab.now(),
        "window": window,
        "deployment_commit": revision,
        "restore_checkout": str(restore),
        "restore_revision": restore_revision,
        "restore_config": args.restore_config,
        "commands": [],
        "restored": False,
        "sync_started": False,
        "remote_gate": {
            "host": "dgx4",
            "unit": unit,
            "container": unit,
            "checkout": str(args.gate_checkout),
        },
    }
    path = out / "receipt.json"
    lab.write_json_atomic(path, state)
    round = Round(path)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (out / "guardian.log").open("a") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("branch_round.py")),
                    "--guardian",
                    str(path),
                    "--parent-pid",
                    str(os.getpid()),
                    "--parent-start",
                    Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        done = threading.Event()
        round.install_lab_guards()

        def heartbeat():
            while not done.wait(30):
                try:
                    require_owner(lab.read_hold(), round.state)
                    lab.beat(
                        "verification capture policies: bounded image gate on DGX4"
                    )
                except (RuntimeError, OSError, SystemExit):
                    return

        threading.Thread(target=heartbeat, daemon=True).start()
        signal.signal(
            signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        status = 1
        try:
            remote_out = (
                args.gate_checkout / "results/private/verification-shapes" / args.run_id
            )
            command = [
                "systemd-run",
                "--user",
                "--wait",
                "--pipe",
                "--unit",
                unit,
                "-p",
                "MemoryMax=4G",
                "-p",
                "CPUQuota=100%",
                "-p",
                "RuntimeMaxSec=11400" if args.screen_receipt else "RuntimeMaxSec=2700",
                "-p",
                "TimeoutStopSec=30",
                "python3",
                str(
                    args.gate_checkout
                    / "experiments/2026-11-05-verification-shapes/gpu_gate.py"
                ),
                "--apply",
                "--window",
                window,
                "--source",
                str(args.source),
                "--build-receipt",
                str(args.build_receipt),
                "--built-inspection",
                str(args.built_inspection),
                "--out",
                str(remote_out),
            ]
            command += (
                ["--screen-receipt", str(args.screen_receipt)]
                if args.screen_receipt
                else ["--screen-only"]
            )
            round.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=8",
                    "dgx4",
                    shlex.join(command),
                ]
            )
            round.state["completed"] = True
            round.state["remote_receipt"] = str(remote_out / "receipt.json")
            round.save()
            status = 0
        except BaseException as error:  # noqa: BLE001 - retain interruption and restore
            round.state["error"] = f"{type(error).__name__}: {error}"
            round.save()
        finally:
            try:
                round.restore()
            except BaseException as error:  # noqa: BLE001 - preserve hold after failed restoration
                round.state["restore_error"] = f"{type(error).__name__}: {error}"
                round.save()
                status = 1
            done.set()
        return status


if __name__ == "__main__":
    raise SystemExit(main())
