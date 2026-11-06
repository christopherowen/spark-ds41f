"""Restore the captured service and checkout for one exclusively owned window."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HOLDER = "http-cancellation-hardware"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def ownership(state, restoring=False):
    hold = json.loads(Path(state["hold_file"]).read_text())
    if hold.get("holder") != HOLDER or hold.get("window") != state["window"]:
        raise RuntimeError("foreign or missing window; no serving operation authorized")
    if hold.get("management_loss_latched"):
        raise RuntimeError("management loss is latched; no automatic launch")
    if hold.get("restoring") and not restoring:
        raise RuntimeError("restoration already started")
    return hold


def beat(state):
    path = Path(state["hold_file"])
    with path.with_suffix(".http.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hold = ownership(state)
        hold["heartbeat"] = stamp()
        save(path, hold)


def restore(state_path):
    with (state_path.parent / "restore.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state.get("restored"):
            return 0
        ownership(state, restoring=True)
        root = Path(state["checkout"])
        hold_path = Path(state["hold_file"])
        records = state.setdefault("restore_steps", [])

        def run(argv, label, timeout=2400):
            ownership(state, restoring=True)
            record = {"argv": argv, "started": stamp(), "log": label + ".log"}
            records.append(record)
            save(state_path, state)
            with (state_path.parent / record["log"]).open("a") as log:
                result = subprocess.run(
                    argv,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            record.update(exit_code=result.returncode, ended=stamp())
            save(state_path, state)
            if result.returncode:
                raise RuntimeError(f"{label} failed; reservation retained")

        def cli(profile, action, *extra):
            argv = [
                sys.executable,
                "bin/spark",
                "--cluster-config",
                profile,
                "cluster",
                action,
                *extra,
            ]
            plan = subprocess.run(
                argv, cwd=root, text=True, capture_output=True, timeout=180
            )
            save(
                state_path.parent / f"restore-{action}-plan.json",
                {
                    "argv": argv,
                    "exit_code": plan.returncode,
                    "stdout": plan.stdout,
                    "stderr": plan.stderr,
                },
            )
            print(plan.stdout, flush=True)
            if (
                plan.returncode
                or "--apply would refuse" in plan.stdout
                or any(
                    line.strip().startswith("shape:")
                    and line.split(":", 1)[1].strip().startswith("CHANGED")
                    for line in plan.stdout.splitlines()
                )
            ):
                raise RuntimeError("restoration plan refused; no shape bypass")
            run([*argv, "--apply"], "restore-" + action)

        try:
            for node in state["nodes"]:
                reply = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=8",
                        node["target"],
                        "true",
                    ],
                    capture_output=True,
                    timeout=15,
                )
                if reply.returncode:
                    hold = ownership(state, restoring=True)
                    hold["management_loss_latched"] = True
                    save(hold_path, hold)
                    raise RuntimeError(
                        f"management unavailable on {node['name']}; no restart"
                    )
            with hold_path.with_suffix(".http.lock").open("a") as edit:
                fcntl.flock(edit, fcntl.LOCK_EX)
                hold = ownership(state, restoring=True)
                hold["restoring"] = True
                save(hold_path, hold)
            if not state.get("stopped"):
                cli(state["active_profile"], "stop", "--parallel")
                state["stopped"] = True
                save(state_path, state)
            cli(state["restore_profile"], "sync", "--revision", state["restore_commit"])
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            if actual != state["restore_commit"]:
                raise RuntimeError("head checkout was not restored exactly")
            cli(state["restore_profile"], "start", "--replace")
            run(
                [
                    sys.executable,
                    "bin/spark",
                    "--cluster-config",
                    state["restore_profile"],
                    "doctor",
                    "--live",
                ],
                "restore-doctor",
                timeout=300,
            )
            for node in state["nodes"]:
                command = shlex.join(
                    ["docker", "inspect", state["container"], "--format", "{{.Image}}"]
                )
                actual = subprocess.check_output(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=8",
                        node["target"],
                        command,
                    ],
                    text=True,
                    timeout=15,
                ).strip()
                if actual != state["restore_images"][node["name"]]:
                    raise RuntimeError(f"{node['name']} restore image changed")
            state.update(restored=True, ended=stamp())
            save(state_path, state)
            ownership(state, restoring=True)
            hold_path.unlink()
            return 0
        except BaseException as error:
            state.update(restore_error=repr(error), ended=stamp())
            save(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--beat", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    if not args.apply:
        print(
            json.dumps(
                {
                    k: state[k]
                    for k in ("window", "restore_profile", "restore_commit", "nodes")
                },
                indent=2,
            )
        )
        return 0
    if args.beat:
        beat(state)
        return 0
    if args.watch:
        while True:
            state = json.loads(args.state.read_text())
            if state.get("restored") or state.get("restore_error"):
                return 0
            try:
                hold = ownership(state, restoring=True)
            except (FileNotFoundError, RuntimeError):
                return 0
            then = datetime.fromisoformat(hold["heartbeat"].replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - then).total_seconds() > 900:
                return restore(args.state)
            time.sleep(30)
    return restore(args.state)


if __name__ == "__main__":
    raise SystemExit(main())
