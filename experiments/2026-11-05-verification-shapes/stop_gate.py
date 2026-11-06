"""Stop one named qualification unit and its container before releasing a window."""

import argparse
import re
import subprocess


def stop(unit, container):
    if (
        not re.fullmatch(r"verification-shapes-gate-dgx1-[0-9]+", unit)
        or container != unit
    ):
        raise RuntimeError("gate names do not identify this experiment/window")
    subprocess.run(["systemctl", "--user", "stop", unit], check=False)
    active = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if active not in {"inactive", "failed", "unknown", ""}:
        raise RuntimeError("remote gate unit remains active: " + active)
    subprocess.run(
        ["docker", "stop", "--time", "15", container],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    running = subprocess.check_output(
        ["docker", "ps", "-q", "--filter", "name=^" + container + "$"], text=True
    )
    if running.strip():
        raise RuntimeError("remote gate container remains running")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unit")
    parser.add_argument("container")
    args = parser.parse_args()
    stop(args.unit, args.container)
