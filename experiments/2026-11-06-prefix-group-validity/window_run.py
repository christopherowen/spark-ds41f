#!/usr/bin/env python3
"""Supervise an existing benchmark command under the same owned-window lease."""
import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from window_lease import require_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-state', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required')
    require_window(args.window_state)
    child = subprocess.Popen(command, start_new_session=True)
    try:
        while child.poll() is None:
            require_window(args.window_state)
            time.sleep(1)
        require_window(args.window_state)
        return child.returncode
    finally:
        if child.poll() is None:
            for sig, timeout in [(signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
                if child.poll() is not None:
                    break
                os.killpg(child.pid, sig)
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue


if __name__ == '__main__':
    raise SystemExit(main())
