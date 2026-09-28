#!/usr/bin/env python3
"""Sample the dgx1 container's Python processes with repeated py-spy dumps.

usage: pyspy_sampler.py OUT   (stop it with SIGTERM once the boot is done)

py-spy's `record` hangs on these workers; one-shot `dump --nonblocking`
works. Each sample is headed "=== <utc time> <pid> <process title>".
"""
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PY_SPY = str(Path.home() / ".local/bin/py-spy")
out = open(sys.argv[1], "a")
running = True
signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("running", False))


def processes() -> list[tuple[str, str]]:
    top = subprocess.run(["docker", "top", "dsv41-karmic-kraken", "-eo", "pid,args"],
                         capture_output=True, text=True)
    found = []
    for line in top.stdout.splitlines()[1:]:
        pid, _, args = line.strip().partition(" ")
        # Every process: vLLM renames its workers (VLLM::Worker_TP0), and py-spy
        # reports the live name in each dump; skip only the resource tracker.
        if "resource_tracker" not in args:
            found.append((pid, args.strip()[:60]))
    return found


deadline = time.time() + 900
while running and time.time() < deadline:
    procs = processes()
    if not procs:
        time.sleep(0.2)
        continue
    for pid, title in procs:
        dump = subprocess.run(["sudo", "-n", "timeout", "5", PY_SPY, "dump", "--nonblocking", "--pid", pid],
                              capture_output=True, text=True)
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        out.write(f"=== {stamp} {pid} {title}\n{dump.stdout}\n")
    out.flush()
out.close()
