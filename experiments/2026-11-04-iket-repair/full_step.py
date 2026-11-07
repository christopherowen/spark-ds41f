#!/usr/bin/env python3
"""Loading-only diagnostic around the unchanged published full-step harness."""
import argparse
import os
from pathlib import Path
import runpy
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["full"])
parser.add_argument("--out", required=True)
parser.add_argument("--wait-file")
parser.add_argument("--no-finalize", action="store_true")
args, rest = parser.parse_known_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.wait_file:
    import torch
    torch.cuda.init()
    (out / "ready").write_text("ready\n")
    deadline = time.monotonic() + 120
    while not Path(args.wait_file).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("debugger did not release workload")
        time.sleep(.05)
sys.path[:0] = ["/h"]
sys.argv = ["/h/iket_step.py", "off", "--out", str(out / "model"), *rest]
runpy.run_path("/h/iket_step.py", run_name="__main__")
if args.no_finalize:
    # Both sides of this diagnostic exclude profiler exit/export identically.
    # Success here establishes startup behavior, not trace-export validity.
    os._exit(0)
