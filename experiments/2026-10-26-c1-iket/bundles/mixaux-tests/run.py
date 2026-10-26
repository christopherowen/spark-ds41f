#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. ``{python}`` is this interpreter and ``{out}`` the output root."""
import json
import os
import subprocess
import sys
import time

BUNDLE = os.environ["IKET_BUNDLE"]
OUT = f"/c/iket/{BUNDLE}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
os.makedirs(OUT)
print(f"=== output {OUT}", flush=True)
codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    argv = [a.replace("{python}", sys.executable).replace("{out}", OUT) for a in argv]
    print(f"=== {label}", flush=True)
    started = time.monotonic()
    try:
        code = subprocess.run(argv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f} s", flush=True)
    codes.append((label, code))
print(json.dumps({"output": OUT, "steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
