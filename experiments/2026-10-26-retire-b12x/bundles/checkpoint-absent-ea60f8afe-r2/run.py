#!/usr/bin/env python3
"""Run steps.json's steps in order (each in its own interpreter, with a timeout); print each
step's exit status; exit 1 if any failed."""
import json
import subprocess
import sys

codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    print(f"=== {label}", flush=True)
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code}", flush=True)
    codes.append((label, code))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
