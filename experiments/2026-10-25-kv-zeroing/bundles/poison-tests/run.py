#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it. A step is python's
arguments, or a command when its first word is a path ("{python}" names python); its
options set environment variables and expect an exit code (a positive control)."""
import json
import os
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout, *rest in STEPS:
    options = rest[0] if rest else {}
    command = [sys.executable if a == "{python}" else a for a in argv]
    if not command[0].startswith("/"):
        command = [sys.executable, *command]
    print(f"=== {label}", flush=True)
    try:
        code = subprocess.run(command, timeout=timeout,
                              env={**os.environ, **options.get("env", {})}).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    except FileNotFoundError:
        code = f"missing {command[0]}"
    passed = code == options.get("expect_exit", 0)
    print(f"=== {label}: exit {code}" + ("" if passed else " (FAILED)"), flush=True)
    codes.append((label, code, passed))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(passed for _, _, passed in codes) else 1)
