#!/usr/bin/env python3
"""Bounded component gates, two fresh processes; no serving claims."""
import json
import os
from pathlib import Path
import subprocess
import sys


out = Path(os.environ.get("BLOCK_METADATA_OUTPUT", "/c/block-metadata"))
out.mkdir(parents=True, exist_ok=True)
steps = []
for boot in (1, 2):
    for label, args, timeout in [
        ("host-publication", ["/b/test_host.py"], 30),
        ("GPU-transition-gates", ["-m", "pytest", "-q", "-rfEs", "--noconftest",
                                  "-p", "no:cacheprovider", "/b/test_gpu.py"], 300),
        ("component-benchmark", ["/b/bench.py", "--output", str(out / f"boot-{boot}.json")], 600),
    ]:
        print(f"=== process {boot}: {label}", flush=True)
        try:
            code = subprocess.run([sys.executable, *args], timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        steps.append({"process": boot, "step": label, "exit": code})
        (out / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
        print(json.dumps(steps[-1]), flush=True)
        if code != 0:
            sys.exit(1)
print("Component gates completed; native serving validation remains required", flush=True)
