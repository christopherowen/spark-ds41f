#!/usr/bin/env python3
"""The step harness (iket_step.py) in this process, then PyTorch's peak memory.

usage: peak.py OUT.json iket_step.py-arguments...
"""
import json
import sys
import traceback

import torch

import iket_step

out = sys.argv[1]
sys.argv = ["iket_step.py", *sys.argv[2:]]
error = None
try:
    iket_step.main()
except BaseException as exc:  # an assertion is a result here: record it, then fail
    error = f"{type(exc).__name__}: {exc}"[:400]
    traceback.print_exc()
gib = 2**30
stats = {"summary": "peak memory", "argv": sys.argv[1:], "error": error,
         "max_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
         "max_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3)}
open(out, "w").write(json.dumps(stats) + "\n")
print(json.dumps(stats), flush=True)
sys.exit(1 if error else 0)
