#!/usr/bin/env python3
"""Pytest outcome lines of run_tests.sh outputs: the summary line, the FAILED/ERROR ids, and
the mounted files' sha256 header. Usage: test_counts.py <raw dir> <label>... -> JSON on stdout"""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
out = {}
for label in sys.argv[2:]:
    text = (d / f"{label}.txt").read_text(errors="replace")
    summary = [line.strip("= ").strip() for line in text.splitlines()
               if re.search(r"\b(passed|failed|error|errors|deselected|skipped)\b.* in [0-9.]+s", line)]
    out[label] = {
        "summary": summary[-1] if summary else None,
        "failed": sorted({m for m in re.findall(r"^FAILED (\S+)", text, re.M)}),
        "errors": sorted({m for m in re.findall(r"^ERROR (\S+)", text, re.M) if "::" in m or m.endswith(".py")}),
        "exit": (re.findall(r"^# exit (\d+)", text, re.M) or [None])[-1],
    }
json.dump(out, sys.stdout, indent=1)
