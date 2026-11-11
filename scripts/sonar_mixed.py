#!/usr/bin/env python3
"""`sonar mixed` with the reasoning switch: latency under mixed traffic.

usage: scripts/sonar_mixed.py URL [--rounds R]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
The long prompts are experiments/2026-09-29-determinism's Python files unless --long-source names
others, so its requests are those of mixed_latency.py there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('mixed', sonar.mixed_arguments(sys.argv[1:]), reasoning_off=True))
