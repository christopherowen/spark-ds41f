#!/usr/bin/env python3
"""Speculative acceptance per prompt, one prompt at a time (`sonar acceptance`, with the reasoning switch).

usage: scripts/acceptance_sweep.py URL [--prompts N] [--tokens T]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
The name is kept for the lab profiles and recorded specs that call it; requests and output
are those of the script it replaced.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('acceptance', sys.argv[1:], reasoning_off=True))
