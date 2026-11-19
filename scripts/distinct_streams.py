#!/usr/bin/env python3
"""N concurrent streams with N distinct prompts (`sonar streams`, with the reasoning switch).

usage: scripts/distinct_streams.py URL [--streams N] [--samples S] [--tokens T]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
The name is kept for the lab profiles and recorded specs that call it; requests and output
are those of the script it replaced (and of experiments/2026-09-29-determinism/c8_distinct.py
at --streams 8). Exits 3 when someone else's requests finished during a sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('streams', sys.argv[1:], reasoning_off=True))
