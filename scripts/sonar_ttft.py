#!/usr/bin/env python3
"""`sonar ttft` with the reasoning switch: short prompts' time to first token.

usage: scripts/sonar_ttft.py URL [--reps N]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
Its requests are those of experiments/2026-09-29-determinism/ttft_short.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('ttft', sys.argv[1:], reasoning_off=True))
