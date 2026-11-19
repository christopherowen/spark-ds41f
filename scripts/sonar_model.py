#!/usr/bin/env python3
"""`sonar model`: greedy generation and teacher-forced scores.

usage: scripts/sonar_model.py URL [sonar model options]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('model', sys.argv[1:], reasoning_off=False))
