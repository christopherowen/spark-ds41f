#!/usr/bin/env python3
"""`sonar determinism` with the chat template's reasoning switch: the determinism gate.

usage: scripts/sonar_determinism.py URL [--tokens T] [--long-lines L] [--wide] [--no-logprobs]

Lab extras call ``python3 SCRIPT URL ARGS...``; this runs the pinned Sonar
(spark_ds41f/sonar.py; the pin is config/sonar.json).
Its requests and digests are those of experiments/2026-10-12-r6g/token_determinism_text.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark_ds41f import sonar  # noqa: E402

if __name__ == "__main__":
    sys.exit(sonar.exec_command('determinism', sys.argv[1:], reasoning_off=True))
