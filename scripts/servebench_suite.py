#!/usr/bin/env python3
"""`servebench suite` for lab measure jobs, which call `python3 SCRIPT URL ARGS...`.

usage: scripts/servebench_suite.py URL --suite NAME --out DIR [--corpus FILE] [--label L]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["suite", *sys.argv[1:]]))
