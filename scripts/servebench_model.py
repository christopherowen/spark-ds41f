#!/usr/bin/env python3
"""`servebench model` for lab extras, which call `python3 SCRIPT URL ARGS...`.

usage: scripts/servebench_model.py URL [servebench model options]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["model", *sys.argv[1:]]))
