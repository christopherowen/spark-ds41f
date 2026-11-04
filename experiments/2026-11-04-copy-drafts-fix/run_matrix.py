#!/usr/bin/env python3
"""Reuse the isolated sp8 runner, mounting this clean repository read-only."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "experiments/2026-11-04-iket-streaming/run_matrix.py"
spec = importlib.util.spec_from_file_location("isolated_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.HERE = ROOT
runner.main()
