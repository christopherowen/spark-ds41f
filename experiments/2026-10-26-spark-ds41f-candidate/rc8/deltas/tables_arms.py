#!/usr/bin/env python3
"""Candidate 8 against spark-ds41f-1025 (the arm labelled "release"): the softmax fix's
serving table, experiments/2026-10-25-softmax-fix/tables_arms.py, run as it is.

usage: tables_arms.py RUN LABEL [LABEL...]
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "2026-10-25-softmax-fix/tables_arms.py"),
               run_name="__main__")
