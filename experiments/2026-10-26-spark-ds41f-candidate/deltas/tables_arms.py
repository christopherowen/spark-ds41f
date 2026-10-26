#!/usr/bin/env python3
"""Candidate 7 against spark-ds41f-1025 (the arm labelled "release"): the softmax fix's
serving table, experiments/2026-10-25-softmax-fix/tables_arms.py, run as it is.

usage: tables_arms.py RUN LABEL [LABEL...]
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[2] / "2026-10-25-softmax-fix/tables_arms.py"),
               run_name="__main__")
