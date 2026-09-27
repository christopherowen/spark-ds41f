#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-parallel-prefills/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())

arms = {"control": copy.deepcopy(base), "mpp8": copy.deepcopy(base)}
args = arms["mpp8"]["serve_args"]
# Admit up to eight prefills per step; the 4,096-token step budget still
# bounds each step's work.
args[args.index("--max-parallel-prefills") + 1] = "8"
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
