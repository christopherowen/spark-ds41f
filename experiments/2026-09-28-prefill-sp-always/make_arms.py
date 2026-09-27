#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-28-prefill-sp-always/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(env: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["environment"].update(env)
    cfg["environment"] = dict(sorted(cfg["environment"].items()))
    return cfg


arms = {
    # Promoted: sequence-parallel prefill from 2048 tokens.
    "current": arm({}),
    # Every forward above the decode graph sizes (the code raises 1 to 49).
    "always": arm({"SPARK3_DS41_PREFILL_SP_MIN_ROWS": "1"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
