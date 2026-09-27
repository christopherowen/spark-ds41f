#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-prefill-moe-tile/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(name: str, env: dict) -> dict:
    cfg = copy.deepcopy(base)
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/vllm-r5e-k5-{name}"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    "control": arm("control", {}),
    # Every dynamic W4A8 MoE launch on the M32 tile (the fused persistent
    # kernel), prefill and decode alike.
    "tile32": arm("tile32", {"B12X_DYNAMIC_TILE_MN": "32x128"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
