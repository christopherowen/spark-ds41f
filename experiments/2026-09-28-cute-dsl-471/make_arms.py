#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-28-cute-dsl-471/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())


def candidate(env: dict) -> dict:
    """The r5i image: r5h sources with CuTe DSL 4.7.1 and B12X patch 0002."""
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = "vllm-ds41f-kkref:04c30fa98e79-r5i"
    cfg["container"]["expected_labels"]["local.spark3.b12x.tree"] = (
        "ec4cced90ce8323adae2856a6b46935f41980847"
    )
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5i"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    "control": copy.deepcopy(base),
    # CuTe DSL 4.7.1: quack-kernels imports again, so the DS4.1 L2 weight
    # prefetch (on by default on SM121) compiles and runs.
    "candidate": candidate({}),
    # The same image with the prefetch off: separates the upgrade from it.
    "candidate-nol2": candidate({"VLLM_DS41_L2_PREFETCH": "0"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
