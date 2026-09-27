#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-draft-precision/make_arms.py   (from the deployment checkout)
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
    # One compile cache per arm: the switches change the drafter's modules.
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/vllm-r5e-k5-{name}"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    "control": arm("control", {}),
    # Drafter-owned NVFP4 vocabulary head (draft logits only).
    "dhead": arm("dhead", {"VLLM_DS41_DRAFT_NVFP4_HEAD": "1"}),
    # NVFP4 Markov projection (markov_w2), read once per draft position.
    "markov": arm("markov", {"VLLM_DS41_MARKOV_NVFP4": "1"}),
    "both": arm("both", {"VLLM_DS41_DRAFT_NVFP4_HEAD": "1", "VLLM_DS41_MARKOV_NVFP4": "1"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
