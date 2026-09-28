#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-28-boot-time/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())

# The driver's PTX JIT cache (~195 MB) and TileLang's cache live in the
# container's home directory, outside every mount, so each boot rebuilds them.
PERSIST = {
    "CUDA_CACHE_PATH": "/cache/kkref/jit/nv-compute",
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "TILELANG_CACHE_DIR": "/cache/kkref/jit/tilelang",
}
NCCL_TIMINGS = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}


def arm(env: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["environment"].update(env)
    cfg["environment"] = dict(sorted(cfg["environment"].items()))
    return cfg


arms = {
    "base": arm({}),
    "persist": arm(PERSIST),
    # Diagnostic: NCCL prints per-phase communicator init timings.
    "nccl-info": arm({**PERSIST, **NCCL_TIMINGS}),
    # NCCL's GPU-initiated networking setup, off (NCCL_GIN_ENABLE=0).
    "nogin": arm({**PERSIST, **NCCL_TIMINGS, "NCCL_GIN_ENABLE": "0"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
