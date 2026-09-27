#!/usr/bin/env python3
"""Derive this experiment's profiling config from config/cluster.json.

usage: experiments/2026-09-27-single-stream-profile/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
cfg = copy.deepcopy(json.loads((ROOT / "config/cluster.json").read_text()))
cfg["serve_args"] += [
    "--profiler-config",
    json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": "/cache/kkref/profiles/r5e",
        "torch_profiler_with_stack": False,
        "ignore_frontend": True,
        "max_iterations": 24,
        "torch_profiler_use_gzip": True,
    }),
]
(E / "cluster-profile.json").write_text(json.dumps(cfg, indent=2) + "\n")
