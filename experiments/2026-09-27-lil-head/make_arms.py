#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-lil-head/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5a"
LABELS = {
    "local.spark3.vllm.tree": "2b59dbc21901f9fb70a2a34f964c4a55f563e542",
    "local.spark3.b12x.tree": "f77d175f0605b1cedae57caf8f00d535198318c2",
}
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(env: dict | None = None) -> dict:
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = TAG
    cfg["container"]["expected_labels"] = dict(LABELS)
    args = cfg["serve_args"]
    # Thinking is already the default when a request names neither key; the
    # explicit default overrode a client's enable_thinking=false.
    i = args.index("--default-chat-template-kwargs")
    del args[i:i + 2]
    environment = cfg["environment"]
    # Patch 0002 is not carried: the base's own Engram overlap replaces it.
    environment.pop("SPARK3_ENGRAM_ASYNC", None)
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5"
    environment["SPARK3_DSPARK_PROFILE_REPLAYS"] = "15"
    environment["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/k3-r5"
    environment.update(env or {})
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    "base": arm(),
    # The prefetch windows only hint the cache, so graphs are identical and
    # the pinned curves differ only through timing: give it its own.
    "nol2": arm({
        "VLLM_DS41_L2_PREFETCH": "0",
        "SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/k3-r5-nol2",
    }),
    # Cost profile on distinct tokens (patch 0009), so extra verification rows
    # are priced with realistic expert routing.
    "realprof": arm({
        "SPARK3_DSPARK_PROFILE_TOKENS": "random",
        "SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/k3-r5-realprof",
    }),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
