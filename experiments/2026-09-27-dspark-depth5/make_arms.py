#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-dspark-depth5/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
TAG = "vllm-ds41f-kkref:01f1b874c774-r4b"
VLLM_TREE = "ee3a0fd4ab59883e4a9e15dfb101b42b09e47d00"
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(env: dict, cache: str, drafts: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = TAG
    cfg["container"]["expected_labels"]["local.spark3.vllm.tree"] = VLLM_TREE
    args = cfg["serve_args"]
    # Thinking is already the default when a request names neither key; the
    # explicit default overrode a client's enable_thinking=false.
    i = args.index("--default-chat-template-kwargs")
    del args[i:i + 2]
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/{cache}"
    environment["SPARK3_DSPARK_PROFILE_REPLAYS"] = "15"
    # Profile on distinct real rows (patch 0009): verification rows are then
    # priced with their routed-expert work, so the cost scale is 1.
    environment["SPARK3_DSPARK_PROFILE_TOKENS"] = "random"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    i = args.index("--speculative-config")
    speculative = json.loads(args[i + 1])
    speculative.update(num_speculative_tokens=drafts, adaptive_verification_cost_scale=1.0)
    args[i + 1] = json.dumps(speculative, separators=(",", ":"))
    if drafts > 3:
        # Graphs for eight streams at 1 + 5 rows each.
        args[args.index("--max-cudagraph-capture-size") + 1] = "48"
        i = args.index("--compilation-config")
        compilation = json.loads(args[i + 1])
        compilation["cudagraph_capture_sizes"] += [40, 48]
        args[i + 1] = json.dumps(compilation, separators=(",", ":"))
    return cfg


arms = {
    "k3real": arm({"SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/r4b-k3-real"}, "vllm-r4a", 3),
    "k5real": arm({"SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/r4b-k5-real"}, "vllm-r4a-k5", 5),
    # With verification priced on real rows, the per-step ratio rule trims
    # drafts that beat the long-run rate; the marginal rule keeps them.
    "k3realm": arm({"SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/r4b-k3-real",
                    "SPARK3_DSPARK_VERIFY_RULE": "marginal"}, "vllm-r4a", 3),
    "k5realm": arm({"SPARK3_DSPARK_COST_DIR": "/cache/kkref/dspark-costs/r4b-k5-real",
                    "SPARK3_DSPARK_VERIFY_RULE": "marginal"}, "vllm-r4a-k5", 5),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
