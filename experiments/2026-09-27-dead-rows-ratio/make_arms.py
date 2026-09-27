#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-dead-rows-ratio/make_arms.py   (from the deployment checkout)

The screen runs the r5d image with patch 0011's three runtime files mounted
over its vLLM tree (tree 7b839dc1 with them); stage them first with
overlay.sh. A promotion builds the image instead.
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())
OVERLAY = "{home}/spark3-overlay/r5e"
FILES = (
    "vllm/v1/worker/gpu/input_batch.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py",
)


def arm(env: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = "vllm-ds41f-kkref:04c30fa98e79-r5d"
    cfg["container"]["expected_labels"]["local.spark3.vllm.tree"] = (
        "e94095189f47352b7a936b0a69eeda86cc114e43"
    )
    cfg["container"]["mounts"] += [
        [f"{OVERLAY}/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"] for f in FILES
    ]
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5e-k5"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    # Promoted host budget; inside it the device keeps live the drafts that
    # maximize this step's expected tokens per millisecond.
    "ratio": arm({"SPARK3_DSPARK_DEAD_ROWS": "ratio"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
