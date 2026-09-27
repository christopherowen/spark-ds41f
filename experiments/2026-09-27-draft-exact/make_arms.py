#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-draft-exact/make_arms.py   (from the deployment checkout)

The arms run the r5e image with the runtime files of vLLM patches 0012-0013
mounted over its vLLM tree (staged by overlay.sh).
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())
OVERLAY = "{home}/spark3-overlay/r5f"
FILES = (
    "vllm/v1/worker/gpu/spec_decode/dspark/greedy.py",
    "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py",
    "vllm/model_executor/models/qwen3_dspark.py",
    "vllm/models/deepseek_v4_1/nvidia/dspark.py",
)


def arm(name: str, env: dict, overlay: bool = True) -> dict:
    cfg = copy.deepcopy(base)
    if overlay:
        cfg["container"]["mounts"] += [
            [f"{OVERLAY}/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"] for f in FILES
        ]
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/vllm-r5f-{name}"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    # Promoted behaviour: no overlay.
    "control": arm("control", {}, overlay=False),
    # Vocabulary-parallel greedy drafts (patch 0012).
    "vp": arm("vp", {"SPARK3_DSPARK_VOCAB_PARALLEL": "1"}),
    # Plus the column-sharded main_proj (patch 0013).
    "vpmp": arm("vpmp", {"SPARK3_DSPARK_VOCAB_PARALLEL": "1", "SPARK3_DSPARK_MAIN_PROJ_TP": "1"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
