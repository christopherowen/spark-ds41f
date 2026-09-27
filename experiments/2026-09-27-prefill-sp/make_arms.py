#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-prefill-sp/make_arms.py   (from the deployment checkout)

The sp arm runs the promoted image with the prefill sequence-parallelism
files mounted over its vLLM tree (staged by overlay.sh).
"""
import copy
import json
import subprocess
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())
OVERLAY = "{home}/spark3-overlay/r5f-sp"
FILES = subprocess.check_output(
    ["git", "-C", str(Path.home() / "projects/spark3-vllm-ds41f/.work/upstreams/vllm"),
     "diff", "--name-only", "spark3/r5f-check", "spark3/r5f-sp", "--", "vllm"],
    text=True,
).split()


def arm(name: str, env: dict, overlay: bool) -> dict:
    cfg = copy.deepcopy(base)
    if overlay:
        cfg["container"]["mounts"] += [
            [f"{OVERLAY}/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"] for f in FILES
        ]
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = f"/cache/kkref/jit/vllm-r5f-sp-{name}"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    "control": arm("control", {}, overlay=False),
    # Prefill forwards of 2048 tokens or more run sequence-parallel.
    "sp": arm("sp", {"SPARK3_DS41_PREFILL_SP_MIN_ROWS": "2048"}, overlay=True),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
