#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-27-dead-rows/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(env: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["container"]["image"] = "vllm-ds41f-kkref:04c30fa98e79-r5d"
    cfg["container"]["expected_labels"]["local.spark3.vllm.tree"] = (
        "e94095189f47352b7a936b0a69eeda86cc114e43"
    )
    environment = cfg["environment"]
    environment["VLLM_CACHE_DIR"] = environment["VLLM_CACHE_ROOT"] = "/cache/kkref/jit/vllm-r5d-k5"
    environment.update(env)
    cfg["environment"] = dict(sorted(environment.items()))
    return cfg


arms = {
    # Promoted behaviour on the same image: dead rows off.
    "control": arm({}),
    # Every draft scheduled; rows past the survival cut are dead.
    "dead03": arm({"SPARK3_DSPARK_VERIFY_RULE": "all", "SPARK3_DSPARK_DEAD_ROWS_TAU": "0.3"}),
    "dead05": arm({"SPARK3_DSPARK_VERIFY_RULE": "all", "SPARK3_DSPARK_DEAD_ROWS_TAU": "0.5"}),
    # Host budget as promoted (caps the row count at high batch, where dead
    # rows still pay attention, logits and sampling), with the on-device cut
    # applied inside it.
    "budget01": arm({"SPARK3_DSPARK_DEAD_ROWS_TAU": "0.1"}),
    "budget02": arm({"SPARK3_DSPARK_DEAD_ROWS_TAU": "0.2"}),
    "dead01": arm({"SPARK3_DSPARK_VERIFY_RULE": "all", "SPARK3_DSPARK_DEAD_ROWS_TAU": "0.1"}),
}
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
