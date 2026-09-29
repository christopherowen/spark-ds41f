#!/usr/bin/env python3
"""Write the r5l candidate arms from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-r5l")
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5l"
TREES = {
    "local.spark3.vllm.tree": "d1886d36ab243dc6c886b4a9306b7f453e2174b9",
    "local.spark3.b12x.tree": "640c8544b3242e858c962ad6de98febd8f74e8e3",
}
CACHE = "/cache/kkref/jit/vllm-r5l"

base = json.loads(Path("config/cluster.json").read_text())
arm = copy.deepcopy(base)
arm["container"]["image"] = TAG
arm["container"]["expected_labels"].update(TREES)
arm["environment"].update(
    VLLM_CACHE_DIR=CACHE,
    VLLM_CACHE_ROOT=CACHE,
    # Patch 0026, opt-in; review by 2026-10-29 (TODO.md).
    SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS="300",
)
# Validation boot: every rank also runs the full-row indexer and compares its
# split rows (patch 0025), and the carve-out check runs every minute.
check = copy.deepcopy(arm)
check["environment"].update(
    SPARK3_DS41_INDEXER_SPLIT_CHECK="1", SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS="60"
)

for name, config in (("candidate", arm), ("candidate-check", check)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
