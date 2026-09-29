#!/usr/bin/env python3
"""Write the r5m candidate arm from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-r5m")
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5m"
TREES = {
    "local.spark3.vllm.tree": "c108cd6d1fe8e2d3b91c065feefe818742159020",
    "local.spark3.b12x.tree": "35299956ce4954b71a7bc3b0529e3c6417fa975f",
}
CACHE = "/cache/kkref/jit/vllm-r5m"

arm = copy.deepcopy(json.loads(Path("config/cluster.json").read_text()))
arm["container"]["image"] = TAG
arm["container"]["expected_labels"].update(TREES)
arm["environment"].update(VLLM_CACHE_DIR=CACHE, VLLM_CACHE_ROOT=CACHE)
path = E / "cluster-candidate.json"
path.write_text(json.dumps(arm, indent=2) + "\n")
print("wrote", path)
