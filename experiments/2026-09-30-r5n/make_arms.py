#!/usr/bin/env python3
"""Write the r5n arms from config/cluster.json (run from the repository root).

cluster-overlay.json: r5m with the fenced dense GEMM mounted over the image
(the file 0004 ships, from experiments/2026-09-29-determinism/overlay_fence.sh).
cluster-candidate.json: the r5n image built from the lock.
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-30-r5n")
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5n"
TREES = {
    "local.spark3.vllm.tree": "c108cd6d1fe8e2d3b91c065feefe818742159020",
    "local.spark3.b12x.tree": "693aaed550673dcd86655ad4b2c3589882f0bd29",
}
CACHE = "/cache/kkref/jit/vllm-r5n"
base = json.loads(Path("config/cluster.json").read_text())

overlay = copy.deepcopy(base)
overlay["container"]["mounts"].append(
    ["{home}/spark3-overlay/gemm-fence/b12x/_lib/dense_gemm.py",
     "/opt/spark3/candidate/b12x/b12x/_lib/dense_gemm.py", "ro"]
)
candidate = copy.deepcopy(base)
candidate["container"]["image"] = TAG
candidate["container"]["expected_labels"].update(TREES)
candidate["environment"].update(VLLM_CACHE_DIR=CACHE, VLLM_CACHE_ROOT=CACHE)
for name, arm in (("overlay", overlay), ("candidate", candidate)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(arm, indent=2) + "\n")
    print("wrote", path)
