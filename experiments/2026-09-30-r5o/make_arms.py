#!/usr/bin/env python3
"""Write the r5o arms from config/cluster.json (r5n; run from the repository root).

r5n-pin, overlay-pin: r5n without and with the three files B12X 0005 changes
mounted over the image (overlay.sh), sharing one pinned adaptive-verification
cost table (vLLM patch 0005), so verification policy is held fixed.
detm, detm-fence: the experimental deterministic MoE (determinism experiment's
det-masked overlay) without and with those files, for the long-prompt
repeatability probe. candidate: the r5o image built from the lock.
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-30-r5o")
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5o"
TREE = "1a8b9401584ada0372939df49e658c3dbeae7658"
CACHE = "/cache/kkref/jit/vllm-r5o"
PIN = "/cache/kkref/dspark-costs/r5o-pin-20260930"
B12X = "/opt/spark3/candidate/b12x/b12x"
FENCE = ("attention/_shared/contiguous/forward.py", "gemm/bf16_gemv/_prefill.py", "norm/mhc/_kernels.py")
DET = ("moe/fused_moe/_impl.py", "moe/fused_moe/_preparation.py", "moe/fused_moe/_tuning.py",
       "moe/_shared/kernels/dynamic.py", "moe/_shared/kernels/silu.py", "moe/_shared/kernels/w4a16/kernel.py")
base = json.loads(Path("config/cluster.json").read_text())
assert base["promoted_baseline"] == "2026-09-30-karmic-kraken-r5n", base["promoted_baseline"]


def mount(config, overlay, files):
    for f in files:
        config["container"]["mounts"].append([f"{{home}}/spark3-overlay/{overlay}/b12x/{f}", f"{B12X}/{f}", "ro"])


r5n_pin = copy.deepcopy(base)
r5n_pin["environment"]["SPARK3_DSPARK_COST_DIR"] = PIN
overlay_pin = copy.deepcopy(r5n_pin)
mount(overlay_pin, "r5o-fence", FENCE)
detm = copy.deepcopy(base)
mount(detm, "det-masked", DET)
detm["environment"].update(B12X_DYNAMIC_DETERMINISTIC_OUTPUT="1", B12X_DENSE_SPLITK_TURBO="0")
detm_fence = copy.deepcopy(detm)
mount(detm_fence, "r5o-fence", FENCE)
candidate = copy.deepcopy(base)
candidate["container"]["image"] = TAG
candidate["container"]["expected_labels"]["local.spark3.b12x.tree"] = TREE
candidate["environment"].update(VLLM_CACHE_DIR=CACHE, VLLM_CACHE_ROOT=CACHE)
for name, arm in (("r5n-pin", r5n_pin), ("overlay-pin", overlay_pin), ("detm", detm),
                  ("detm-fence", detm_fence), ("candidate", candidate)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(arm, indent=2) + "\n")
    print("wrote", path)
