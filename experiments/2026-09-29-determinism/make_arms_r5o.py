#!/usr/bin/env python3
"""Write the r5o-based batch-trace arm (run from the repository root, config = r5o).

detm-r5o-trace: r5o with the experimental deterministic MoE (det-masked
overlay: 0004-0006 and 0008, split-K through the FP32 reducer), the r5o
screen's pinned adaptive-verification cost table, and the batch-trace debug
overlay: per-row checksums of every MoE layer and the shared expert (up to 256
rows) and a per-step log of the scheduled batch (row offsets after draft
reallocation, dead rows, request ids), all behind SPARK3_MOE_CHECKSUM_DIR.
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-determinism")
PIN = "/cache/kkref/dspark-costs/r5o-pin-20260930"
B12X = "/opt/spark3/candidate/b12x/b12x"
VLLM = "/opt/spark3/candidate/vllm/vllm"
DET = ("moe/fused_moe/_impl.py", "moe/fused_moe/_preparation.py", "moe/fused_moe/_tuning.py",
       "moe/_shared/kernels/dynamic.py", "moe/_shared/kernels/silu.py", "moe/_shared/kernels/w4a16/kernel.py")
TRACE = {
    "checksum_debug.py": "model_executor/layers/fused_moe/runner/checksum_debug.py",
    "moe_runner.py": "model_executor/layers/fused_moe/runner/moe_runner.py",
    "model.py": "models/deepseek_v4/nvidia/model.py",
    "model_runner.py": "v1/worker/gpu/model_runner.py",
}
base = json.loads(Path("config/cluster.json").read_text())
assert base["promoted_baseline"] == "2026-09-30-karmic-kraken-r5o", base["promoted_baseline"]
arm = copy.deepcopy(base)
for f in DET:
    arm["container"]["mounts"].append([f"{{home}}/spark3-overlay/det-masked/b12x/{f}", f"{B12X}/{f}", "ro"])
for name, target in TRACE.items():
    arm["container"]["mounts"].append([f"{{home}}/spark3-overlay/batch-trace/{name}", f"{VLLM}/{target}", "ro"])
arm["environment"].update(
    B12X_DYNAMIC_DETERMINISTIC_OUTPUT="1", B12X_DENSE_SPLITK_TURBO="0",
    SPARK3_DSPARK_COST_DIR=PIN, SPARK3_MOE_CHECKSUM_DIR="/cache/kkref/moe-checksums",
    SPARK3_MOE_CHECKSUM_ROWS="256", SPARK3_MOE_CHECKSUM_RUNNER_CAPACITY="8192",
    SPARK3_MOE_CHECKSUM_TAGS_CAPACITY="16384",
)
path = E / "cluster-detm-r5o-trace.json"
path.write_text(json.dumps(arm, indent=2) + "\n")
print("wrote", path)
