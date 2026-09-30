#!/usr/bin/env python3
"""Write the GEMV lookup arms (run from the repository root, config = r5o).

Every arm pins the r5o screen's adaptive-verification cost table, so the
verification policy is the same in all of them.

detm-r5o-lookup-trace: detm-r5o-trace's deterministic MoE (det-masked
overlay), plus the gemv-lookup overlay (the two files of
vllm-0027-gemv-smallest-capacity.patch) and the attn-trace debug overlay
(batch-trace plus attention checksums and the first-call router gate capture
on rank 0), all behind SPARK3_MOE_CHECKSUM_DIR / SPARK3_GATE_CAPTURE_ROWS.
r5o-pin, r5o-lookup-pin: production r5o without and with the lookup overlay.
detm-r5o-pin, detm-r5o-lookup-pin: the deterministic MoE without and with it.
The four performance arms carry no debug overlay or logging.
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
LOOKUP = ("models/deepseek_v4_1/b12x_layers.py", "models/deepseek_v4_1/compressor.py")
TRACE = {
    "checksum_debug.py": "model_executor/layers/fused_moe/runner/checksum_debug.py",
    "moe_runner.py": "model_executor/layers/fused_moe/runner/moe_runner.py",
    "model.py": "models/deepseek_v4/nvidia/model.py",
    "model_runner.py": "v1/worker/gpu/model_runner.py",
    "attention.py": "models/deepseek_v4_1/attention.py",
}
base = json.loads(Path("config/cluster.json").read_text())
assert base["promoted_baseline"] == "2026-09-30-karmic-kraken-r5o", base["promoted_baseline"]


def mount(arm, source, target):
    arm["container"]["mounts"].append([f"{{home}}/spark3-overlay/{source}", target, "ro"])


def with_lookup(arm):
    arm = copy.deepcopy(arm)
    for f in LOOKUP:
        mount(arm, f"gemv-lookup/vllm/{f}", f"{VLLM}/{f}")
    return arm


r5o_pin = copy.deepcopy(base)
r5o_pin["environment"]["SPARK3_DSPARK_COST_DIR"] = PIN
detm_pin = copy.deepcopy(r5o_pin)
for f in DET:
    mount(detm_pin, f"det-masked/b12x/{f}", f"{B12X}/{f}")
detm_pin["environment"].update(B12X_DYNAMIC_DETERMINISTIC_OUTPUT="1", B12X_DENSE_SPLITK_TURBO="0")
trace = with_lookup(detm_pin)
for name, target in TRACE.items():
    mount(trace, f"attn-trace/{name}", f"{VLLM}/{target}")
trace["environment"].update(
    SPARK3_MOE_CHECKSUM_DIR="/cache/kkref/moe-checksums", SPARK3_MOE_CHECKSUM_ROWS="128",
    SPARK3_MOE_CHECKSUM_RUNNER_CAPACITY="8192", SPARK3_MOE_CHECKSUM_TAGS_CAPACITY="16384",
    SPARK3_MOE_CHECKSUM_ATTN_CAPACITY="98304", SPARK3_GATE_CAPTURE_ROWS="64",
)
for name, arm in (("detm-r5o-lookup-trace", trace), ("r5o-pin", r5o_pin),
                  ("r5o-lookup-pin", with_lookup(r5o_pin)), ("detm-r5o-pin", detm_pin),
                  ("detm-r5o-lookup-pin", with_lookup(detm_pin))):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(arm, indent=2) + "\n")
    print("wrote", path)
