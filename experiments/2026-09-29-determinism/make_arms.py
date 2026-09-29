#!/usr/bin/env python3
"""Write the determinism arms from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-determinism")
OVERLAY = "{home}/spark3-overlay/det-planning/b12x/moe/fused_moe"
DECODE_OVERLAY = "{home}/spark3-overlay/det-decode/b12x/moe/fused_moe"
SLICES_OVERLAY = "{home}/spark3-overlay/det-slices/b12x"
SLICES_TARGET = "/opt/spark3/candidate/b12x/b12x"
SLICES_FILES = (
    "moe/fused_moe/_impl.py",
    "moe/fused_moe/_preparation.py",
    "moe/fused_moe/_tuning.py",
    "moe/_shared/kernels/dynamic.py",
    "moe/_shared/kernels/silu.py",
)
TARGET = "/opt/spark3/candidate/b12x/b12x/moe/fused_moe"

base = json.loads(Path("config/cluster.json").read_text())
det = copy.deepcopy(base)
for name in ("_impl.py", "_preparation.py", "_tuning.py"):
    det["container"]["mounts"].append([f"{OVERLAY}/{name}", f"{TARGET}/{name}", "ro"])
# Routed-MoE combine through per-route rows and B12X's fixed-order top-k sum.
det["environment"]["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
# Also dense split-K through the FP32 workspace reducer instead of atomics.
detsk = copy.deepcopy(det)
detsk["environment"]["B12X_DENSE_SPLITK_TURBO"] = "0"

# 0005 as well: deterministic output keeps the W4A8 decode regime and direct
# routing. detfast-t keeps split-K turbo on to price the FP32 reducer.
detfast_t = copy.deepcopy(base)
for name in ("_impl.py", "_preparation.py", "_tuning.py"):
    detfast_t["container"]["mounts"].append([f"{DECODE_OVERLAY}/{name}", f"{TARGET}/{name}", "ro"])
detfast_t["environment"]["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
detfast = copy.deepcopy(detfast_t)
detfast["environment"]["B12X_DENSE_SPLITK_TURBO"] = "0"

# 0006 as well: deterministic fused decode keeps one task per intermediate
# slice and stores slice partials for the fixed-order sum.
detslice = copy.deepcopy(base)
for name in SLICES_FILES:
    detslice["container"]["mounts"].append(
        [f"{SLICES_OVERLAY}/{name}", f"{SLICES_TARGET}/{name}", "ro"]
    )
detslice["environment"]["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
detslice["environment"]["B12X_DENSE_SPLITK_TURBO"] = "0"

# Diagnostic: detslice without the shared-expert side-stream overlap.
detslice_noovl = copy.deepcopy(detslice)
detslice_noovl["environment"]["VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD"] = "0"

for name, config in (("det", det), ("detsk", detsk), ("detfast", detfast),
                     ("detfast-t", detfast_t), ("detslice", detslice),
                     ("detslice-noovl", detslice_noovl)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)

# Decode profiles: r5m and detfast with the torch profiler (profile_decode.py).
for name, config in (("r5m-prof", base), ("detfast-prof", detfast)):
    profiled = copy.deepcopy(config)
    profiled["serve_args"] += [
        "--profiler-config",
        json.dumps({
            "profiler": "torch",
            "torch_profiler_dir": f"/cache/kkref/profiles/det-{name}",
            "torch_profiler_with_stack": False,
            "ignore_frontend": True,
            "torch_profiler_use_gzip": True,
        }),
    ]
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(profiled, indent=2) + "\n")
    print("wrote", path)
