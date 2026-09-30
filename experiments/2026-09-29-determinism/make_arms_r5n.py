#!/usr/bin/env python3
"""Write the r5n-based determinism arms (run from the repository root, config = r5n).

Separate from make_arms.py, whose arms were generated from r5m and document the
earlier rounds. Both arms pin the adaptive-verification cost curves to one
shared directory (vLLM patch 0005), so verification policy is held fixed:
r5n-pin-prof boots first and writes the table; detm-pin-prof reuses it.
detm-pin-prof adds the deterministic MoE (0004-0006 and 0008, det-masked
overlay) with split-K through the FP32 reducer. Both carry the idle torch
profiler for profile_c8.py.
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-determinism")
PIN = "/cache/kkref/dspark-costs/r5n-pin-20260930"
OVERLAY = "{home}/spark3-overlay/det-masked/b12x"
TARGET = "/opt/spark3/candidate/b12x/b12x"
FILES = (
    "moe/fused_moe/_impl.py",
    "moe/fused_moe/_preparation.py",
    "moe/fused_moe/_tuning.py",
    "moe/_shared/kernels/dynamic.py",
    "moe/_shared/kernels/silu.py",
    "moe/_shared/kernels/w4a16/kernel.py",
)
base = json.loads(Path("config/cluster.json").read_text())
assert base["promoted_baseline"] == "2026-09-30-karmic-kraken-r5n", base["promoted_baseline"]


def profiled(config, name):
    config = copy.deepcopy(config)
    config["environment"]["SPARK3_DSPARK_COST_DIR"] = PIN
    config["serve_args"] += [
        "--profiler-config",
        json.dumps({
            "profiler": "torch",
            "torch_profiler_dir": f"/cache/kkref/profiles/det-{name}",
            "torch_profiler_with_stack": False,
            "ignore_frontend": True,
            "torch_profiler_use_gzip": True,
        }),
    ]
    return config


detm = copy.deepcopy(base)
for name in FILES:
    detm["container"]["mounts"].append([f"{OVERLAY}/{name}", f"{TARGET}/{name}", "ro"])
detm["environment"]["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
detm["environment"]["B12X_DENSE_SPLITK_TURBO"] = "0"
for name, config in (("r5n-pin-prof", base), ("detm-pin-prof", detm)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(profiled(config, name), indent=2) + "\n")
    print("wrote", path)
