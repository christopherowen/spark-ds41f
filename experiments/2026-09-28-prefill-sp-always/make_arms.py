#!/usr/bin/env python3
"""Derive this experiment's arm configs from config/cluster.json.

usage: experiments/2026-09-28-prefill-sp-always/make_arms.py   (from the deployment checkout)
"""
import copy
import json
from pathlib import Path

E = Path(__file__).resolve().parent
ROOT = E.parents[1]
base = json.loads((ROOT / "config/cluster.json").read_text())


def arm(env: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["environment"].update(env)
    cfg["environment"] = dict(sorted(cfg["environment"].items()))
    return cfg


arms = {
    # Promoted: sequence-parallel prefill from 2048 tokens.
    "current": arm({}),
    # Every forward above the decode graph sizes (the code raises 1 to 49).
    "always": arm({"SPARK3_DS41_PREFILL_SP_MIN_ROWS": "1"}),
}
# Patch 0018 mounted over the r5g image (overlay.sh of
# experiments/2026-09-27-prefill-sp): no setting, SP above the decode sizes,
# small reduce-scatters over the RoCE all-reduce.
import subprocess
FILES = subprocess.check_output(
    ["git", "-C", str(Path.home() / "projects/spark3-vllm-ds41f/.work/upstreams/vllm"),
     "diff", "--name-only", "spark3/r5f-check", "spark3/r5f-sp", "--", "vllm"],
    text=True,
).split()
# The branch also carries patch 0019 (custom-op fill_defaults reads the schema
# once, in vllm/env_override.py), screened on its own as "fd".
FD_FILES = ["vllm/env_override.py"]


def overlay(files: list[str], sp: bool) -> dict:
    cfg = arm({})
    cfg["container"]["mounts"] += [
        [f"{{home}}/spark3-overlay/r5f-sp/{f}", f"/opt/spark3/candidate/vllm/{f}", "ro"]
        for f in files
    ]
    if sp:
        cfg["environment"].pop("SPARK3_DS41_PREFILL_SP_MIN_ROWS", None)
    return cfg


arms["always2"] = overlay([f for f in FILES if f not in FD_FILES], sp=True)
arms["fd"] = overlay(FD_FILES, sp=False)
arms["both"] = overlay(FILES, sp=True)
# Profiling variants (capture_tiny.py records a few ~73-token prefills).
for name in ("current", "always2"):
    profile = copy.deepcopy(arms[name])
    profile["serve_args"] += [
        "--profiler-config",
        json.dumps({
            "profiler": "torch",
            "torch_profiler_dir": f"/cache/kkref/profiles/spa-{name}",
            "torch_profiler_with_stack": False,
            "ignore_frontend": True,
            "max_iterations": 32,
            "torch_profiler_use_gzip": True,
        }),
    ]
    arms[f"{name}-profile"] = profile
# Python call stacks for the host side of the current arm (slower, attribution only).
stack = copy.deepcopy(arms["current-profile"])
config = json.loads(stack["serve_args"][-1])
config.update(torch_profiler_dir="/cache/kkref/profiles/spa-current-stack", torch_profiler_with_stack=True)
stack["serve_args"][-1] = json.dumps(config)
arms["current-stack"] = stack
for name, cfg in arms.items():
    (E / f"cluster-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
