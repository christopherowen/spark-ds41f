#!/usr/bin/env python3
"""Write the indexer-split arms from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-indexer-split")
base = json.loads(Path("config/cluster.json").read_text())

# The promoted r5k configuration with the torch profiler (capture_depth.py).
profile = copy.deepcopy(base)
profile["serve_args"] += [
    "--profiler-config",
    json.dumps({
        "profiler": "torch",
        "torch_profiler_dir": "/cache/kkref/profiles/idx-baseline",
        "torch_profiler_with_stack": False,
        "ignore_frontend": True,
        "torch_profiler_use_gzip": True,
    }),
]

# Patch 0025 (this directory) mounted over r5k's attention module, with its
# own vLLM compile cache. split-check also runs the full-row indexer and
# compares every rank's rows.
OVERLAY = "{home}/spark3-overlay/indexer-split/vllm/models/deepseek_v4_1/attention.py"
TARGET = "/opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/attention.py"
CACHE = "/cache/kkref/jit/vllm-r5k-indexer-split"
split = copy.deepcopy(base)
split["container"]["mounts"].append([OVERLAY, TARGET, "ro"])
split["environment"].update(VLLM_CACHE_DIR=CACHE, VLLM_CACHE_ROOT=CACHE)
check = copy.deepcopy(split)
check["environment"]["SPARK3_DS41_INDEXER_SPLIT_CHECK"] = "1"

for name, config in (("baseline-profile", profile), ("split", split), ("split-check", check)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
