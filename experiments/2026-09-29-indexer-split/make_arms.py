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

for name, config in (("baseline-profile", profile),):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
