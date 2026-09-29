#!/usr/bin/env python3
"""Write this experiment's arms from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-display-carveout-kv")
OVERLAY = "{home}/spark3-overlay/display-kv/vllm/v1/worker"
TARGET = "/opt/spark3/candidate/vllm/vllm/v1/worker"
FILES = ("display_carveout.py", "gpu/model_runner.py")
# The embedding and output head free 842.5 MiB per rank; grow the KV cache by
# 0.8 GiB of it and keep the rest as margin.
KV_GROWTH = 858_993_459

base = json.loads(Path("config/cluster.json").read_text())
arm = copy.deepcopy(base)
arm["container"]["docker_run_args"].append("--device=/dev/dri/card0:/dev/dri/card0:rw")
arm["container"]["mounts"] += [[f"{OVERLAY}/{name}", f"{TARGET}/{name}", "ro"] for name in FILES]
arm["environment"]["SPARK3_DISPLAY_CARVEOUT_WEIGHTS"] = "1"
args = arm["serve_args"]
kv = args.index("--kv-cache-memory-bytes") + 1
args[kv] = str(int(args[kv]) + KV_GROWTH)

long = copy.deepcopy(arm)
largs = long["serve_args"]
largs[largs.index("--max-model-len") + 1] = str(256 * 1024)

for name, config in (("weights", arm), ("weights-256k", long)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
