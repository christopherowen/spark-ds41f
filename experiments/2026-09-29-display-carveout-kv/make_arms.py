#!/usr/bin/env python3
"""Write this experiment's arm from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-display-carveout-kv")
CARVEOUT_BYTES = 2032 << 20  # the most dgx3 can take with its 1920x1080 console kept
OVERLAY = "{home}/spark3-overlay/display-kv/vllm/v1/worker"
TARGET = "/opt/spark3/candidate/vllm/vllm/v1/worker"

base = json.loads(Path("config/cluster.json").read_text())
arm = copy.deepcopy(base)
arm["container"]["docker_run_args"].append("--device=/dev/dri/card0:/dev/dri/card0:rw")
arm["container"]["mounts"] += [
    [f"{OVERLAY}/{name}", f"{TARGET}/{name}", "ro"] for name in ("utils.py", "display_carveout.py")
]
arm["environment"]["SPARK3_KV_DISPLAY_CARVEOUT"] = "1"
args = arm["serve_args"]
args[args.index("--kv-cache-memory-bytes") + 1] = str(CARVEOUT_BYTES)
(E / "cluster-carveout.json").write_text(json.dumps(arm, indent=2) + "\n")
print("wrote", E / "cluster-carveout.json")

# The same backing with a 256K context limit: about 3.1 full windows fit.
long = copy.deepcopy(arm)
largs = long["serve_args"]
largs[largs.index("--max-model-len") + 1] = str(256 * 1024)
(E / "cluster-carveout-256k.json").write_text(json.dumps(long, indent=2) + "\n")
print("wrote", E / "cluster-carveout-256k.json")
