#!/usr/bin/env python3
"""Write the r5k candidate arms from config/cluster.json (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-r5k")
TAG = "vllm-ds41f-kkref:04c30fa98e79-r5k"
TREES = {
    "local.spark3.vllm.tree": "9ba14ba1c4b77fcfdb727784b3b869bd8e382c18",
    "local.spark3.b12x.tree": "640c8544b3242e858c962ad6de98febd8f74e8e3",
}
KV_BYTES = 1_503_238_553 + 858_993_459  # r5j KV plus 0.8 GiB of the freed 842.5 MiB
CACHE = "/cache/kkref/jit/vllm-r5k"

base = json.loads(Path("config/cluster.json").read_text())
arm = copy.deepcopy(base)
arm["container"]["image"] = TAG
arm["container"]["expected_labels"].update(TREES)
arm["container"]["docker_run_args"].append("--device=/dev/dri/card0:/dev/dri/card0:rw")
arm["environment"].update(
    SPARK3_DISPLAY_CARVEOUT_WEIGHTS="1", VLLM_CACHE_DIR=CACHE, VLLM_CACHE_ROOT=CACHE
)
args = arm["serve_args"]
args[args.index("--kv-cache-memory-bytes") + 1] = str(KV_BYTES)
args[args.index("--max-model-len") + 1] = str(256 * 1024)

auto = copy.deepcopy(arm)
auto["environment"]["VLLM_DS41_ATTENTION_COMPUTE"] = "auto"

for name, config in (("candidate", arm), ("candidate-auto", auto)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
