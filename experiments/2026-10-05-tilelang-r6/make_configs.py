#!/usr/bin/env python3
"""Write r6's TP4 configurations from the TP4 TileLang recipe (experiments/2026-10-05-tp4-500k).

- tp4.json: the r6 image (this directory's lock: the TileLang 1M series plus
  vLLM 0041-0044), its own DSpark cost directory, nothing mounted over the
  image and no profiler;
- tp4-profile.json: tp4.json with the torch profiler endpoints, for the
  per-kernel decode profile of the benchmark window.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
IMAGE = "vllm-ds41f-kkref:04c30fa98e79-r6"
VLLM_TREE = json.loads((HERE / "source.json").read_text())["vllm"]["expected_tree"]


def main():
    tp4 = json.loads((ROOT / "experiments/2026-10-05-tp4-500k/tilelang.json").read_text())
    tp4["upstreams_config"] = f"{REL}/upstreams.lock.json"
    tp4["container"]["image"] = IMAGE
    tp4["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/ring4-r6-20261005"
    tp4["container"]["expected_labels"]["local.spark3.vllm.tree"] = VLLM_TREE
    assert not [m for m in tp4["container"]["mounts"] if "overlay" in m[0]]
    args = tp4["serve_args"]
    i = args.index("--profiler-config")
    del args[i:i + 2]
    profile = copy.deepcopy(tp4)
    value = json.dumps({"profiler": "torch", "torch_profiler_dir": "/cache/kkref/profiles/ring4-r6-20261005",
                        "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                        "ignore_frontend": True, "torch_profiler_use_gzip": True}, separators=(",", ":"))
    profile["serve_args"] += ["--profiler-config", value]
    for name, cluster in (("tp4", tp4), ("tp4-profile", profile)):
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
