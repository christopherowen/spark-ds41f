#!/usr/bin/env python3
"""Write r6's TP4 and TP3 configurations from the TileLang recipes of 2026-10-05.

- tp4.json: the r6 image (this directory's lock: the TileLang 1M series plus
  vLLM 0041-0044), its own DSpark cost directory, nothing mounted over the
  image and no profiler;
- tp4-profile.json: tp4.json with the torch profiler endpoints, for the
  per-kernel decode profile of the benchmark window;
- tp3.json and tp3-profile.json: the same for the TP3 recipe (8 x 512K), from
  the 2026-10-05 TP3 TileLang benchmark's configuration.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
IMAGE = "vllm-ds41f-kkref:04c30fa98e79-r6"
VLLM_TREE = json.loads((HERE / "source.json").read_text())["vllm"]["expected_tree"]


def profiled(cluster, name):
    """The configuration with the torch profiler endpoints writing to profiles/<name>."""
    profile = copy.deepcopy(cluster)
    value = json.dumps({"profiler": "torch", "torch_profiler_dir": f"/cache/kkref/profiles/{name}",
                        "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                        "ignore_frontend": True, "torch_profiler_use_gzip": True}, separators=(",", ":"))
    profile["serve_args"] += ["--profiler-config", value]
    return profile


def without_profiler(cluster):
    args = cluster["serve_args"]
    if "--profiler-config" in args:
        i = args.index("--profiler-config")
        del args[i:i + 2]


def tp3():
    """The TP3 TileLang benchmark's configuration (2026-10-05) on r6: 0041 and 0042
    come from the image instead of mounted files, and the checkout and cache
    paths follow the rename to spark-ds41f."""
    cluster = json.loads((ROOT / "experiments/2026-10-05-tp3-benchmark/tilelang.json").read_text())
    cluster["upstreams_config"] = f"{REL}/upstreams.lock.json"
    cluster["container"]["image"] = IMAGE
    cluster["container"]["expected_labels"]["local.spark3.vllm.tree"] = VLLM_TREE
    cluster["container"]["mounts"] = [
        [m[0].replace("/projects/spark3-vllm-ds41f/", "/projects/spark-ds41f/"), *m[1:]]
        for m in cluster["container"]["mounts"] if "/overlay/" not in m[0]]
    cluster["deployment"]["repository"] = "https://github.com/christopherowen/spark-ds41f.git"
    cluster["deployment"]["path"] = "{home}/projects/spark-ds41f"
    cluster["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/tp3-r6-20261005"
    without_profiler(cluster)
    text = json.dumps(cluster)
    assert "spark3-ring4-qualification" not in text and "spark3-vllm-ds41f" not in text, "stale path"
    return cluster


def main():
    tp4 = json.loads((ROOT / "experiments/2026-10-05-tp4-500k/tilelang.json").read_text())
    tp4["upstreams_config"] = f"{REL}/upstreams.lock.json"
    tp4["container"]["image"] = IMAGE
    tp4["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/ring4-r6-20261005"
    tp4["container"]["expected_labels"]["local.spark3.vllm.tree"] = VLLM_TREE
    assert not [m for m in tp4["container"]["mounts"] if "overlay" in m[0]]
    without_profiler(tp4)
    three = tp3()
    for name, cluster in (("tp4", tp4), ("tp4-profile", profiled(tp4, "ring4-r6-20261005")),
                          ("tp3", three), ("tp3-profile", profiled(three, "tp3-r6-20261005"))):
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
