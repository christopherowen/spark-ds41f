#!/usr/bin/env python3
"""Write the decode-kernel window's arms from the TP4 serving configurations.

- control.json: TileLang TP4 on -tilelang-1m-v3 with vLLM 0041 and 0042 mounted
  (the TP3 benchmark's overlay), its own profiler directory;
- candidate.json: control plus this experiment's kernel files (overlay/),
  with its own DSpark cost and profiler directories;
- candidate-nopf.json: candidate without the L2 weight prefetch (diagnostic);
- b12x.json: the r5p B12X TP4 recipe, with a profiler directory.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
IMAGE_VLLM = "/opt/spark3/candidate/vllm/vllm"
FIX_FILES = (  # vLLM 0041 and 0042
    "v1/worker/gpu/input_batch.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/spec_decode/dflash/speculator.py",
)
KERNEL_FILES = (
    "models/deepseek_v4_1/tilelang/gemm.py",
    "models/deepseek_v4_1/tilelang/linear.py",
    "models/deepseek_v4_1/l2_prefetch.py",
)


def mount(path):
    return [f"{{home}}/projects/spark-ds41f/{REL}/overlay/vllm/{path}", f"{IMAGE_VLLM}/{path}", "ro"]


def set_profiler(cluster, name):
    args = cluster["serve_args"]
    profiler = {"profiler": "torch", "torch_profiler_dir": f"/cache/kkref/profiles/{name}",
                "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                "ignore_frontend": True, "torch_profiler_use_gzip": True}
    value = json.dumps(profiler, separators=(",", ":"))
    if "--profiler-config" in args:
        args[args.index("--profiler-config") + 1] = value
    else:
        args += ["--profiler-config", value]


def main():
    base = json.loads((ROOT / "experiments/2026-10-05-tp4-500k/tilelang.json").read_text())
    control = copy.deepcopy(base)
    control["container"]["mounts"] += [mount(f) for f in FIX_FILES]
    set_profiler(control, "ring4-tl-control-20261005")
    candidate = copy.deepcopy(control)
    candidate["container"]["mounts"] += [mount(f) for f in KERNEL_FILES]
    candidate["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/ring4-tl-decode-v3-20261005"
    set_profiler(candidate, "ring4-tl-decode-v3-20261005")
    b12x = json.loads((ROOT / "experiments/2026-10-05-tp4-500k/b12x.json").read_text())
    set_profiler(b12x, "ring4-b12x-1m-20261005")
    # Diagnostic: the candidate without the L2 weight prefetch, to separate kernel
    # time from prefetch overlap in the decode profile.
    no_prefetch = copy.deepcopy(candidate)
    no_prefetch["environment"]["VLLM_DS41_L2_PREFETCH"] = "0"
    no_prefetch["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/ring4-tl-decode-v3-nopf-20261005"
    set_profiler(no_prefetch, "ring4-tl-decode-v3-nopf-20261005")
    # Arm b: the candidate with every Q-B and indexer Q-B decode CTA resident
    # (128 CTAs of 16 x 64 and 16 x 32 tiles) instead of 64 at one per SM.
    alt = copy.deepcopy(candidate)
    gemm = "models/deepseek_v4_1/tilelang/gemm.py"
    alt["container"]["mounts"] = [m for m in alt["container"]["mounts"] if not m[1].endswith(gemm)]
    alt["container"]["mounts"].append(
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay-b/vllm/{gemm}", f"{IMAGE_VLLM}/{gemm}", "ro"])
    alt["environment"]["SPARK3_DSPARK_COST_DIR"] = "/cache/kkref/dspark-costs/ring4-tl-decode-v3b-20261005"
    set_profiler(alt, "ring4-tl-decode-v3b-20261005")
    for name, cluster in (("control", control), ("candidate", candidate), ("candidate-nopf", no_prefetch),
                          ("candidate-b", alt), ("b12x", b12x)):
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
