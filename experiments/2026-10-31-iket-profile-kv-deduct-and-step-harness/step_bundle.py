#!/usr/bin/env python3
"""Write a kernel-local bundle that times DS4.1 decode steps kernel by kernel and phase by phase.

usage: step_bundle.py VLLM_REPO REF BUNDLE_DIR [--tp 1|3|4] [--requests N] [--tokens N]
                      [--ctas N] [--iters N] [--warps N] [--granularity {fine,phase}]
                      [--payloads] [--iket-cache-dir DIR] [--deduct-kv-bytes]
                      [--use-tracker-dir DIR] [--base-mode {iket,nsys}]
                      [--skip REGEX] [--image IMAGE --base COMMIT]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:1030"
BASE = "b6de07aad"
NSYS = ["/opt/nvidia/nsight-systems/2025.3.2", "/opt/nsys"]
HUB = ["~/.cache/huggingface/hub", "/hf/hub"]
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
STEP = "/b/iket_step.py"
TABLE = "/b/iket_table.py"
NSYS_PROFILE = [
    "/opt/nsys/bin/nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
    "--export=sqlite", "--force-overwrite=true", "--output",
]
STEP_ENV = {
    "TRANSFORMERS_OFFLINE": "1",
    "MAX_JOBS": "8",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_DS41_HEAD_DTYPE": "float32",
    "VLLM_DS41_PACKED_BF16_LM_HEAD": "1",
    "VLLM_L2_PREFETCH_KERNELS": "tilelang",
    "VLLM_L2_PREFETCH_GRID": "2",
    "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2",
    "SPARK_DS41F_DSPARK_MAIN_PROJ_TP": "1",
    "SPARK_DS41F_DSPARK_VOCAB_PARALLEL": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
}
RECIPE_ENV = {
    **STEP_ENV,
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_DS41_KERNEL_BACKEND": "tilelang",
    "VLLM_LATE_ADMISSION_LAYER": "4",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_MMAP_THRESHOLD_": "1048576",
    "OMP_NUM_THREADS": "8",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_DISABLE_PYNCCL": "0",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
    "VLLM_MAX_IMAGE_PIXELS": "16777216",
}
RECIPE_SIZING = {4: ["--max-num-seqs", "16"], 3: ["--max-num-seqs", "8"]}


def recipe_sizing(tp, max_model_len=1048576, kv_bytes=6 << 30):
    return [
        *RECIPE_SIZING.get(tp, []),
        "--max-model-len", str(max_model_len),
        "--kv-bytes", str(kv_bytes),
        "--prefix-caching",
        "--capture-ladder",
    ]


def step_spec(
    ref,
    tp=4,
    requests=1,
    tokens=32,
    ctas=1,
    iters=1,
    warps=0,
    granularity="phase",
    payloads=False,
    iket_cache_dir="/c/iket-tilelang-cache",
    deduct_kv_bytes=True,
    use_tracker_dir="",
    base_mode="iket",
    skip="^(vision_|topk_topp)",
    grid=8,
    grid_rows=0,
):
    eff_grid_rows = grid_rows or (2 if requests >= 8 else 0)
    shape = [
        "--tp-shape", str(tp),
        "--requests", str(requests),
        "--tokens", str(tokens),
        "--grid", str(grid),
        *(["--grid-rows", str(eff_grid_rows)] if eff_grid_rows else []),
    ]
    # Both IKET and base runs use --prefix-caching and --capture-ladder so both runs replay
    # the exact same CUDA graph sizes; the IKET run keeps an 8K context to bound indexer warps.
    iket_sizing = recipe_sizing(tp, max_model_len=8192, kv_bytes=2 << 30)
    base_sizing = recipe_sizing(tp, max_model_len=8192 if base_mode == "iket" else 1048576,
                                kv_bytes=2 << 30 if base_mode == "iket" else 6 << 30)
    profile_opts = [
        "--log-level", "debug",
        *(["--deduct-kv-bytes"] if deduct_kv_bytes else []),
        *(["--use-tracker-dir", use_tracker_dir] if use_tracker_dir else []),
    ]
    iket_opts = [
        "--iket", "{out}/trace-run/iket",
        "--ctas", str(ctas),
        "--iters", str(iters),
        "--warps", str(warps),
        "--granularity", granularity,
        *(["--payloads"] if payloads else []),
        *(["--iket-cache-dir", iket_cache_dir] if iket_cache_dir else []),
        "--skip", skip,
    ]
    steps = [
        (
            "decode steps, on, under run-iket",
            [
                "{python}", "/b/iket_profile.py", "{out}", *profile_opts, "--",
                "{python}", STEP, "on", *iket_opts, *shape, *iket_sizing,
                "--out", "{out}/trace-run",
            ],
            7200,
        ),
    ]
    if base_mode == "iket":
        steps.extend([
            (
                "decode steps, off, under run-iket --warp-trace",
                [
                    "{python}", "/b/iket_profile.py", "{out}/base-warp-trace",
                    "--warp-trace", "--",
                    "{python}", STEP, "off", *shape, *base_sizing,
                    "--out", "{out}/base-warp-trace-run",
                ],
                5400,
            ),
            (
                "decode steps against the grid's step",
                [
                    "{python}", "/b/step_diff.py",
                    "{out}/base-warp-trace/trace/iket.decoded_results.json",
                    "--grid-runs", str(grid + 1),
                    "--json", "{out}/step-diff.json",
                ],
                600,
            ),
        ])
    else:
        steps.extend([
            (
                "decode steps, nsys, off, graphs",
                [
                    *NSYS_PROFILE, "{out}/nsys-graphs", "--cuda-graph-trace=node",
                    "{python}", STEP, "off", *shape, *base_sizing,
                    "--out", "{out}/nsys-graphs-run",
                ],
                5400,
            ),
            (
                "decode steps against the grid's step",
                [
                    "{python}", "/b/step_diff.py", "{out}/nsys-graphs.sqlite",
                    "--grid-runs", str(grid + 1),
                    "--json", "{out}/step-diff.json",
                ],
                600,
            ),
        ])
    return dict(
        ref=ref,
        what=(
            f"DS4.1 decode steps timed by kernel and phase: TP{tp} rank shapes at TP1, "
            f"c{requests}, {tokens} tokens, {grid} engine-grid steps (grid_rows={eff_grid_rows or 'default'}); "
            f"ctas={ctas}, iters={iters}, warps={warps}, granularity={granularity}, base_mode={base_mode}"
        ),
        env={
            **RECIPE_ENV,
            **({"VLLM_DS41_IKET_CACHE_DIR": iket_cache_dir} if iket_cache_dir else {}),
        },
        mounts=[HUB],
        steps=steps,
    )


RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. ``{python}`` is this interpreter and ``{out}`` the output root."""
import json
import os
import subprocess
import sys
import time

BUNDLE = os.environ["IKET_BUNDLE"]
OUT = f"/c/iket/{BUNDLE}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
os.makedirs(OUT)
print(f"=== output {OUT}", flush=True)
codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    argv = [a.replace("{python}", sys.executable).replace("{out}", OUT) for a in argv]
    print(f"=== {label}", flush=True)
    started = time.monotonic()
    try:
        code = subprocess.run(argv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f} s", flush=True)
    codes.append((label, code))
print(json.dumps({"output": OUT, "steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def _show(repo, ref, path):
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def write_bundle(repo, name, spec, bundle, image=IMAGE, base=BASE):
    ref = spec["ref"]
    head = _git(repo, "rev-parse", "--short=9", ref).strip()
    status = [line.split("\t") for line in
              _git(repo, "diff", "--no-renames", "--name-status", base, ref).splitlines() if line]
    package = [(kind, path) for kind, path in status if path.startswith("vllm/")]
    bundle = Path(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    scripts = {
        "iket_step.py": HERE / "harness" / "iket_step.py",
        "iket_profile.py": HERE / "harness" / "iket_profile.py",
        "step_diff.py": HERE / "step_diff.py",
    }
    for kind, path in package:
        target = bundle / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(MISSING.encode() if kind == "D" else _show(repo, ref, path))
    for target, source in scripts.items():
        if source.exists():
            shutil.copy(source, bundle / target)
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(spec["steps"], indent=1) + "\n")
    candidate = {
        "description": f"{spec['what']} (vLLM {ref} {head}, {len(package)} package files over "
                       f"{base}) on {image}.",
        "image": image,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "IKET_BUNDLE": name,
            "TILELANG_CACHE_DIR": f"/c/tilelang-iket-{name}",
            "TRITON_CACHE_DIR": f"/c/triton-iket-{name}",
            "VLLM_CACHE_ROOT": f"/c/vllm-iket-{name}",
            "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
            **spec.get("env", {}),
        },
        "workdir": "vllm",
        "argv": ["/b/run.py"],
        "verdict": "status",
        "mounts": [
            ["run.py", "/b/run.py"],
            ["steps.json", "/b/steps.json"],
            *([target, f"/b/{target}"] for target in scripts),
            NSYS,
            *spec.get("mounts", []),
        ],
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    return f"{name}: {ref} {head}, {len(package)} package files, {len(spec['steps'])} steps"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("ref")
    parser.add_argument("bundle")
    parser.add_argument("--tp", type=int, choices=(1, 3, 4), default=4)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--ctas", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warps", type=int, default=0)
    parser.add_argument("--granularity", choices=("fine", "phase"), default="phase")
    parser.add_argument("--payloads", action="store_true")
    parser.add_argument("--iket-cache-dir", default="/c/iket-tilelang-cache")
    parser.add_argument("--deduct-kv-bytes", action="store_true", default=True)
    parser.add_argument("--use-tracker-dir", default="")
    parser.add_argument("--base-mode", choices=("iket", "nsys"), default="iket")
    parser.add_argument("--skip", default="^(vision_|topk_topp)")
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--base", default=BASE, help="the image's vLLM tree")
    args = parser.parse_args()
    iket = "vllm/models/deepseek_v4_1/tilelang/iket.py"
    if subprocess.run(["git", "-C", args.repo, "cat-file", "-e", f"{args.ref}:{iket}"],
                      capture_output=True).returncode:
        raise SystemExit(f"{args.ref} has no {iket}: cherry-pick vLLM branch iket-debug's commits")
    bundle = Path(args.bundle).resolve()
    spec = step_spec(
        args.ref,
        tp=args.tp,
        requests=args.requests,
        tokens=args.tokens,
        ctas=args.ctas,
        iters=args.iters,
        warps=args.warps,
        granularity=args.granularity,
        payloads=args.payloads,
        iket_cache_dir=args.iket_cache_dir,
        deduct_kv_bytes=args.deduct_kv_bytes,
        use_tracker_dir=args.use_tracker_dir,
        base_mode=args.base_mode,
        skip=args.skip,
        grid_rows=args.grid_rows,
    )
    print(write_bundle(args.repo, bundle.name, spec, bundle, args.image, args.base))


if __name__ == "__main__":
    main()
