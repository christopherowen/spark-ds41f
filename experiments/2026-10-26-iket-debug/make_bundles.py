#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles from the vLLM branch.

usage: make_bundles.py VLLM_REPO [--only NAME ...]

Each bundle runs on candidate 6's image, whose vLLM tree is the release's (BASE,
26382f5cf). Every package file a bundle's ref changes from BASE is mounted over it
(overlay/vllm); a package module the ref deletes is mounted as one that fails to
import (as experiments/2026-10-26-tp3-budget-memory/make_bundles.py does). ``hot-base``
runs next-candidate (NEXT, the branch's base), the others the IKET branch (IKET).

The steps run in order (run.py); each prints its exit status and the bundle fails if
any failed. ``{python}`` in a step is the image's interpreter and ``{out}`` the
bundle's output root under the node's compile cache, /c/iket/<bundle>-<UTC stamp>,
which the runner prints first. dgx4's Nsight Systems is mounted at /opt/nsys, and the
step bundles mount the node's Hugging Face hub (DS4.1-Flash-1017's config and
tokenizer; their weights are dummies) at /hf/hub.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "26382f5cf"
NEXT = "577091997"  # next-candidate when the branch was cut
IKET = "iket-debug"
NSYS = ["/opt/nvidia/nsight-systems/2025.3.2", "/opt/nsys"]
HUB = ["~/.cache/huggingface/hub", "/hf/hub"]
NVDISASM = ["/usr/local/cuda-13.0/bin/nvdisasm", "/opt/nvdisasm"]
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
HOT = "/b/iket_hot.py"
STEP = "/b/iket_step.py"
TABLE = "/b/iket_table.py"
PROFILE = ["run-iket", "--output-dir"]
NSYS_PROFILE = ["/opt/nsys/bin/nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--export=sqlite", "--force-overwrite=true", "--output"]

TESTS = ["tests/kernels/test_deepseek_v41_tilelang_iket.py"]
PYTEST = ["{python}", "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest"]
# The production recipe's variables that the TP1 engine reads (config/cluster.json).
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

BUNDLES = {
    "hot-base": dict(
        ref=NEXT,
        what="next-candidate without the IKET mode: the hot kernels' generated sources and output bits",
        steps=[("hot kernels, next-candidate", ["{python}", HOT, "off", "--out", "{out}/off"], 3600)],
    ),
    "hot": dict(
        ref=IKET,
        what="the IKET mode on the hot kernels (routed experts, sparse MLA decode, the mHC entry, "
             "sparknet's all-reduce with simulated peers): its tests, off, on outside the profiler, "
             "on under run-iket, and nsys durations of both builds",
        tests=TESTS,
        steps=[
            ("IKET mode tests", [*PYTEST, "--basetemp", "{out}/pytest", *TESTS], 1800),
            ("off", ["{python}", HOT, "off", "--out", "{out}/off"], 3600),
            ("on, outside the profiler",
             ["{python}", HOT, "on", "--iket", "{out}/on/iket", "--out", "{out}/on"], 3600),
            ("on, under run-iket",
             [*PROFILE, "{out}/trace", "profile", "--postprocess", "all", "--",
              "{python}", HOT, "on", "--iket", "{out}/trace-run/iket", "--out", "{out}/trace-run",
              "--iters", "3"], 5400),
            ("all-reduce, peers 20 us late, under run-iket",
             [*PROFILE, "{out}/trace-late", "profile", "--postprocess", "all", "--",
              "{python}", HOT, "on", "--iket", "{out}/trace-late-run/iket", "--out", "{out}/trace-late-run",
              "--iters", "3", "--only", "allreduce", "--delay-us", "20"], 3600),
            ("nsys, off", [*NSYS_PROFILE, "{out}/nsys-off", "{python}", HOT, "off",
                           "--out", "{out}/nsys-off-run", "--iters", "50"], 3600),
            ("nsys, on", [*NSYS_PROFILE, "{out}/nsys-on", "{python}", HOT, "on",
                          "--iket", "{out}/nsys-on-run/iket", "--out", "{out}/nsys-on-run", "--iters", "50"],
             3600),
            ("per-phase table",
             ["{python}", TABLE, "{out}/trace", "--sources", "{out}/trace-run/iket/sources",
              "--nsys", "off={out}/nsys-off.sqlite", "on={out}/nsys-on.sqlite", "--json", "{out}/table.json"],
             600),
            ("per-phase table, peers 20 us late",
             ["{python}", TABLE, "{out}/trace-late", "--sources", "{out}/trace-late-run/iket/sources",
              "--json", "{out}/table-late.json"], 600),
        ],
    ),
    "sass": dict(
        ref=IKET,
        what="where ptxas puts the IKET events' timer reads around barriers, in the hot kernels",
        env={"NVDISASM": "/opt/nvdisasm"},
        mounts=[NVDISASM],
        steps=[
            ("hot kernels, on", ["{python}", HOT, "on", "--iket", "{out}/on/iket", "--out", "{out}/on",
                                 "--iters", "1"], 3600),
            ("SASS check", ["{python}", "/b/sass_check.py", "{out}/on/iket/sources", "{out}/sass",
                            "oneshot", "moe_grouped", "sparse_mla", "mhc_finalize"], 1800),
        ],
    ),
    "step": dict(
        ref=IKET,
        what="decode steps end to end at TP1 (six layers and the drafter, dummy weights, the "
             "production recipe's kernels and graphs): off, on outside the profiler (four CTAs a "
             "launch record ranges), on under run-iket, and nsys durations off",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[
            ("decode steps, off", ["{python}", STEP, "off", "--out", "{out}/off"], 5400),
            ("decode steps, on", ["{python}", STEP, "on", "--iket", "{out}/on/iket", "--ctas", "4",
                                  "--out", "{out}/on"], 5400),
            # run-iket sizes its pool as if every launch of the run were in flight at
            # once (898 GB here: 16,894 launches, 5,016 timestamps a warp); the pool
            # is reused as launches finish, so a fixed one holds the step's.
            ("decode steps, on, under run-iket",
             ["run-iket", "--context-buffer-size", "8G", "--output-dir", "{out}/trace", "profile",
              "--postprocess", "all", "--",
              "{python}", STEP, "on", "--iket", "{out}/trace-run/iket", "--ctas", "4",
              "--out", "{out}/trace-run"], 7200),
            ("decode steps, nsys, off", [*NSYS_PROFILE, "{out}/nsys-off", "--cuda-graph-trace=node",
                                         "{python}", STEP, "off", "--out", "{out}/nsys-off-run"], 5400),
            ("decode-step table",
             ["{python}", TABLE, "{out}/trace", "--sources", "{out}/trace-run/iket/sources",
              "--between", "iket_step_marker_kernel", "--nsys", "off={out}/nsys-off.sqlite",
              "--json", "{out}/table.json"], 1800),
        ],
    ),
    "tests": dict(
        ref=IKET,
        what="the IKET mode's tests alone",
        tests=TESTS,
        steps=[("IKET mode tests", [*PYTEST, "--basetemp", "{out}/pytest", *TESTS], 1800)],
    ),
    "inspect": dict(
        ref=IKET,
        what="the image's run-iket CLI sources, copied out for reading (how it injects into a "
             "process and its children, the dry run, the buffer sizing)",
        steps=[("copy iket.cli", ["{python}", "-c",
                                  "import iket.cli, pathlib, shutil, sys; "
                                  "src = pathlib.Path(iket.cli.__file__).parent; "
                                  "shutil.copytree(src, sys.argv[1], ignore=shutil.ignore_patterns('__pycache__'))",
                                  "{out}/iket-cli"], 300),
               ("copy iket.profiler's Python", ["{python}", "-c",
                                                "import iket.profiler, pathlib, shutil, sys; "
                                                "src = pathlib.Path(iket.profiler.__file__).parent; "
                                                "shutil.copytree(src, sys.argv[1], ignore=shutil.ignore_patterns("
                                                "'__pycache__', '*.so', 'perfetto_trace_pb2.py'))",
                                                "{out}/iket-profiler"], 300)],
    ),
    "step-off": dict(
        ref=IKET,
        what="decode steps end to end at TP1 (six layers and the drafter, dummy weights, the "
             "production recipe's kernels and graphs) with the IKET mode off: does the engine run",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[("decode steps, off", ["{python}", STEP, "off", "--out", "{out}/off"], 5400)],
    ),
}

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
SCRIPTS = {"iket_hot.py": HERE / "harness" / "iket_hot.py", "iket_step.py": HERE / "harness" / "iket_step.py",
           "sass_check.py": HERE / "harness" / "sass_check.py", "iket_table.py": HERE / "iket_table.py"}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def show(repo, ref, path):
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def main():
    repo = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, spec in BUNDLES.items():
        if only and name not in only:
            continue
        ref, tests = spec["ref"], spec.get("tests", [])
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  git(repo, "diff", "--no-renames", "--name-status", BASE, ref).splitlines() if line]
        package = [(kind, path) for kind, path in status if path.startswith("vllm/")]
        bundle = HERE / "bundles" / name
        planned = {"run.py", "steps.json", "candidate.json", *SCRIPTS}
        planned |= {f"overlay/{path}" for _, path in package}
        planned |= {path.replace("/", "_") for path in tests}
        # Files are overwritten in place, never deleted: a file this run would not
        # write stops it, for a person to remove.
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        for kind, path in package:
            target = bundle / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if kind == "D" else show(repo, ref, path))
        for path in tests:
            (bundle / path.replace("/", "_")).write_bytes(show(repo, ref, path))
        for target, source in SCRIPTS.items():
            shutil.copy(source, bundle / target)
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(spec["steps"], indent=1) + "\n")
        candidate = {
            "description": f"{spec['what']} (vLLM {ref} {head}, {len(package)} package files over the "
                           f"release's {BASE}) on candidate 6's image.",
            "image": IMAGE,
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
            "mounts": [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                       *([target, f"/b/{target}"] for target in SCRIPTS), NSYS, *spec.get("mounts", []),
                       *([path.replace("/", "_"), f"{ROOT}/{path}"] for path in tests)],
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(package)} package files, {len(spec['steps'])} steps")


if __name__ == "__main__":
    main()
