#!/usr/bin/env python3
"""Write this experiment's overlays, kernel-lab bundles, serving arms and lab specs from
the vLLM branch.

usage: make_bundles.py VLLM_REPO

From experiments/2026-10-26-tp3-budget-memory/make_bundles.py. Everything runs a vLLM
tree over candidate 6's image, whose tree is the release's (BASE, tilelang-ports
26382f5cf). Two trees, one per arm:

- head: next-candidate (HEAD, c579b3fe1);
- pertype: HEAD with the owner and Engram layers' own windows (PERTYPE, 530205c6a on
  the branch ``l2-prefetch-pertype``; the branch's later commits add the benchmark, no
  package file). Its package files are the prototype's (650fe3679 on
  ``l2-prefetch-schedule`` over next-candidate 1bc005688, where the dgx2 runs and the
  first dgx1 runs measured it); next-candidate changes none of them.

The runs before the ramp/tail ones on next-candidate used this file at 4d5e8fc1, with
HEAD 1bc005688 and PERTYPE 650fe3679 (README).

The benchmark and its chain change no package file, so both trees run the branch
head's copy of them.

overlay/ holds every package file HEAD changes from BASE (a module HEAD deletes is one
that fails to import); overlay-pertype/ holds only the files pertype changes from HEAD.
bench/ holds every test and benchmark file the branch adds or changes over the image's
tree, at its path. Bundles and serving arms mount
each package path from the most specific overlay that has it, so the trees share one
copy of HEAD's files. Overlays are rewritten in place; a file this run would not write
stops it, for a person to remove.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = "experiments/2026-10-26-l2-prefetch-schedule"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "26382f5cf"
HEAD = "c579b3fe1"
PERTYPE = "530205c6a"
BRANCH = "l2-prefetch-pertype"
TREES = {"head": HEAD, "pertype": PERTYPE}
RC = "experiments/2026-10-25-spark-ds41f-candidate/rc6"
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
BENCH = "benchmarks/kernels/benchmark_deepseek_v41_l2_schedule.py"
SCHEDULE_TESTS = "tests/models/test_deepseek_v4_1_l2_schedule.py"
FIXTURES = "experiments/2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]
# The files the prototype changes and the tests that cover them.
TESTS = [
    "tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py",
    "tests/model_executor/layers/test_l2_prefetch.py",
    "tests/models/test_glm5next_tilelang_l2_prefetch.py",
    "tests/models/test_deepseek_v4_1_attention.py",
    "tests/models/test_deepseek_v4_1_engram.py",
    "tests/models/test_deepseek_v4_1_sp_prefill.py",
    "tests/models/test_deepseek_v4_1_ced_model.py",
]
# The recipe's prefetch settings (config/cluster.json, both shapes).
PREFETCH_ENV = {"VLLM_L2_PREFETCH_KERNELS": "tilelang", "VLLM_L2_PREFETCH_GRID": "2"}

# name: (tree, description, [(label, argv after python, timeout seconds[, env])])
BUNDLES = {
    "schedule-smoke": (
        "pertype",
        "the schedule benchmark at 6 rows, two rounds: does the harness run",
        [("schedule benchmark, TP4, 6 rows", [BENCH, "--rows", "6", "--rounds", "2",
                                               "--replays", "3", "--trace", "2"], 3600)],
    ),
    "schedule-tp4": (
        "pertype",
        "the window schedules on the forty layers at TP4 shapes, 6, 48 and 96 rows",
        [("schedule benchmark, TP4", [BENCH, "--rows", "6,48,96", "--tp", "4"], 7200)],
    ),
    "schedule-tp3": (
        "pertype",
        "the window schedules on the forty layers at TP3 shapes, 6, 48 and 96 rows",
        [("schedule benchmark, TP3", [BENCH, "--rows", "6,48,96", "--tp", "3"], 7200)],
    ),
    "tests-head": (
        "head",
        "the prototype's test files on next-candidate's own tree, for failures that "
        "predate it",
        [("test files on head", PYTEST + ["--basetemp", "/c/pytest-l2-schedule-head",
                                         *TESTS], 5400)],  # not SCHEDULE_TESTS: head lacks the points
    ),
    "tests-pertype": (
        "pertype",
        "the prototype's test files on its tree",
        [("test files on pertype", PYTEST + ["--basetemp", "/c/pytest-l2-schedule-pertype",
                                            SCHEDULE_TESTS, *TESTS], 5400)],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import os
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout, *rest in STEPS:
    print(f"=== {label}", flush=True)
    env = dict(os.environ)
    for name, value in (rest[0] if rest else {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout, env=env).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code}", flush=True)
    codes.append((label, code))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, ref: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def diff(repo: str, old: str, new: str) -> list[tuple[str, str]]:
    lines = git(repo, "diff", "--no-renames", "--name-status", old, new).split("\n")
    status = [tuple(line.split("\t")) for line in lines if line]
    assert all(kind in ("A", "M", "D") for kind, _ in status), status
    return status


def write_tree(directory: Path, files: dict[str, bytes]) -> None:
    stale = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                   if p.is_file() and str(p.relative_to(directory)) not in files)
    if stale:
        raise SystemExit(f"{directory} holds files this run would not write: {stale}")
    for path, data in files.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main() -> None:
    repo = sys.argv[1]
    heads = {name: git(repo, "rev-parse", "--short=9", ref).strip()
             for name, ref in {**TREES, "branch": BRANCH}.items()}
    # The branch head may only add tests and benchmarks over PERTYPE.
    extra = [path for _, path in diff(repo, PERTYPE, BRANCH) if path.startswith("vllm/")]
    assert not extra, f"{BRANCH} changes package files over {PERTYPE}: {extra}"
    base = {path: (MISSING.encode() if kind == "D" else show(repo, HEAD, path))
            for kind, path in diff(repo, BASE, HEAD) if path.startswith("vllm/")}
    write_tree(HERE / "overlay", base)
    layers = {"head": {}}
    for name in ("pertype",):
        changed = {path: show(repo, TREES[name], path)
                   for kind, path in diff(repo, HEAD, TREES[name]) if path.startswith("vllm/")}
        assert all(kind != "D" for kind, _ in diff(repo, HEAD, TREES[name]))
        write_tree(HERE / f"overlay-{name}", changed)
        layers[name] = changed
    # Every test and benchmark file the branch head adds or changes over the image's
    # tree: the image's copies are the release's, written for modules next-candidate
    # removed.
    bench = [path for kind, path in diff(repo, BASE, BRANCH)
             if kind != "D" and path.startswith(("tests/", "benchmarks/"))]
    assert BENCH in bench and SCHEDULE_TESTS in bench
    write_tree(HERE / "bench", {path: show(repo, BRANCH, path) for path in bench})
    package = sorted(set(base) | {p for files in layers.values() for p in files})

    def source(tree: str, path: str) -> str:
        if path in layers[tree]:
            return f"{EXPERIMENT}/overlay-{tree}/{path}"
        return f"{EXPERIMENT}/overlay/{path}"

    for name, (tree, what, steps) in BUNDLES.items():
        bundle = HERE / "bundles" / name
        bundle.mkdir(parents=True, exist_ok=True)
        mounts = [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                  [FIXTURES, "/b/prepare_fixtures.py"]]
        mounts += [[source(tree, path), f"{ROOT}/{path}"] for path in package]
        mounts += [[f"{EXPERIMENT}/bench/{path}", f"{ROOT}/{path}"] for path in bench]
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"{what} (vLLM {tree} {heads[tree]}: next-candidate {HEAD} with PDL"
                + ("" if tree == "head" else " and this experiment's commits")
                + f"; {len(package)} package files over the release's {BASE}; the "
                f"benchmark from {BRANCH} {heads['branch']}) on candidate 6's image."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": "/c/tilelang-l2-schedule",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
                **PREFETCH_ENV,
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {tree} {heads[tree]}, {len(steps)} steps")

    # Serving arms: candidate 6's recipe at tp4-ring and tp3-triangle with each tree.
    home = "{home}/projects/spark-ds41f/"
    for tree in TREES:
        changed = sorted(layers[tree])
        for shape, rc in (("tp4", "rc-tp4.json"), ("tp3", "rc-tp3.json")):
            arm = {
                "notes": (
                    f"Candidate 6's {'tp4-ring' if shape == 'tp4' else 'tp3-triangle'} "
                    f"recipe on vLLM {tree} {heads[tree]}: next-candidate {HEAD} with PDL"
                    + ("" if tree == "head" else
                       f" and the L2 prefetch schedule's {tree} windows ({', '.join(changed)})")
                    + f"; {len(package)} package files over the image's ({BASE}'s), mounted "
                    "from overlay/" + ("" if tree == "head" else f" and overlay-{tree}/") + "."),
                "extends": {"config": f"{RC}/{rc}"},
                "set": {"container": {"mounts": [
                    [home + "cache", "/cache", "rw"],
                    *[[home + source(tree, path), f"{ROOT}/{path}", "ro"] for path in package],
                ]}},
            }
            write_json(HERE / f"{tree}-{shape}.json", arm)
            write_json(HERE / f"{tree}-{shape}-dev.json", {
                "notes": f"{tree}-{shape}.json with the dev-mode collective RPC, for the "
                         "suites' engine grid.",
                "extends": {"config": f"{EXPERIMENT}/{tree}-{shape}.json"},
                "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
            })
        profiler = {"profiler": "torch", "torch_profiler_dir": f"/cache/spark-ds41f/profiles/l2-schedule-{tree}",
                    "torch_profiler_with_stack": True, "torch_profiler_record_shapes": True,
                    "ignore_frontend": True, "torch_profiler_use_gzip": True,
                    "torch_profiler_dump_cuda_time_total": False}
        write_json(HERE / f"prof-{tree}-tp4.json", {
            "notes": (f"{tree}-tp4.json under the torch profiler with Python stacks and "
                      "record_shapes, as request-inventory-5's prof-ports.json. serve_args are "
                      "candidate 6's at tp4-ring (as the PDL experiment's prof arms resolve "
                      "them) plus --profiler-config."),
            "extends": {"config": f"{EXPERIMENT}/{tree}-tp4.json"},
            "set": {"serve_args": [*SERVE_ARGS_TP4, "--profiler-config", json.dumps(profiler)]},
        })
    write_specs()
    print(f"overlay: {len(base)} files; pertype {len(layers['pertype'])}; arms for "
          f"{', '.join(TREES)} at tp4 and tp3")


# Candidate 6's tp4-ring serve_args as bin/recipe.py resolves them (the PDL experiment's
# prof-pdl.json, without its --profiler-config).
SERVE_ARGS_TP4 = [
    "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170",
    "--served-model-name", "deepseek-v4.1-flash", "--host", "0.0.0.0", "--port", "8000",
    "--tokenizer-mode", "deepseek_v41", "--dtype", "bfloat16", "--tensor-parallel-size", "4",
    "--load-format", "b12x", "--block-size", "256", "--enable-prefix-caching",
    "--enable-prompt-tokens-details", "--enable-chunked-prefill", "--max-model-len", "1048576",
    "--max-num-seqs", "16", "--max-parallel-prefills", "1", "--gpu-memory-utilization", "0.82",
    "--max-num-batched-tokens", "16384", "--kv-cache-memory-bytes", "11274289152",
    "--max-cudagraph-capture-size", "96", "--generation-config", "vllm",
    "--override-generation-config", '{"temperature":1.0,"top_p":1.0}',
    "--engram-config", '{"table_memory":"disk","disk_resident_scales":false,"projection_tp":true}',
    "--attention-backend", "TILELANG", "--linear-backend", "tilelang", "--moe-backend", "tilelang",
    "--speculative-config",
    '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,'
    '"attention_backend":"TILELANG","draft_sample_method":"greedy","rejection_sample_method":"block",'
    '"enable_adaptive_verification":true,"adaptive_verification_row_cost":0.035}',
    "--jit-monitor-mode", "off", "--compilation-config",
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],'
    '"cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,20,24,28,32,40,48,56,64,72,80,88,96]}',
    "--prefix-cache-retention-interval", "8192", "--long-prefill-token-threshold", "16288",
    "--async-scheduling", "--reasoning-parser", "deepseek_v41", "--tool-call-parser", "deepseek_v41",
    "--enable-auto-tool-choice", "--distributed-executor-backend", "mp", "--nnodes", "4",
    "--no-scheduler-reserve-full-isl", "--no-enable-flashinfer-autotune",
    "--limit-mm-per-prompt", '{"image":4}', "--mm-processor-cache-gb", "0",
]


def determinism_extras(streams: tuple[int, ...]) -> list:
    """The determinism gate the PDL and moe-overlap A/Bs run: every arm's hashes, sampled
    and controls digests and distinct streams must be the first arm's."""
    extras = [
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
        ["experiments/2026-10-24-tilelang-completion/sampled_digest.py",
         ["--tokens", "192"], "sampled"],
        ["experiments/2026-10-24-tilelang-completion/controls_digest.py",
         ["--tokens", "96"], "controls-digest"],
    ]
    extras += [["scripts/distinct_streams.py",
                ["--streams", str(n), "--samples", "4", "--tokens", "256"], f"c{n}-distinct"]
               for n in streams]
    return extras


def write_specs() -> None:
    """Cluster A/B specs (not run here: they need this branch on origin/main and a window)."""
    for shape, suite, streams in (("tp4", "screen", (16, 8, 4)), ("tp3", "screen-tp3", (8, 4))):
        write_json(HERE / f"screen-{shape}.json", {
            "experiment": EXPERIMENT,
            "run": f"l2-schedule-screen-{shape}",
            "jobs": [
                {"kind": "sync"},
                {"kind": "measure", "bench": False, "bracket": False,
                 "extras": determinism_extras(streams),
                 "arms": [{"config": f"{tree}-{shape}.json", "label": tree} for tree in TREES]},
                {"kind": "measure", "suite": [suite, "single-stream"],
                 "arms": [{"config": f"pertype-{shape}-dev.json", "label": "pertype"},
                          {"config": f"head-{shape}-dev.json", "label": "head"},
                          {"config": f"pertype-{shape}-dev.json", "label": "pertype-again"}]},
            ],
        })
    write_json(HERE / "profile-tp4.json", {
        "experiment": EXPERIMENT,
        "run": "l2-schedule-profile",
        "jobs": [
            {"kind": "sync"},
            {"kind": "profile", "workloads": ["decode"],
             "workload_args": {"decode": ["--tokens", "48"]},
             "arms": [{"config": f"prof-{tree}-tp4.json", "label": tree} for tree in TREES]},
        ],
    })


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
