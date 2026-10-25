#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles and serving arms from the vLLM branches.

usage: make_bundles.py VLLM_REPO [--only NAME ...]

From experiments/2026-10-25-moe-overlap/make_bundles.py. Each lever is one vLLM commit
on the release's source (BASE, tilelang-ports 26382f5cf, which candidate 6's image
holds with two test-only patches), on a branch of its own, so that each arm carries one
variable. For each, every file the branch changes from BASE goes into bundles/<change>:
package files under overlay/vllm (the lab mounts them over the image's tree), tests and
benchmarks as mounts, and the release's copy of a module its benchmark compares against
(``baseline_*``). <change>-tp3.json and <change>-tp4.json are serving arms: candidate
6's arm on the triangle and on the ring (rc6/rc-tp3.json, rc6/rc-tp4.json) with the
same package files mounted.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_EXPERIMENT = "experiments/2026-10-26-tp3-budget-memory"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "26382f5cf"
BASE_ARMS = {
    "tp3": "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp3.json",
    "tp4": "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json",
}
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

MODEL_TESTS = [
    "tests/models/test_deepseek_v4_1_attention.py",
    "tests/models/test_deepseek_v4_1_ced_model.py",
    "tests/models/test_deepseek_v4_1_sp_prefill.py",
    "tests/models/test_deepseek_v4_dspark_context_kv.py",
]

# Tests that fail on the release's own tree (bundles/release, runs/release.json: hand-built
# attention modules that lack attributes the release's geometry now sets). Every lever's
# bundle skips them, so its verdict reads its own change; the release bundle runs them.
RELEASE_FAILURES = (
    "test_indexer_declares_bounded_score_rows_at_model_context_capacity",
    "test_indexer_primer_restores_live_cache",
    "test_wo_prefill_remainders_reuse_declared_chunk_capacity",
    "test_wo_preparation_exact_rows_owns_output_and_replays",
)


def without_release_failures(step):
    label, argv, *rest = step
    if "tests/models/test_deepseek_v4_1_attention.py" not in argv:
        return step
    return (label, [*argv, "-k", f"not ({' or '.join(RELEASE_FAILURES)})"], *rest)


# change: (branch, description, baseline modules, [(label, argv after python, timeout
# seconds[, env])], extra files of this experiment mounted at /b); env sets a step's
# variables over the bundle's (None unsets one). ``memory-cuts`` is a branch merging the
# three memory cuts' branches over BASE (git merge, in lever order).
CHANGES = {
    "prefill-tiles": (
        "prefill-moe-tiles",
        "the routed experts take 64-row tiles in prefill calls (from 2,048 tokens), "
        "decode keeps 32",
        ["vllm/models/deepseek_v4_1/tilelang/moe.py"],
        [
            ("MoE, SP prefill and CED model tests", PYTEST + [
                "--basetemp", "/c/pytest-tp3-levers-prefill-tiles",
                "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
                "tests/models/test_deepseek_v4_1_sp_prefill.py",
                "tests/models/test_deepseek_v4_1_ced_model.py"], 5400),
            ("prefill tiles benchmark", [
                "benchmarks/kernels/benchmark_deepseek_v41_moe_prefill_tiles.py",
                "--baseline", "/b/baseline/moe.py", "--gate"], 5400),
        ],
        [],
    ),
    "unread-buffers": (
        "unread-buffers",
        "the model and the drafter allocate no budget-sized buffer nothing reads",
        [],
        [("model tests", PYTEST + ["--basetemp", "/c/pytest-tp3-levers-unread-buffers",
                                   *MODEL_TESTS], 5400)],
        [],
    ),
    "drafter-lane": (
        "drafter-lane-scratch",
        "each TileLang attention layer reserves its scratch in its own model's workspace "
        "lane",
        [],
        [("workspace and model tests", PYTEST + [
            "--basetemp", "/c/pytest-tp3-levers-drafter-lane",
            "tests/v1/worker/test_workspace.py", *MODEL_TESTS], 5400),
         ("lane reservations by layer role", ["/b/lane_check.py"], 1800)],
        ["lane_check.py"],
    ),
    "candidate-ids": (
        "candidate-block-ids",
        "layer 20's candidates are stored as 2,048 block ids a row, not 16,384 positions",
        ["vllm/models/deepseek_v4_1/tilelang/indexer.py"],
        [("indexer, attention and model tests", PYTEST + [
            "--basetemp", "/c/pytest-tp3-levers-candidate-ids",
            "tests/kernels/attention/test_deepseek_v41_tilelang.py", *MODEL_TESTS], 5400),
         ("candidate ids against the release's positions", [
             "benchmarks/kernels/benchmark_deepseek_v41_candidate_ids.py",
             "--baseline", "/b/baseline/indexer.py", "--gate"], 5400)],
        [],
    ),
    "release": (
        BASE,
        "the release's own tree, no overlay: every test file the levers' bundles run, for "
        "the failures that predate them",
        [],
        [("the levers' test files on the release", PYTEST + [
            "--basetemp", "/c/pytest-tp3-levers-release",
            "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
            "tests/v1/worker/test_workspace.py",
            "tests/kernels/attention/test_deepseek_v41_tilelang.py", *MODEL_TESTS], 5400)],
        [],
    ),
    "memory-cuts": (
        "memory-cuts",
        "the three memory cuts together (unread buffers, the drafter's lane, candidate "
        "block ids)",
        [],
        [("workspace, indexer, attention and model tests", PYTEST + [
            "--basetemp", "/c/pytest-tp3-levers-memory-cuts",
            "tests/v1/worker/test_workspace.py",
            "tests/kernels/attention/test_deepseek_v41_tilelang.py", *MODEL_TESTS], 5400)],
        [],
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


def main() -> None:
    repo, only = sys.argv[1], sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, (ref, what, baselines, steps, extras) in CHANGES.items():
        if only and name not in only:
            continue
        if name != "release":
            steps = [without_release_failures(step) for step in steps]
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line for line in git(repo, "diff", "--name-status", BASE, ref).split("\n")
                  if line]
        assert all(line.split("\t")[0] in ("A", "M") for line in status), status
        changed = [line.split("\t")[-1] for line in status]
        bundle = HERE / "bundles" / name
        # Files are overwritten in place, never deleted: a file this run would not write
        # (a path the branch no longer changes) stops it, for a person to remove.
        planned = {"prepare_fixtures.py", "run.py", "steps.json", "candidate.json", *extras}
        planned |= {f"overlay/{p}" if p.startswith("vllm/") else p.replace("/", "_")
                    for p in changed}
        planned |= {f"baseline_{Path(p).name}" for p in baselines}
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        (bundle / "overlay").mkdir(parents=True, exist_ok=True)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
        package = []
        for path in changed:
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
                package.append(path)
            else:
                target = bundle / path.replace("/", "_")
                mounts.append([target.name, f"{ROOT}/{path}"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(show(repo, ref, path))
        for path in baselines:
            target = bundle / f"baseline_{Path(path).name}"
            target.write_bytes(show(repo, BASE, path))
            mounts.append([target.name, f"/b/baseline/{Path(path).name}"])
        for extra in extras:
            shutil.copy(HERE / extra, bundle / extra)
            mounts.append([extra, f"/b/{extra}"])
        shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"{what} (vLLM {ref} {head}, {len(changed)} files over the "
                f"release's {BASE}) on candidate 6's image: GPU tests, then the gated "
                "benchmark or check."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-tp3-levers-{name}",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        experiment = "{home}/projects/spark-ds41f/" + REPO_EXPERIMENT
        arm_mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
        arm_mounts += [[f"{experiment}/bundles/{name}/overlay/{path}", f"{ROOT}/{path}", "ro"]
                       for path in package]
        for shape, base in BASE_ARMS.items() if package else ():
            arm = {
                "notes": (f"Candidate 6 ({shape}) with {what}: the package files of vLLM "
                          f"{ref} {head} mounted over the image "
                          f"({len(package)})."),
                "extends": {"config": base},
                "set": {"container": {"mounts": arm_mounts}},
            }
            (HERE / f"{name}-{shape}.json").write_text(json.dumps(arm, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(package)} package files, "
              f"{len(changed) - len(package)} test and benchmark files, {len(steps)} steps")


if __name__ == "__main__":
    main()
