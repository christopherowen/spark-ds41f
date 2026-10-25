#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles and serving arms from the vLLM branches.

usage: make_bundles.py VLLM_REPO

Each change is one vLLM commit on the release's source (BASE, tilelang-ports 26382f5cf,
which candidate 6's image holds with two test-only patches), on a branch of its own so
that each arm carries one variable: ``mhc-mxfp8-cast``. (``moe-evict-first`` was dropped
after its dgx4 run, README "Results"; its bundle stays as that run's record.) For each,
every file the branch changes from BASE goes into bundles/<change>: package files under
overlay/vllm (the lab mounts them over the image's tree), tests and benchmarks as
mounts, and the release's copy of the module its benchmark compares against
(``baseline_*``). <change>-tp4-dev.json and <change>-tp3-dev.json are the serving arms:
candidate 6's dev arm on the ring and on the triangle, with the same package files mounted.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_EXPERIMENT = "experiments/2026-10-25-moe-overlap"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "26382f5cf"
# Candidate 6 (spark-ds41f-1025's image) dev arms by shape: the ring and the triangle.
BASE_ARMS = {
    "tp4": "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4-dev.json",
    "tp3": "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp3-dev.json",
}
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# change: (branch, description, baseline modules, [(label, argv after python, timeout
# seconds[, env])]); env sets a step's variables over the bundle's (None unsets one).
CHANGES = {
    "mhc-cast": (
        "mhc-mxfp8-cast",
        "mHC finalize writes its output's MXFP8 cast; the projections it feeds (fused "
        "Q-A/KV, shared gate/up, routed experts) stop casting",
        ["vllm/models/deepseek_v4_1/tilelang/mhc.py"],
        [
            ("mHC, linear, MoE and model-forward tests", PYTEST + [
                "--basetemp", "/c/pytest-moe-overlap-mhc-cast",
                "tests/kernels/test_deepseek_v41_tilelang_mhc.py",
                "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
                "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
                "tests/models/test_deepseek_v4_1_ced_model.py",
                "tests/models/test_deepseek_v4_1_sp_prefill.py"], 5400),
            ("cast benchmark", [
                "benchmarks/kernels/benchmark_deepseek_v41_mhc_cast.py",
                "--baseline", "/b/baseline/mhc.py", "--gate"], 3600),
        ],
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
    repo = sys.argv[1]
    for name, (branch, what, baselines, steps) in CHANGES.items():
        head = git(repo, "rev-parse", "--short=9", branch).strip()
        status = [line for line in git(repo, "diff", "--name-status", BASE, branch).split("\n")
                  if line]
        assert all(line.split("\t")[0] in ("A", "M") for line in status), status
        changed = [line.split("\t")[-1] for line in status]
        bundle = HERE / "bundles" / name
        if bundle.exists():
            shutil.rmtree(bundle)
        (bundle / "overlay").mkdir(parents=True)
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
            target.write_bytes(show(repo, branch, path))
        for path in baselines:
            target = bundle / f"baseline_{Path(path).name}"
            target.write_bytes(show(repo, BASE, path))
            mounts.append([target.name, f"/b/baseline/{Path(path).name}"])
        shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"{what} (vLLM branch {branch} {head}, {len(changed)} files over the "
                f"release's {BASE}) on candidate 6's image: GPU tests, then the gated "
                "benchmark against the release's kernels."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-moe-overlap-{name}",
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
        for shape, base in BASE_ARMS.items():
            arm = {
                "notes": (f"Candidate 6 ({shape}, dev-mode RPC) with {what}: the package files "
                          f"of vLLM branch {branch} {head} mounted over the image "
                          f"({len(package)})."),
                "extends": {"config": base},
                "set": {"container": {"mounts": arm_mounts}},
            }
            (HERE / f"{name}-{shape}-dev.json").write_text(json.dumps(arm, indent=1) + "\n")
        print(f"{name}: {branch} {head}, {len(package)} package files, "
              f"{len(changed) - len(package)} test and benchmark files, {len(steps)} steps")


if __name__ == "__main__":
    main()
