#!/usr/bin/env python3
"""Write the MoE exchange experiment's kernel bundle and arms from the vLLM branch.

usage: make_bundles.py VLLM_REPO [BASE] [HEAD]

BASE defaults to 26382f5cf (the package files of candidate 6's image, spark-ds41f-1025)
and HEAD to the branch ``moe-exchange-direct``. Every file the branch changes between
them goes into the bundle: package files under overlay/vllm (the lab mounts them over
the image's tree), tests and benchmarks as mounts. The bundle (bundles/exchange) runs
the MoE and SP prefill tests, then the gated benchmark of the combine into the exchange
slots against the combine and copies it replaces. The package files also go to
bundles/serving/overlay, which exchange-tp4.json mounts over candidate 6 (the serving
check), and prof-exchange.json / prof-candidate6.json are the two arms under the torch
profiler (the request inventory's settings without Python stacks).

The kernel audit's make_audit_bundles.py, with this experiment's bundle and arms.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-25-moe-exchange"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
CANDIDATE6 = "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json"
FIXTURES = REPO / "experiments/2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# name: (description, [(label, argv after python, timeout seconds[, env])]): env sets a
# step's environment variables over the bundle's (None unsets one).
PARTS = {
    "exchange": (
        "the SP prefill MoE combine written straight into the exchange slots, with the "
        "exchange under it, and the runner and SP code that hand it over",
        [
            # The deferred combine against the one combine, beside the experts' other
            # kernel tests.
            ("MoE kernel tests", PYTEST + [
                "--basetemp", "/c/pytest-exchange-moe",
                "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"], 3600),
            # The exchange slots with the real combine, the slice order, the runner's
            # and experts' hand-over, and the SP row layout and model forwards.
            ("SP prefill tests", PYTEST + [
                "--basetemp", "/c/pytest-exchange-sp",
                "tests/models/test_deepseek_v41_sp_project_reduce_scatter.py",
                "tests/models/test_deepseek_v4_1_sp_prefill.py"], 3600),
            ("combine-into-slots benchmark",
             ["benchmarks/kernels/benchmark_deepseek_v41_moe_exchange.py", "--gate"], 3600),
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

PROFILER = {
    "profiler": "torch",
    "torch_profiler_dir": "/cache/spark-ds41f/profiles/moe-exchange",
    "torch_profiler_with_stack": False,
    "torch_profiler_record_shapes": False,
    "ignore_frontend": True,
    "torch_profiler_use_gzip": True,
    "torch_profiler_dump_cuda_time_total": False,
}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, head: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"], check=True,
                          capture_output=True).stdout


def read_cluster(relative: str, shape: str | None = None) -> dict:
    """``bin/spark``'s read_cluster: an arm resolved through what it extends."""
    cluster = json.loads((REPO / relative).read_text())
    if recipe.is_overlay(cluster):
        base = cluster["extends"]
        return recipe.overlay(cluster, read_cluster(base["config"], base.get("shape")))
    return recipe.resolve(cluster, shape) if recipe.is_recipe(cluster) else cluster


def write_bundle(repo: str, head: str, commit: str, changed: list[str], part: str) -> None:
    what, steps = PARTS[part]
    bundle = HERE / "bundles" / part
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "overlay").mkdir(parents=True)
    mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
              ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
    for path in changed:
        if path.startswith("vllm/"):
            target = bundle / "overlay" / path
        else:
            name = path.replace("/", "_")
            target = bundle / name
            mounts.append([name, f"{ROOT}/{path}"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(show(repo, head, path))
    shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
    candidate = {
        "description": (
            f"The MoE exchange (vLLM branch moe-exchange-direct {commit}, {len(changed)} "
            f"files over 26382f5cf) on image -next-6 (candidate 6, spark-ds41f-1025): {what}; "
            "GPU tests, then the gated benchmark against the combine and copies the branch "
            "replaces."),
        "image": IMAGE,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "TILELANG_CACHE_DIR": f"/c/tilelang-moe-{part}",
            "VLLM_DS41_KERNEL_BACKEND": "tilelang",
            "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
        },
        "workdir": "vllm",
        "argv": ["/b/run.py"],
        "verdict": "exit",
        "mounts": mounts,
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    print(f"{bundle.name}: {len(changed)} files, {len(steps)} steps")


def write_arms(repo: str, head: str, commit: str, changed: list[str]) -> None:
    serving = HERE / "bundles" / "serving" / "overlay"
    if serving.exists():
        shutil.rmtree(serving)
    package = [path for path in changed if path.startswith("vllm/")]
    for path in package:
        target = serving / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(show(repo, head, path))
    base = read_cluster(CANDIDATE6)
    mounts = [mount for mount in base["container"]["mounts"]
              if mount[1] not in recipe.MODEL_MOUNTS]
    mounts += [[f"{{home}}/projects/spark-ds41f/{EXPERIMENT}/bundles/serving/overlay/{path}",
                f"{ROOT}/{path}", "ro"] for path in package]
    arms = {
        "exchange-tp4.json": {
            "notes": (f"The MoE exchange's package files (vLLM branch moe-exchange-direct "
                      f"{commit}) mounted over candidate 6 at tp4-ring: the serving check "
                      "(outputs identical to candidate 6 at temperature 0, no undeclared "
                      "kernel) and the prefill rates."),
            "extends": {"config": CANDIDATE6},
            "set": {"container": {"mounts": mounts}},
        },
    }
    for name, config, label in (("prof-candidate6.json", CANDIDATE6, "candidate 6"),
                                ("prof-exchange.json", f"{EXPERIMENT}/exchange-tp4.json",
                                 "the MoE exchange")):
        args = list(base["serve_args"])
        assert "--profiler-config" not in args, "candidate 6 already profiles"
        arms[name] = {
            "notes": (f"{label} under the torch profiler: the request inventory's settings "
                      "(experiments/2026-10-25-request-inventory/prof-ports.json) without "
                      "Python stacks or shapes, which inflated its traces and its host "
                      "time; serve arguments are candidate 6's, resolved."),
            "extends": {"config": config},
            "set": {"serve_args": [*args, "--profiler-config",
                                   json.dumps(PROFILER, separators=(", ", ": "))]},
        }
    for name, arm in arms.items():
        (HERE / name).write_text(json.dumps(arm, indent=1) + "\n")
    print(f"serving: {len(package)} package files of {commit}; arms {', '.join(arms)}")


def main() -> None:
    repo = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "26382f5cf"
    head = sys.argv[3] if len(sys.argv) > 3 else "moe-exchange-direct"
    commit = git(repo, "rev-parse", "--short=9", head).strip()
    status = [line for line in git(repo, "diff", "--name-status", base, head).split("\n") if line]
    assert all(line.split("\t")[0] in ("A", "M") for line in status), status
    changed = [line.split("\t")[-1] for line in status]
    for part in PARTS:
        write_bundle(repo, head, commit, changed, part)
    write_arms(repo, head, commit, changed)


if __name__ == "__main__":
    main()
