#!/usr/bin/env python3
"""Write the kernel-audit bundles and arms from the vLLM branch.

usage: make_audit_bundles.py VLLM_REPO [BASE] [HEAD] [--next]

BASE defaults to 26382f5cf (the package files of candidate 6's image, spark-ds41f-1025)
and HEAD to the branch ``kernel-audit``. Every file the branch changes between them goes
into each bundle: package files under overlay/vllm (the lab mounts them over the image's
tree), tests and benchmarks as mounts. Each bundle runs one family's GPU tests, then its
gated benchmarks against the kernels the branch replaces (bundles/audit-*). The package
files also go to bundles/serving/overlay, which audit-tp4.json mounts over candidate 6
(the serving check), and prof-audit.json / prof-candidate6.json are the two arms under
the torch profiler (the request inventory's settings without Python stacks).

--next writes the bundles of the audit's port onto vLLM ``next-candidate`` (HEAD
defaults to ``kernel-audit-next``) as bundles/audit-next-*, on the same image: the
overlay then carries every package file next-candidate changes too, and each one it
deletes as a module that fails to import, as a missing one does. Its attention layer
step runs the whole layer test file (the B12X tests in it are gone), without the
kernel-family variable or B12X's path (both gone). It writes no serving arms: the
candidate's own serving check covers them.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import recipe  # noqa: E402

EXPERIMENT = "experiments/2026-10-25-kernel-audit"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
CANDIDATE6 = "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json"
FIXTURES = REPO / "experiments/2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# name: (description, [(label, argv after python, timeout seconds[, env])]): env sets a
# step's environment variables over the bundle's (None unsets one).
PARTS = {
    "attention": (
        "the sparse-MLA metadata (page tables over each row's live pages, each builder in "
        "one launch) and the attention kernels that read it",
        [
            ("metadata and attention kernel tests", PYTEST + [
                "--basetemp", "/c/pytest-audit-attention",
                "tests/kernels/test_deepseek_v41_tilelang_metadata.py",
                # The sparse MLA and the indexer with every dead page poisoned, beside
                # the file's other kernel tests.
                "tests/kernels/attention/test_deepseek_v41_tilelang.py"], 3600),
            # The layer's declarations (its metadata kernels' calls changed). The file's
            # other tests are B12X's, so the backend variable is unset, as the
            # integration bundles run them.
            ("attention layer tests", PYTEST + [
                "--basetemp", "/c/pytest-audit-attention-layer",
                *(f"tests/models/test_deepseek_v4_1_attention.py::{name}" for name in (
                    "test_tilelang_attention_declares_no_b12x_unit",
                    "test_tilelang_binding_prepares_the_attention_staging",
                    "test_tilelang_context_kv_write_needs_no_helper_plan"))], 1800,
             {"VLLM_DS41_KERNEL_BACKEND": None}),
            ("live-page benchmark",
             ["benchmarks/kernels/benchmark_deepseek_v41_metadata_pages.py", "--gate"], 3600),
            ("one-launch builder benchmark",
             ["benchmarks/kernels/benchmark_deepseek_v41_metadata_builder.py", "--gate"], 1800),
        ],
    ),
    # Fix 2 (the mHC projection's fn tiles loaded ahead) was reverted after run 1: the
    # bundle keeps the mHC and model-forward tests, run with the other changes in place.
    "mhc": (
        "the mHC kernels and the model forwards (CED, SP prefill) with the audit's "
        "metadata changes in place",
        [
            ("mHC tests", PYTEST + [
                "--basetemp", "/c/pytest-audit-mhc",
                "tests/kernels/test_deepseek_v41_tilelang_mhc.py"], 3600),
            ("model-forward tests", PYTEST + [
                "--basetemp", "/c/pytest-audit-forward",
                "tests/models/test_deepseek_v4_1_ced_model.py",
                "tests/models/test_deepseek_v4_1_sp_prefill.py"], 3600),
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

# The next port's attention layer step: the whole file.
NEXT_LAYER_STEP = ("attention layer tests", PYTEST + [
    "--basetemp", "/c/pytest-audit-attention-layer",
    "tests/models/test_deepseek_v4_1_attention.py"], 1800)

MISSING = '''# Deleted by vLLM next-candidate; mounted over the image's copy so that importing it
# fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''

PROFILER = {
    "profiler": "torch",
    "torch_profiler_dir": "/cache/spark-ds41f/profiles/kernel-audit",
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


def write_bundle(repo: str, head: str, commit: str, changed: list[str], part: str,
                 deleted: list[str], port: bool) -> None:
    what, steps = PARTS[part]
    name = f"audit-next-{part}" if port else f"audit-{part}"
    if port:
        steps = [NEXT_LAYER_STEP if step[0] == NEXT_LAYER_STEP[0] else step
                 for step in steps]
    bundle = HERE / "bundles" / name
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
    for path in deleted:
        target = bundle / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(MISSING)
    shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
    branch = "kernel-audit-next (the audit's port onto next-candidate)" if port else "kernel-audit"
    gone = f" and {len(deleted)} deleted modules that fail to import" if deleted else ""
    env = {
        "HF_HUB_OFFLINE": "1",
        "TILELANG_CACHE_DIR": f"/c/tilelang-{bundle.name}",
        "VLLM_DS41_KERNEL_BACKEND": "tilelang",
        "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
    }
    if port:
        del env["VLLM_DS41_KERNEL_BACKEND"]
        env["PYTHONPATH"] = "/b:/opt/spark-ds41f/candidate/vllm"
    candidate = {
        "description": (
            f"The kernel audit (vLLM branch {branch} {commit}, {len(changed)} files{gone} "
            f"over 26382f5cf) on image -next-6 (candidate 6, spark-ds41f-1025): {what}; GPU "
            "tests, then the gated benchmarks against the kernels the branch replaces."),
        "image": IMAGE,
        "env": env,
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
        "audit-tp4.json": {
            "notes": (f"The kernel audit's package files (vLLM branch kernel-audit {commit}) "
                      "mounted over candidate 6 at tp4-ring: the serving check (outputs "
                      "identical to candidate 6 at temperature 0, no undeclared kernel)."),
            "extends": {"config": CANDIDATE6},
            "set": {"container": {"mounts": mounts}},
        },
    }
    for name, config, label in (("prof-candidate6.json", CANDIDATE6, "candidate 6"),
                                ("prof-audit.json", f"{EXPERIMENT}/audit-tp4.json",
                                 "the audit")):
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repo")
    parser.add_argument("base", nargs="?", default="26382f5cf")
    parser.add_argument("head", nargs="?")
    parser.add_argument("--next", action="store_true", dest="port")
    args = parser.parse_args()
    repo, base = args.repo, args.base
    head = args.head or ("kernel-audit-next" if args.port else "kernel-audit")
    commit = git(repo, "rev-parse", "--short=9", head).strip()
    status = [line.split("\t") for line in
              git(repo, "diff", "--name-status", "--no-renames", base, head).split("\n")
              if line]
    allowed = ("A", "M", "D") if args.port else ("A", "M")
    assert all(line[0] in allowed for line in status), status
    changed = [line[-1] for line in status if line[0] != "D"]
    deleted = [line[-1] for line in status if line[0] == "D" and line[-1].startswith("vllm/")]
    for part in PARTS:
        write_bundle(repo, head, commit, changed, part, deleted, args.port)
    if not args.port:
        write_arms(repo, head, commit, changed)


if __name__ == "__main__":
    main()
