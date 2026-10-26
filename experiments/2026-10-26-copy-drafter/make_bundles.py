#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles (dgx2, ``kernel-local``) from the vLLM
branch.

usage: make_bundles.py VLLM_REPO [NAME...]

From experiments/2026-10-26-block-verify-cut/make_bundles.py. vLLM ``copy-drafts`` is
commits on next-candidate c579b3fe1 (NEXT). The image, spark-ds41f:19f2c20ed4d6-next-6,
holds the release's tree (tilelang-ports 26382f5cf, BASE), so every file a ref changes
from BASE goes into its bundle: package files under overlay/vllm (the lab mounts them
over the image's tree), a package module the ref deletes as one that fails to import,
tests and benchmarks as mounts. Bundle files are overwritten in place, never deleted: a
file a run would not write stops it, for a person to remove.

- copy-tests: the branch's tests: the scheduler's copy allocation, the copy kernels and
  proposer, the TileLang verification kernels at DSpark's block and at the copy
  lookahead (16), adaptive and greedy verification, the input preparation.
- next-tests: next-candidate's own tree, the existing test files of copy-tests: any
  failure that predates the branch.
- copy-bench: the gated benchmarks: the copy kernels (contexts 4K to 1M, 1 to 16
  requests) and the wide verification step at c1, c8 and c16 shapes (the rejection
  sampler's kernels at the lookahead against next-candidate's at the block, and the
  routed experts' GEMMs at those rows).
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
NEXT = "c579b3fe1"
BRANCH = "copy-drafts"
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
# Bundle-only scripts (this folder), mounted at /b.
SCRIPTS = ("wide_verify_rows.py",)
BASELINE = "vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]
LOOKAHEAD = {"VLLM_TEST_VERIFICATION_STEPS": "16"}


def existing_steps(name: str) -> list:
    """The existing test files the branch touches the paths of."""
    return [
        ("verification tests at the block", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-verification",
            "tests/v1/spec_decode/test_tilelang_rejection.py"], 5400),
        ("adaptive and greedy verification, dead rows, input preparation", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-inputs",
            "tests/v1/spec_decode/test_adaptive_verification.py",
            "tests/v1/worker/gpu/test_greedy_verification.py",
            "tests/v1/worker/test_dead_rows.py",
            "tests/v1/worker/test_gpu_input_batch_v2.py",
            "tests/v1/worker/test_prepare_kernels.py"], 5400),
        ("scheduler tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-scheduler",
            "tests/v1/core/test_aligned_prefill_chunks.py",
            "tests/v1/core/test_sink_draft_zero.py"], 3600),
        # A wide block's rows: the compressor's state ring, and rows bit-equal
        # whatever the step's row count (GEMMs to 200 rows, MoE to 300, sparse
        # MLA's routes, mHC).
        ("DS4.1 decode-row kernels", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-decode-rows",
            "tests/kernels/test_deepseek_v41_tilelang_compress.py",
            "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
            "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
            "tests/kernels/attention/test_deepseek_v41_tilelang.py",
            "tests/kernels/test_deepseek_v41_tilelang_mhc.py"], 5400),
    ]


def copy_steps(name: str) -> list:
    return [
        ("copy-draft scheduler tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-copy-scheduler",
            "tests/v1/core/test_copy_drafts_scheduler.py"], 1800),
        ("copy kernels and proposer", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-copy",
            "tests/v1/spec_decode/test_copy_drafts.py"], 5400),
        ("verification tests at the copy lookahead", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-verification-16",
            "tests/v1/spec_decode/test_tilelang_rejection.py"], 5400, LOOKAHEAD),
    ] + existing_steps(name)


BENCH_COPY = "benchmarks/kernels/benchmark_copy_drafts.py"


def bench_steps() -> list:
    return [
        ("copy kernels, and verification compiled at the lookahead and the block",
         [BENCH_COPY, "--gate"], 3600),
        ("wide verification: the routed experts at c1, c8 and c16 steps",
         ["/b/wide_verify_rows.py"], 3600),
    ]


# name: (ref, description, steps, baseline rejection kernels)
CHANGES = {
    "copy-tests": (
        BRANCH,
        "copy drafts beside DSpark: the branch's tests",
        copy_steps("copy"),
        False,
    ),
    "next-tests": (
        NEXT,
        "next-candidate's own tree: the existing test files of copy-tests, for the "
        "failures that predate the branch",
        existing_steps("next"),
        False,
    ),
    "copy-bench": (
        BRANCH,
        "copy drafts beside DSpark: the gated benchmarks of the copy kernels and the "
        "wide verification step",
        bench_steps(),
        False,
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
    names = sys.argv[2:] or list(CHANGES)
    for name in names:
        ref, what, steps, baseline = CHANGES[name]
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  git(repo, "diff", "--no-renames", "--name-status", BASE, ref).split("\n")
                  if line]
        assert all(kind in ("A", "M", "D") for kind, _ in status), status
        deleted = {path for kind, path in status if kind == "D"}
        changed = [path for kind, path in status
                   if kind != "D" or path.startswith("vllm/")]
        bundle = HERE / "bundles" / name
        scripts = SCRIPTS if name == "copy-bench" else ()
        planned = {"prepare_fixtures.py", "run.py", "steps.json", "candidate.json",
                   *scripts}
        if baseline:
            planned.add("baseline_tilelang_rejection.py")
        planned |= {f"overlay/{p}" if p.startswith("vllm/") else p.replace("/", "_")
                    for p in changed}
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts
                       and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        (bundle / "overlay").mkdir(parents=True, exist_ok=True)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
        for path in changed:
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
            else:
                target = bundle / path.replace("/", "_")
                mounts.append([target.name, f"{ROOT}/{path}"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if path in deleted else show(repo, ref, path))
        for script in scripts:
            shutil.copy(HERE / script, bundle / script)
            mounts.append([script, f"/b/{script}"])
        if baseline:
            target = bundle / "baseline_tilelang_rejection.py"
            target.write_bytes(show(repo, NEXT, BASELINE))
            mounts.append([target.name, "/b/baseline/tilelang_rejection.py"])
        shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"{what} (vLLM {ref} {head}, {len(changed)} files over the release's "
                f"{BASE}) on candidate 6's image."),
            "image": IMAGE,
            # next-candidate has no kernel-family switch and imports no B12X package.
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-copy-drafter-{name}",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            # A benchmark's own gates and exit statuses are its verdict; a test
            # suite's is pytest's summary.
            "verdict": "status" if name == "copy-bench" else "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(changed)} files, {len(steps)} steps")


if __name__ == "__main__":
    main()
