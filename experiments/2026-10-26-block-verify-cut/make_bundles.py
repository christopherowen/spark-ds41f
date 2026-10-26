#!/usr/bin/env python3
"""Write this experiment's dgx4 kernel-lab bundles from the vLLM branch.

usage: make_bundles.py VLLM_REPO

From experiments/2026-10-26-tp3-budget-memory/make_bundles.py (its next-candidate
bundles). vLLM block-verify-cut is three commits on next-candidate 1bc005688 (NEXT).
The image, spark-ds41f:19f2c20ed4d6-next-6, holds the release's tree (tilelang-ports
26382f5cf, BASE), so every file a ref changes from BASE goes into its bundle: package
files under overlay/vllm (the lab mounts them over the image's tree), a package module
the ref deletes as one that fails to import, tests and benchmarks as mounts. NEXT's
tilelang_rejection.py goes to /b/baseline, what the benchmark times the branch against
(``--baseline /b/baseline``).

- cut-tests: the branch's verification, rejection-sampler, sink and preparation tests.
- cut-bench: the gated rejection-sampler benchmark against NEXT's kernels at 1 to 96
  rows (c1 is 6, c8 48, c16 96), without dead rows and with them (``--dead-rows``).
- next-tests: NEXT's own tree, the same test files: any failure that predates the
  branch.
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
NEXT = "1bc005688"
BRANCH = "block-verify-cut"
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
BASELINE = "vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]


def test_steps(name: str) -> list:
    return [
        ("verification tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-verification",
            "tests/v1/spec_decode/test_tilelang_rejection.py"], 5400),
        # Unbiasedness through rejection_sample (the TileLang path for one-hot drafts),
        # chunking, int64 offsets, adaptive verification and its dead rows.
        ("rejection sampler tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-rejection",
            "tests/v1/spec_decode/test_rejection_sampler_utils.py",
            "tests/v1/worker/test_gpu_rejection_sampler_chunking.py",
            "tests/v1/worker/test_gpu_rejection_sampler_i64.py",
            "tests/v1/spec_decode/test_adaptive_verification.py",
            "tests/v1/worker/test_dead_rows.py"], 5400),
        # The sink's candidates (the alternates of a request's last draw) and policy.
        ("sink kernel tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-sink-kernels",
            "tests/v1/worker/test_gpu_tilelang_sampling.py", "-k", "sink"], 3600),
        ("sink tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-sink",
            "tests/v1/worker/test_gpu_sink.py"], 3600),
        ("preparation tests", PYTEST + [
            "--basetemp", f"/c/pytest-{name}-prepare",
            "tests/v1/worker/test_prepare_kernels.py",
            "tests/v1/worker/test_gpu_input_batch_v2.py"], 3600),
    ]


BENCH = "benchmarks/kernels/benchmark_tilelang_rejection.py"

# name: (ref, description, [(label, argv after python, timeout seconds)]).
CHANGES = {
    "cut-tests": (
        BRANCH,
        "block verification ends each block at the dead-row cut: its tests",
        test_steps("cut"),
    ),
    "cut-bench": (
        BRANCH,
        "block verification ends each block at the dead-row cut: the gated benchmark "
        "against next-candidate's kernels",
        [
            ("rejection-sampler benchmark against next-candidate",
             [BENCH, "--gate", "--baseline", "/b/baseline"], 3600),
            ("rejection-sampler benchmark against next-candidate, dead rows",
             [BENCH, "--gate", "--baseline", "/b/baseline", "--dead-rows",
              "--kernels", "rejection_sample,verify,resample"], 3600),
        ],
    ),
    "next-tests": (
        NEXT,
        "next-candidate's own tree: the same test files, for the failures that "
        "predate the branch",
        test_steps("next"),
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
    for name, (ref, what, steps) in CHANGES.items():
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  git(repo, "diff", "--no-renames", "--name-status", BASE, ref).split("\n")
                  if line]
        assert all(kind in ("A", "M", "D") for kind, _ in status), status
        # A deleted package module is mounted as one that fails to import; a deleted
        # test or benchmark is not run.
        deleted = {path for kind, path in status if kind == "D"}
        changed = [path for kind, path in status
                   if kind != "D" or path.startswith("vllm/")]
        bundle = HERE / "bundles" / name
        # Files are overwritten in place, never deleted: a file this run would not
        # write (a path the ref no longer changes) stops it, for a person to remove.
        planned = {"prepare_fixtures.py", "run.py", "steps.json", "candidate.json",
                   "baseline_tilelang_rejection.py"}
        planned |= {f"overlay/{p}" if p.startswith("vllm/") else p.replace("/", "_")
                    for p in changed}
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
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
        baseline = bundle / "baseline_tilelang_rejection.py"
        baseline.write_bytes(show(repo, NEXT, BASELINE))
        mounts.append([baseline.name, "/b/baseline/tilelang_rejection.py"])
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
                "TILELANG_CACHE_DIR": f"/c/tilelang-block-verify-{name}",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(changed)} files, {len(steps)} steps")


if __name__ == "__main__":
    main()
