#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles from the vLLM refs.

usage: make_bundles.py VLLM_REPO [--only NAME ...]

From experiments/2026-10-26-tp3-budget-memory/make_bundles.py. Both bundles run over
candidate 6's image, whose tree is the release's (BASE, tilelang-ports 26382f5cf):
every file the ref changes from BASE is mounted, package files under overlay/vllm and
tests and benchmarks at their tree paths, and a package module the ref deletes is
mounted as one that fails to import.

- ``next-candidate``: vLLM next-candidate (NEXT): the CED test files there, and the
  chunking check as a record of the per-chunk anchor.
- ``anchor``: ced-prompt-anchor: the CED test files with the new ones, the chunking
  check (every chunking bit-identical) and the CED kernel benchmark.

Both mount ced-prompt-anchor's decoder-stack test at /b/ced_replay_stack.py for the
chunking check, which imports it. Files are written in place, never deleted.
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
ANCHOR = "ced-prompt-anchor"
STACK = "tests/kernels/attention/test_deepseek_v41_ced_replay.py"
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]
CED_TESTS = [
    "tests/v1/attention/test_b12x_v41_ced_metadata.py",
    "tests/kernels/test_deepseek_v41_tilelang_ced.py",
    "tests/kernels/test_deepseek_v41_tilelang_metadata.py",
    "tests/models/test_deepseek_v4_1_ced_model.py",
    "tests/models/test_deepseek_v4_1_engram.py",
    "tests/kernels/moe/test_deepseek_v41_tilelang_router.py",
    "tests/v1/worker/test_gpu_tilelang_inputs.py",
    "tests/v1/worker/test_gpu_model_runner_v2.py",
    "tests/v1/core/test_aligned_prefill_chunks.py",
]

# name: (ref, description, steps [(label, argv after python, timeout seconds)])
BUNDLES = {
    "next-candidate": (
        NEXT,
        "next-candidate's own tree: the CED test files, and the chunking check as a "
        "record of the per-chunk replay anchor",
        [
            ("CED test files on next-candidate", PYTEST + [
                "--basetemp", "/c/pytest-ced-anchor-next", *CED_TESTS], 5400),
            ("chunking check (record)", ["/b/chunking_check.py", "--expect", "record"],
             3600),
        ],
    ),
    "anchor": (
        ANCHOR,
        "CED's decoder replay anchored at the prompt end: the CED test files and the "
        "new decoder-stack test, the chunking check and the CED kernel benchmark",
        [
            ("CED test files", PYTEST + [
                "--basetemp", "/c/pytest-ced-anchor", *CED_TESTS, STACK], 5400),
            ("chunking check (every chunking bit-identical)",
             ["/b/chunking_check.py", "--expect", "identical"], 3600),
            ("CED kernel benchmark", ["benchmarks/kernels/benchmark_deepseek_v41_ced.py"],
             5400),
        ],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout in STEPS:
    print(f"=== {label}", flush=True)
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout).returncode
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
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, (ref, what, steps) in BUNDLES.items():
        if only and name not in only:
            continue
        head = git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  git(repo, "diff", "--no-renames", "--name-status", BASE, ref).split("\n")
                  if line]
        assert all(kind in ("A", "M", "D") for kind, _ in status), status
        deleted = {path for kind, path in status if kind == "D"}
        changed = [path for kind, path in status
                   if kind != "D" or path.startswith("vllm/")]
        bundle = HERE / "bundles" / name
        extras = ["chunking_check.py", "ced_replay_stack.py"]
        planned = {"prepare_fixtures.py", "run.py", "steps.json", "candidate.json",
                   *extras}
        planned |= {f"overlay/{p}" if p.startswith("vllm/") else p.replace("/", "_")
                    for p in changed}
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        (bundle / "overlay").mkdir(parents=True, exist_ok=True)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                  ["chunking_check.py", "/b/chunking_check.py"],
                  ["ced_replay_stack.py", "/b/ced_replay_stack.py"]]
        for path in changed:
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
            else:
                target = bundle / path.replace("/", "_")
                mounts.append([target.name, f"{ROOT}/{path}"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if path in deleted else show(repo, ref, path))
        shutil.copy(HERE / "chunking_check.py", bundle / "chunking_check.py")
        (bundle / "ced_replay_stack.py").write_bytes(show(repo, ANCHOR, STACK))
        shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"{what} (vLLM {ref} {head}, {len(changed)} files over the release's "
                f"{BASE}) on candidate 6's image"),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-ced-anchor-{name}",
                "PYTHONPATH": f"/b:{ROOT}",
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
