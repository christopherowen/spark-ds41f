#!/usr/bin/env python3
"""Candidate 10 reworks, one change per spec: kernel bundles, the bit gates, the lean suite.

usage: make_reworks.py CODED_ARRIVE_WORKTREE EXPERT_CODES_WORKTREE

CODED_ARRIVE_WORKTREE is a vLLM checkout whose HEAD is one commit over candidate 10's head
(58938a074); EXPERT_CODES_WORKTREE is one commit over that. Each commit's vllm/ files
become an overlay arm on candidate 10's image, its tests the bundle's mounted tests.

  coded-arrive     the grouped MoE GEMM's stage arrivals wait for the thread's cp.async
                   copies (cp.async.mbarrier.arrive.noinc), so the coded-scale expand
                   reads landed rows; against candidate 10.
  expert-codes-v2  2-bit scale codes for narrow experts, every thread expanding its own
                   rows as the 4-bit path does; against coded-arrive, its base.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = "experiments/2026-10-28-candidate-10-reworks"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/rc10"
LEAN = "experiments/2026-10-27-one-at-a-time/lean-suite.json"
HEAD = "58938a074"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-10"
SOURCE_ROOT = "/opt/spark-ds41f/candidate"
ROOT = f"{SOURCE_ROOT}/vllm"
FIXTURES = HERE.parent / "2026-10-03-packed-bf16-head/bundles/vllm-tests/ds41_fixtures.py"
ENV = {"HF_HUB_OFFLINE": "1", "VLLM_DS41_KERNEL_BACKEND": "tilelang"}
PYTEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "ds41_fixtures", "-rfE"]
MOE_TESTS = "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"
CODES_TESTS = "tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py"
GATES = [
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
    ["experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"], "sampled"],
]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def commit_files(repo: Path, parent: str) -> tuple[list[str], list[str]]:
    """(vllm/ files, tests/ files) HEAD changes or adds over ``parent``, its only parent."""
    if git(repo, "rev-parse", "HEAD~1").strip() != git(repo, "rev-parse", parent).strip():
        raise SystemExit(f"{repo}: HEAD is not one commit over {parent}")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{repo} has local changes")
    paths = []
    for line in filter(None, git(repo, "diff", "--name-status", "HEAD~1", "HEAD").splitlines()):
        status, path = line.split("\t", 1)
        if status not in ("M", "A"):
            raise SystemExit(f"{repo}: {line!r}: an overlay can only replace or add files")
        paths.append(path)
    return ([p for p in paths if p.startswith("vllm/") and p.endswith(".py")],
            [p for p in paths if p.startswith("tests/") and p.endswith(".py")])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def bundle(name: str, description: str, tests: list[str], repo: Path | None = None,
           sources: list[str] = (), mounted: list[str] = ()) -> str:
    """A kernel-lab bundle running ``tests`` on candidate 10's image, with ``sources``
    from ``repo`` as its overlay and ``mounted`` tests from ``repo`` over the image's."""
    target = HERE / "bundles" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy(FIXTURES, target / "ds41_fixtures.py")
    mounts = [["ds41_fixtures.py", "/b/ds41_fixtures.py"]]
    for path in sources:
        (target / "overlay" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / "overlay" / path)
    for path in mounted:
        local = "tests-" + path.removeprefix("tests/").replace("/", "-")
        shutil.copy(repo / path, target / local)
        mounts.append([local, f"{ROOT}/{path}"])
    write_json(target / "candidate.json", {
        "description": description, "image": IMAGE,
        "env": {**ENV, "PYTHONPATH": f"/b:{ROOT}:{SOURCE_ROOT}/b12x", "TILELANG_CACHE_DIR": f"/c/tilelang-{name}"},
        "source_root": SOURCE_ROOT, "workdir": "vllm", "argv": [*PYTEST, *tests], "verdict": "exit",
        "mounts": mounts,
    })
    return f"{FOLDER}/bundles/{name}"


def arm(label: str, repo: Path, files: list[str], notes: str, extends: str) -> None:
    """``label``'s overlay (``files`` from ``repo``) over the arm ``extends`` names, and its
    dev twin for the engine grid."""
    target = HERE / "overlay" / label
    if target.exists():
        shutil.rmtree(target)
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    if extends != f"{CANDIDATE}/rc-tp4.json":
        mounts = json.loads((HERE.parents[1] / extends).read_text())["set"]["container"]["mounts"]
        extends = f"{CANDIDATE}/rc-tp4.json"
    for path in files:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / path)
        mount = [f"{{checkout}}/{FOLDER}/overlay/{label}/{path}", f"{ROOT}/{path}", "ro"]
        mounts = [m for m in mounts if m[1] != mount[1]] + [mount]
    write_json(HERE / f"{label}-tp4.json", {
        "notes": f"Candidate 10 with {notes} (overlay/{label}, make_reworks.py).",
        "extends": {"config": extends},
        "set": {"container": {"mounts": mounts}},
    })
    write_json(HERE / f"{label}-tp4-dev.json", {
        "notes": f"{label}-tp4.json with the dev-mode collective RPC, for the engine grid.",
        "extends": {"config": f"{FOLDER}/{label}-tp4.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    })


def spec(label: str, base: str, base_label: str, kernels: list[dict]) -> None:
    """Kernel bundles, then ``base`` and the arm through the bit gates, then the lean suite
    on ``base``, the arm and ``base`` again."""
    dev = base.replace("-tp4.json", "-tp4-dev.json")
    write_json(HERE / f"{label}.json", {
        "experiment": FOLDER,
        "run": f"c10-{label}",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel", "bundles": kernels},
            {"kind": "measure", "bench": False, "bracket": False, "extras": GATES,
             "arms": [{"config": base, "label": base_label}, {"config": f"{label}-tp4.json", "label": label}]},
            {"kind": "measure", "suite": LEAN,
             "arms": [{"config": dev, "label": base_label},
                      {"config": f"{label}-tp4-dev.json", "label": label},
                      {"config": dev, "label": f"{base_label}-again"}]},
        ],
    })


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    arrive, codes = (Path(a).resolve() for a in sys.argv[1:])
    arrive_files, _ = commit_files(arrive, HEAD)
    codes_files, codes_tests = commit_files(codes, git(arrive, "rev-parse", "HEAD").strip())

    control = bundle("moe-c10", "Candidate 10 unchanged: the grouped MoE tests, among them the "
                     "coded-scale GEMM against the uncoded GEMM bit for bit.", [MOE_TESTS])
    fixed = bundle("coded-arrive", "Candidate 10 with the stage arrivals waiting for each thread's "
                   "copies: the same MoE tests.", [MOE_TESTS], arrive, arrive_files)
    arm("coded-arrive", arrive, arrive_files,
        "the grouped MoE GEMM's stage arrivals waiting for each thread's cp.async copies "
        "(cp.async.mbarrier.arrive.noinc), so the coded-scale expand reads rows that have landed",
        f"{CANDIDATE}/rc-tp4.json")
    spec("coded-arrive", f"{CANDIDATE}/rc-tp4.json", "candidate-10",
         [{"bundle": control, "node": "dgx1"}, {"bundle": fixed, "node": "dgx2"}])

    coded = bundle("expert-codes-v2", "coded-arrive with 2-bit scale codes for narrow experts, "
                   "every thread expanding its own rows: the MoE tests (the mixed 2-bit/4-bit "
                   "launch against the uncoded GEMM bit for bit) and the scale-code tests.",
                   [MOE_TESTS, CODES_TESTS], codes, sorted(set(arrive_files) | set(codes_files)),
                   codes_tests)
    arm("expert-codes-v2", codes, codes_files,
        "coded-arrive and 2-bit scale codes for experts whose block scales span at most 4 "
        "values, every thread expanding its own rows as the 4-bit path does",
        f"{FOLDER}/coded-arrive-tp4.json")
    spec("expert-codes-v2", f"{FOLDER}/coded-arrive-tp4.json", "coded-arrive",
         [{"bundle": coded, "node": "dgx3"}])
    print(f"coded-arrive: {len(arrive_files)} files; expert-codes-v2: {len(codes_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
