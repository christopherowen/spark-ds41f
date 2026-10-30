#!/usr/bin/env python3
"""Seeded adaptive verification on spark-ds41f-1030 (candidate 14, d5df88d83):
the vLLM patch, the kernel-lab test bundle for dgx4, and the overlay serving arms.

usage: make_bundles.py VLLM_WORKTREE
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FOLDER = "experiments/2026-10-30-seeded-adaptive-verification"
PREVIOUS_SERIES = REPO / "experiments/2026-10-26-spark-ds41f-candidate/rc14/vllm/series"
BASE_COMMIT = "d5df88d83"
IMAGE = "spark-ds41f:1030"
SOURCE_ROOT = "/opt/spark-ds41f/candidate"
ROOT = f"{SOURCE_ROOT}/vllm"
ENV = {"HF_HUB_OFFLINE": "1", "VLLM_DS41_KERNEL_BACKEND": "tilelang"}
PYTEST = [
    "-m",
    "pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "--noconftest",
    "-p",
    "prepare_fixtures",
    "-rfEs",
]
TESTS = [
    "tests/v1/spec_decode/test_adaptive_verification.py",
    "tests/v1/worker/test_dead_rows.py",
    "tests/v1/spec_decode/test_tilelang_rejection.py",
    "tests/v1/worker/test_gpu_tilelang_inputs.py",
    "tests/v1/worker/test_prepare_kernels.py",
]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def changed_files(repo: Path) -> tuple[list[str], list[str]]:
    """Return (vllm/ files, tests/ files) modified or added over BASE_COMMIT."""
    if (
        git(repo, "merge-base", BASE_COMMIT, "HEAD").strip()
        != git(repo, "rev-parse", BASE_COMMIT).strip()
    ):
        raise SystemExit(f"{repo}: HEAD is not over {BASE_COMMIT}")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{repo} has uncommitted local changes")
    paths = []
    for line in filter(
        None, git(repo, "diff", "--name-status", BASE_COMMIT, "HEAD").splitlines()
    ):
        status, path = line.split("\t", 1)
        if status not in ("M", "A"):
            raise SystemExit(
                f"{repo}: {line!r}: an overlay can only replace or add files"
            )
        paths.append(path)
    return (
        [p for p in paths if p.startswith("vllm/") and p.endswith(".py")],
        [p for p in paths if p.startswith("tests/") and p.endswith(".py")],
    )


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def write_series(repo: Path, head: str) -> None:
    out = HERE / "vllm"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    written = git(
        repo,
        "format-patch",
        "--start-number",
        "568",
        f"{BASE_COMMIT}..HEAD",
        "-o",
        str(out),
    ).splitlines()
    names = [Path(p.strip()).name for p in written if p.strip()]
    lines = [
        "# Seeded adaptive verification: candidate 14's series",
        f"# (../../2026-10-26-spark-ds41f-candidate/rc14/vllm/series), then {BASE_COMMIT}..{head}.",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS_SERIES.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("../../"):
            lines.append(
                line.replace(
                    "../../", "../../2026-10-26-spark-ds41f-candidate/", 1
                )
            )
        else:
            lines.append(f"../../2026-10-26-spark-ds41f-candidate/rc14/vllm/{line}")
    lines += names
    (out / "series").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_bundle(
    name: str,
    description: str,
    repo: Path,
    sources: list[str],
    tests: list[str],
) -> str:
    target = HERE / "bundles" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy(HERE / "prepare_fixtures.py", target / "prepare_fixtures.py")
    mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"]]
    for path in sources:
        (target / "overlay" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / "overlay" / path)
        mounts.append([f"overlay/{path}", f"{ROOT}/{path}"])
    for path in tests:
        local = "tests-" + path.removeprefix("tests/").replace("/", "-")
        shutil.copy(repo / path, target / local)
        mounts.append([local, f"{ROOT}/{path}"])
    write_json(
        target / "candidate.json",
        {
            "description": description,
            "image": IMAGE,
            "env": {
                **ENV,
                "PYTHONPATH": f"/b:{ROOT}:{SOURCE_ROOT}/b12x",
                "TILELANG_CACHE_DIR": f"/c/tilelang-{name}",
            },
            "source_root": SOURCE_ROOT,
            "workdir": "vllm",
            "argv": [*PYTEST, "--basetemp", f"/c/pytest-{name}", *TESTS],
            "verdict": "exit",
            "mounts": mounts,
        },
    )
    return f"{FOLDER}/bundles/{name}"


def make_overlay_arms(repo: Path, sources: list[str], head: str) -> None:
    target = HERE / "overlay"
    if target.exists():
        shutil.rmtree(target)
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in sources:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / path)
        mounts.append([f"{{checkout}}/{FOLDER}/overlay/{path}", f"{ROOT}/{path}", "ro"])
    for topo, base in (
        ("tp4", "config/production-tp4.json"),
        ("tp3", "config/production-tp3.json"),
    ):
        write_json(
            HERE / f"seeded-{topo}.json",
            {
                "notes": (
                    f"spark-ds41f-1030 with per-request adaptive verification budgets "
                    f"for seeded sampled requests (vLLM {head}; make_bundles.py)."
                ),
                "extends": {"config": base},
                "set": {"container": {"mounts": mounts}},
            },
        )
        write_json(
            HERE / f"seeded-{topo}-dev.json",
            {
                "notes": f"seeded-{topo}.json with the dev-mode collective RPC.",
                "extends": {"config": f"{FOLDER}/seeded-{topo}.json"},
                "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
            },
        )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    sources, tests = changed_files(repo)
    head = git(repo, "rev-parse", "--short=9", "HEAD").strip()
    write_series(repo, head)
    make_bundle(
        "seeded-adaptive",
        f"spark-ds41f-1030 with per-request seeded adaptive verification ({head}): "
        "adaptive verification, dead rows, TileLang rejection, input preparation, "
        "and kernel preparation tests.",
        repo,
        sources,
        tests,
    )
    make_overlay_arms(repo, sources, head)
    print(f"seeded-adaptive: {len(sources)} vllm files, {len(tests)} test files at {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
