"""Seed official build preparation from local Git objects and hashed wheels.

The lock and source manifest remain authoritative. Only committed Git content is
checked out; ignored files and local modifications are never copied.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import build


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def clone(source: Path, target: Path, revision: str, upstream: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(target)],
        check=True,
    )
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", revision], check=True)
    git(target, "remote", "rename", "origin", "upstream")
    git(target, "remote", "set-url", "upstream", upstream)


def submodules(source: Path, target: Path, expected: dict, prefix: str = "") -> None:
    modules = target / ".gitmodules"
    if not modules.exists():
        return
    git(target, "submodule", "init")
    result = subprocess.run(
        ["git", "-C", str(target), "config", "-f", str(modules), "--get-regexp", r"submodule\..*\.path"],
        text=True, stdout=subprocess.PIPE,
    )
    assert result.returncode in (0, 1), result.returncode
    for line in result.stdout.splitlines():
        key, path = line.split(maxsplit=1)
        relative = f"{prefix}{path}"
        revision = expected[relative]
        upstream = git(target, "config", "-f", str(modules), "--get", key[:-4] + "url")
        if not (target / path / ".git").exists():
            clone(source / path, target / path, revision, upstream)
        assert git(target / path, "rev-parse", "HEAD") == revision
        assert not git(target / path, "status", "--porcelain", "--untracked-files=all")
        submodules(source / path, target / path, expected, relative + "/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--variant", choices=("row-ladder", "shape-union"))
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    profile = experiment / "variants" / args.variant if args.variant else experiment
    upstreams = json.loads((profile / "upstreams.lock.json").read_text())
    inputs = build.build_inputs(upstreams)
    destination = build.build_directory(inputs)
    for name in build.build_projects(inputs):
        record = inputs[name]
        source = args.vllm.resolve() if name == "vllm" else args.cache.resolve() / "src" / name
        assert git(source, "rev-parse", record["patch_head"] + "^{tree}") == record["expected_tree"]
        target = destination / "src" / name
        if not target.exists():
            clone(source, target, record["patch_head"], record["upstream"])
            contribution = upstreams["sources"][name].get("contribution_remote")
            if contribution:
                git(target, "remote", "add", "origin", contribution)
        submodules(source, target, record.get("submodules", {}))
        problems = build.project_problems(target, record)
        assert not problems, problems
        print(f"{name}: verified {record['patch_head']}", flush=True)
    cutlass = destination / "src/cutlass"
    record = inputs["cutlass"]
    if not cutlass.exists():
        clone(args.cache.resolve() / "src/cutlass", cutlass, record["revision"], record["upstream"])
    assert git(cutlass, "rev-parse", "HEAD") == record["revision"]
    assert not git(cutlass, "status", "--porcelain", "--untracked-files=all")
    wheel_source = args.cache.resolve() / "context/cutlass-dsl-wheels"
    assert not build.wheel_problems(wheel_source)
    wheels = destination / "context/cutlass-dsl-wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    for wheel in wheel_source.glob("*.whl"):
        shutil.copy2(wheel, wheels / wheel.name)
    assert not build.wheel_problems(wheels)
    print(destination, flush=True)


if __name__ == "__main__":
    main()
