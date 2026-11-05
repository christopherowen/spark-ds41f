"""Validate this source-only packet without importing a compiler or GPU stack."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from spark_ds41f.build import patchset_sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PACKET / ".work/source")
    args = parser.parse_args()
    source = args.source.resolve()
    identity = json.loads((PACKET / "source-identity.json").read_text())
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(source), *argv], text=True, env=env).strip()

    assert git("rev-parse", "HEAD") == identity["final_head"]
    assert git("rev-parse", "HEAD^{tree}") == identity["final_tree"]
    assert not git("status", "--porcelain")
    assert (source / "VERSION").read_text().strip() == identity["version"]
    assert identity["version"] == "0.1.15+snapshot." + identity["correction_tree"]
    assert git("diff", "--name-only", identity["correction_head"], "HEAD") == "VERSION"
    expected = {
        "VERSION", "tilelang/jit/__init__.py",
        "testing/python/cache/test_tilelang_prepared_disabled_cache.py",
    }
    assert set(git("diff", "--name-only", identity["C42_prefix_head"], "HEAD").splitlines()) == expected
    parent = ROOT / "experiments/2026-11-05-compiler-live-qualification"
    series = [s for s in (PACKET / "tilelang/series").read_text().splitlines() if s and not s.startswith("#")]
    original = [s for s in (parent / "tilelang/series").read_text().splitlines() if s and not s.startswith("#")]
    assert series[:10] == original and len(series) == 12
    for name in original:
        assert (PACKET / "tilelang" / name).read_bytes() == (parent / "tilelang" / name).read_bytes()
    lock = json.loads((ROOT / "experiments/2026-10-26-spark-ds41f-candidate/candidate-42/upstreams.lock.json").read_text())["sources"]["tilelang"]
    lock["patch_series"] = str((PACKET / "tilelang/series").relative_to(ROOT))
    assert patchset_sha256(lock) == identity["patchset"]
    assets = {}

    def walk(value):
        if isinstance(value, dict):
            if "source_relative" in value and "sha256" in value:
                assets[value["source_relative"]] = value["sha256"]
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(json.loads((parent / "gpu-gate-plan.json").read_text()))
    for path, sha in assets.items():
        assert hashlib.sha256((source / path).read_bytes()).hexdigest() == sha, path
    changed = git("diff", "--name-only", identity["base_upstream"], "HEAD", "--", "*.py").splitlines()
    for path in changed:
        ast.parse((source / path).read_bytes(), filename=path, feature_version=(3, 12))
    subprocess.run(["git", "-C", str(source), "diff", "--check", identity["base_upstream"], "HEAD"], check=True, env=env)
    print(json.dumps({
        "head": identity["final_head"], "tree": identity["final_tree"],
        "patchset": identity["patchset"], "version": identity["version"],
        "original_patches_byte_identical": len(original),
        "original_canonical_assets_byte_identical": len(assets),
        "Python312_source_files_parsed": len(changed),
        "C42_delta_files": sorted(expected),
        "submodules_not_recursively_populated": True,
        "native_compiler_or_GPU_execution": False,
    }, indent=2))


if __name__ == "__main__":
    main()
