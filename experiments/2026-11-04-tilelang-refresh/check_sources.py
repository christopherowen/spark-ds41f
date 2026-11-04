"""Check the freshly prepared TileLang identity and changed Python source syntax.

This does not load TileLang's native library or execute its compiler or GPU tests.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark


def main() -> None:
    lock = spark.read_json("experiments/2026-11-04-tilelang-refresh/upstreams.lock.json")
    inputs = spark.build_inputs(lock)
    record = inputs["tilelang"]
    source = spark.build_directory(inputs) / "src/tilelang"
    problems = spark.project_problems(source, record)
    if problems:
        raise SystemExit("; ".join(problems))
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACM", record["revision"], "HEAD", "--", "*.py"],
        cwd=source,
        text=True,
    ).splitlines()
    for name in changed:
        ast.parse((source / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 12))
    print(f"TileLang identity verified: {record['patch_head']} (tree {record['expected_tree']})")
    print(f"Python 3.12 syntax: {len(changed)} changed files parsed; no compiler or GPU tests executed")


if __name__ == "__main__":
    main()
