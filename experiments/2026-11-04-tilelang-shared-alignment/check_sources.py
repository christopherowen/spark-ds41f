"""Verify fresh candidate source identity and Python syntax without importing TileLang."""
import ast
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

lock = json.loads((HERE / "upstreams.lock.json").read_text())
inputs = spark.build_inputs(lock)
record = inputs["tilelang"]
source = spark.build_directory(inputs) / "src/tilelang"
problems = spark.project_problems(source, record)
if problems:
    raise SystemExit("; ".join(problems))
evidence = json.loads((HERE / "results.json").read_text())
assert evidence["tested_compiler_tree"] == record["expected_tree"]
changed = subprocess.check_output(["git", "diff", "--name-only", "--diff-filter=ACM", record["revision"], "HEAD", "--", "*.py"], cwd=source, text=True).splitlines()
for name in changed:
    ast.parse((source / name).read_text(), filename=name, feature_version=(3, 12))
print(f"Verified source head {record['patch_head']}, tree {record['expected_tree']}; {len(changed)} Python files parse for 3.12")
print("Tree equals prior paired standalone evidence; no compiler, GPU or serving tests were executed by this check")
