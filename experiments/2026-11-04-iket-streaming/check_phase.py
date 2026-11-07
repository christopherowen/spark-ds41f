#!/usr/bin/env python3
"""Run the upstream phase pass's CPU cases without loading GPU dependencies."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--out", type=Path)
args = parser.parse_args()
path = args.source / "vllm/models/deepseek_v4_1/tilelang/iket.py"
spec = importlib.util.spec_from_file_location("checked_iket_phase", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
nodes = []
for node in ast.parse(
    (args.source / "tests/kernels/test_deepseek_v41_tilelang_iket.py").read_text()
).body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "GPU" for target in node.targets
    ):
        break
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        nodes.append(node)
namespace = {"iket": module, "pytest": pytest}
exec(
    compile(
        ast.Module(body=nodes, type_ignores=[]), "<upstream-cpu-phase-tests>", "exec"
    ),
    namespace,
)
passed = []
for name, function in namespace.copy().items():
    if name.startswith("test_"):
        function()
        passed.append(name)
receipt = {
    "passed": passed,
    "count": len(passed),
    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
