#!/usr/bin/env python3
"""Run CPU protocol checks against the exact prepared candidate source."""
import argparse
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3_ring4_test", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
_, _, lock = spark3.configuration(argparse.Namespace(
    cluster_config="experiments/2026-10-02-rocenante-mesh4/cluster.json"))
inputs = spark3.build_inputs(lock)
source = spark3.build_directory(inputs) / "src/b12x"
errors = spark3.project_problems(source, inputs["b12x"])
if errors:
    raise SystemExit("\n".join(errors))
subprocess.run([sys.executable, str(source / "tests/comm/roce_proxy_fake/run.py")],
               check=True, timeout=130)
