#!/usr/bin/env python3
"""Run the original single-stream workload between copy_probe and concurrent streams.

usage: suite_single_stream.py HEAD_URL LABEL RUN

The lab supplies HEAD_URL; bin/spark resolves and verifies the target from the arm.
Both boot labels use the same candidate-38 arm and retain separate output directories.
The wrapper returns the bench's exit status, including failures.
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARM_BY_LABEL = {
    "copy-m8-nodead": "arm-copy-m8-nodead.json",
    "copy-m8-nodead-repeat": "arm-copy-m8-nodead.json",
}


def main(label: str, run: str) -> int:
    config = (HERE / ARM_BY_LABEL[label]).relative_to(ROOT)
    out = f"results/private/suites/{run}-single-stream/{label}"
    argv = [str(ROOT / "bin/spark"), "--cluster-config", str(config), "bench", "--suite",
            "single-stream", "--compare", "none", "--label", label, "--output", out]
    code = subprocess.run(argv, cwd=ROOT).returncode
    print(json.dumps({"label": label, "config": str(config), "suite": "single-stream",
                      "exit": code, "output": out}), flush=True)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("head_url")
    parser.add_argument("label", choices=ARM_BY_LABEL)
    parser.add_argument("run")
    args = parser.parse_args()
    raise SystemExit(main(args.label, args.run))
