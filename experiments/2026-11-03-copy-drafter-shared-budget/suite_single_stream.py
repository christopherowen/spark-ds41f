#!/usr/bin/env python3
"""servebench's single-stream suite on the booted arm, as a lab extra, so it runs on the
same boot as the copy probe before it and the census after it (#852's order).

usage (from the repository root): suite_single_stream.py HEAD_URL LABEL RUN

LABEL names the arm: this experiment's arm-LABEL.json, whose configuration the bench
checks the live engine against. One JSON line: the bench's exit status and output.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(label: str, run: str) -> int:
    config = (HERE / f"arm-{label}.json").relative_to(ROOT)
    out = f"results/private/suites/{run}-single-stream/{label}"
    argv = [str(ROOT / "bin/spark"), "--cluster-config", str(config), "bench", "--suite",
            "single-stream", "--compare", "none", "--label", label, "--output", out]
    code = subprocess.run(argv, cwd=ROOT).returncode
    print(json.dumps({"label": label, "suite": "single-stream", "exit": code, "output": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[2], sys.argv[3]))
