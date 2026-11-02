#!/usr/bin/env python3
"""servebench's single-stream suite on the booted arm, as a lab extra, so it runs on the
same boot as the copy probe before it and the census collection after it.

usage (from the repository root): suite_single_stream.py HEAD_URL LABEL RUN

#852's repro: copy_probe.py --prompts 3, then this suite, whose warm-up dies with
route_partials' assert on candidate-37 with copy drafts at m=8. LABEL names the arm
(ARMS below), whose configuration the bench checks the live engine against. One JSON
line: the bench's exit status and output directory.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPRO = "experiments/2026-11-01-dspark-default-copy-drafts/repro"
ARMS = {
    "c37-m8": f"{REPRO}/arm-c37-census-m8.json",
    "no0676-m8": f"{REPRO}/arm-c37-no0676-census-m8.json",
    "no0676-m16": f"{REPRO}/arm-c37-no0676-census-m16.json",
}


def main(label: str, run: str) -> int:
    out = f"results/private/suites/{run}-single-stream/{label}"
    argv = [str(ROOT / "bin/spark"), "--cluster-config", ARMS[label], "bench", "--suite",
            "single-stream", "--compare", "none", "--label", label, "--output", out]
    code = subprocess.run(argv, cwd=ROOT).returncode
    print(json.dumps({"label": label, "suite": "single-stream", "exit": code, "output": out}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[2], sys.argv[3]))
