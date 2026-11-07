#!/usr/bin/env python3
"""Exercise the public command with a separately executed numerical workload."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["cli"])
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
here = Path(__file__).parent
subprocess.run(
    [
        sys.executable,
        str(here / "run.py"),
        str(args.out / "stream"),
        "--ring-mib",
        "1",
        "--export",
        "--",
        str(here / "replay_controls.py"),
        "bounds",
        "--out",
        str(args.out),
    ],
    check=True,
)
result = json.loads((args.out / "result.json").read_text())
result["operation"] = "cli"
result["summary"] = json.loads((args.out / "stream/summary.json").read_text())
assert result["summary"]["complete"] and result["summary"]["written_records"] == 760
assert (args.out / "stream/trace.json").is_file()
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
