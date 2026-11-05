"""Resume only the unstarted checks after reconciling the interrupted v20 run."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

assert socket.gethostname() == "spark-0016"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path.home() / "spark-lab/tilelang-runtime-closure-20261105"
parser = argparse.ArgumentParser()
parser.add_argument("--finish", action="store_true")
args = parser.parse_args()
jobs = [
    ("draft-performance-v21", ["--operations", "draft_bench", "--family", "draft"]),
    ("tail-activity-v21", ["--operations", "activity", "--family", "all",
                           "--case-filter", str(HERE.relative_to(ROOT) / "tail-cases.json")]),
    ("final-iket-v21", ["--operations", "iket", "--family", "all"]),
    ("final-tests-v21", ["--operations", "tests", "--family", "all"]),
]
if args.finish:
    jobs = [
        ("draft-performance-v22", ["--operations", "draft_bench", "--family", "draft"]),
        ("final-iket-v21", ["--operations", "iket", "--family", "all"]),
        ("final-tests-v21", ["--operations", "tests", "--family", "all"]),
    ]
# A partial directory requires reconciliation; never overwrite or duplicate it.
assert not any((OUT / name).exists() for name, _ in jobs)
results = []
for name, args in jobs:
    code = subprocess.run([sys.executable, str(HERE / "run.py"), "--experiments",
                           str(HERE.relative_to(ROOT)), "--run", name, *args],
                          cwd=ROOT).returncode
    results.append(dict(run=name, exit=code))
    receipt = "recovery-v22.json" if args.finish else "recovery-v21.json"
    (OUT / receipt).write_text(json.dumps(results, indent=2) + "\n")
    print("RECOVERY", name, code, flush=True)
    if (code == 75 or not (OUT / name / "closure.json").exists()
            or (Path.home() / "spark-request.json").exists()
            or (Path.home() / "spark-hold.json").exists()):
        raise SystemExit(code or 75)
    rows = json.loads((OUT / name / "results.json").read_text())
    assert not any(row.get("memory_guard") for row in rows)
raise SystemExit(any(row["exit"] for row in results))
