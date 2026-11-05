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
options = parser.parse_args()
jobs = [
    ("draft-performance-v21", ["--operations", "draft_bench", "--family", "draft"]),
    ("tail-activity-v21", ["--operations", "activity", "--family", "all",
                           "--case-filter", str(HERE.relative_to(ROOT) / "tail-cases.json")]),
    ("final-iket-v21", ["--operations", "iket", "--family", "all"]),
    ("final-tests-v21", ["--operations", "tests", "--family", "all"]),
]
if options.finish:
    jobs = [
        ("draft-performance-v22", ["--operations", "draft_bench", "--family", "draft"]),
        ("final-iket-v21", ["--operations", "iket", "--family", "all"]),
        ("final-tests-v21", ["--operations", "tests", "--family", "all"]),
    ]
# Completed two-boot jobs may survive an interrupted outer controller.
# A partial or failed directory still requires explicit reconciliation.
pending = []
results = []
for name, job_args in jobs:
    path = OUT / name
    if path.exists():
        assert (path / "completed.json").exists() and (path / "closure.json").exists(), name
        rows = json.loads((path / "results.json").read_text())
        assert len(rows) == 2 and {r["job"]["boot"] for r in rows} == {1, 2}, name
        assert all(r["exit"] == 0 and not r["memory_guard"] for r in rows), name
        results.append(dict(run=name, exit=0, reused=True))
        print("REUSE completed two-boot job", name, flush=True)
    else:
        pending.append((name, job_args))
receipt = "recovery-v22.json" if options.finish else "recovery-v21.json"
for name, job_args in pending:
    code = subprocess.run([sys.executable, str(HERE / "run.py"), "--experiments",
                           str(HERE.relative_to(ROOT)), "--run", name, *job_args],
                          cwd=ROOT).returncode
    results.append(dict(run=name, exit=code))
    (OUT / receipt).write_text(json.dumps(results, indent=2) + "\n")
    print("RECOVERY", name, code, flush=True)
    if (code == 75 or not (OUT / name / "closure.json").exists()
            or (Path.home() / "spark-request.json").exists()
            or (Path.home() / "spark-hold.json").exists()):
        raise SystemExit(code or 75)
    rows = json.loads((OUT / name / "results.json").read_text())
    assert not any(row.get("memory_guard") for row in rows)
raise SystemExit(any(row["exit"] for row in results))
