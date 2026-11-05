"""Run missing boots as individually reserved jobs; preserve completed evidence."""

import json
from pathlib import Path
import socket
import subprocess
import sys

assert socket.gethostname() == "spark-0016"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path.home() / "spark-lab/tilelang-runtime-closure-20261105"
mapping = json.loads((HERE / "accepted-runs.json").read_text())
jobs = [(mapping["drafting"]["2"], "draft_bench", "draft", 2)]
jobs += [(mapping["iket"][str(boot)], "iket", "all", boot) for boot in (1, 2)]
jobs += [(mapping["tests"][str(boot)], "tests", "all", boot) for boot in (1, 2)]
jobs += [(mapping["draft_extension"][str(boot)], "draft_bench", "draft", boot) for boot in (1, 2)]
results = []
for name, operation, family, boot in jobs:
    path = OUT / name
    if path.exists():
        assert (path / "completed.json").exists() and (path / "closure.json").exists(), name
        rows = json.loads((path / "results.json").read_text())
        assert len(rows) == 1 and rows[0]["job"]["boot"] == boot, name
        assert rows[0]["exit"] == 0 and not rows[0]["memory_guard"], name
        results.append(dict(run=name, boot=boot, exit=0, reused=True))
        print("REUSE", name, flush=True)
        continue
    command = [sys.executable, str(HERE / "run.py"), "--experiments",
                           str(HERE.relative_to(ROOT)), "--run", name,
                           "--operations", operation, "--family", family,
                            "--boots", str(boot)]
    if name in mapping["draft_extension"].values():
        command += ["--paired-rounds", "2048", "--case-filter",
                    str(HERE.relative_to(ROOT) / "draft-extension-cases.json")]
    if operation == "tests" and boot == 2:
        command += ["--cache-dir", str(OUT / mapping["tests"]["1"] / "tilelang-cache")]
    code = subprocess.run(command, cwd=ROOT).returncode
    results.append(dict(run=name, boot=boot, exit=code))
    (OUT / "recovery-v23.json").write_text(json.dumps(results, indent=2) + "\n")
    print("REMAINING", name, code, flush=True)
    if (code == 75 or not (path / "closure.json").exists()
            or (Path.home() / "spark-request.json").exists()
            or (Path.home() / "spark-hold.json").exists()):
        # An own request is consumed by run.py; a remaining request belongs to
        # another window. Yield between boots and reuse this job next time.
        raise SystemExit(code or 75)
    assert not any(r.get("memory_guard") for r in json.loads((path / "results.json").read_text()))
raise SystemExit(any(row["exit"] for row in results))
