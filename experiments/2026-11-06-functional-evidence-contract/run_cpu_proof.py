#!/usr/bin/env python3
"""Counterfactual CPU proof; every cluster, hold and timing operation is mocked."""
import argparse
import contextlib
import datetime
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
import types
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f import lab

BASE = "2c4b23afcd834ba2adf6a15ca434e59fa4f49f74"
REQUIRED = {"kind": "summary", "label": "check", "require": {"identical": True}}
SAMPLED = {"kind": "sampled", "batch_invariance": "observe", "isolated_repeat": "require",
           "tokens": 192, "seed": 20261024, "cases": 8}


def sampled_output(repeat):
    scenarios = {"c8": {"differing": [0]}, "again": {"differing": [1] if repeat else []}}
    rows = [dict(scenario=name, **value, sha256=("b" if value["differing"] else "a") * 64)
            for name, value in scenarios.items()]
    rows.append({"summary": "seeded temperature-1 digest", "identical": False,
                 "tokens": 192, "seed": 20261024, "scenarios": scenarios,
                 "reference_sha256": "a" * 64})
    return "\n".join(json.dumps(row) for row in rows)


def trial(module, name, output, contract):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "collector.py").write_text("print(" + repr(output) + ")\n")
        gate = {"kind": "script", "argv": ["collector.py"], "out": "gate.jsonl",
                "label": "fixture", "telemetry": False, "evidence": contract}
        timing = {"kind": "cli", "argv": ["bench"], "label": "fixture"}
        with mock.patch.object(module, "ROOT", root), mock.patch.object(module, "plan", return_value=[gate, timing]), \
                mock.patch.object(module, "read_hold", return_value={"holder": "spark-lab: fixture"}), \
                mock.patch.object(module, "window_should_close", return_value=None), mock.patch.object(module, "beat"), \
                mock.patch.object(module, "save_engine_logs"), mock.patch.object(module, "window_close") as restore, \
                mock.patch.object(module, "spark_cli", return_value=0) as timed, \
                mock.patch.object(module, "Telemetry", return_value=contextlib.nullcontext()), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            code = module.execute({"run": "fixture"}, dry=False, keep_open=True)
        receipt = root / "gate.jsonl.evidence.json"
        return {"case": name, "run_exit": code, "timing_reached": timed.called,
                "restore_requested": restore.called, "collector_output": (root / "gate.jsonl").read_text(),
                "evidence": json.loads(receipt.read_text()) if receipt.exists() else None,
                "runner_log": log.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    old = types.ModuleType("lab_before")
    old.__file__ = str(ROOT / "spark_ds41f/lab.py")
    sys.modules[old.__name__] = old
    source = subprocess.check_output(["git", "show", args.base + ":spark_ds41f/lab.py"], cwd=ROOT)
    exec(compile(source, old.__file__, "exec"), old.__dict__)
    cases = [("required_false", '{"summary":"check","identical":false}', REQUIRED, False),
             ("required_missing", '{"summary":"check"}', REQUIRED, False),
             ("required_malformed", '{broken', REQUIRED, False),
             ("required_repeat_failed", sampled_output(True), SAMPLED, False),
             ("observed_batch_difference", sampled_output(False), SAMPLED, True),
             ("required_pass", '{"summary":"check","identical":true}', REQUIRED, True)]
    results, passed = [], True
    for name, output, contract, advances in cases:
        before, after = trial(old, name, output, contract), trial(lab, name, output, contract)
        matches = (before["run_exit"] == 0 and before["timing_reached"] and not before["restore_requested"]
                   and after["run_exit"] == (0 if advances else 1) and after["timing_reached"] == advances
                   and after["restore_requested"] != advances)
        passed &= matches
        results.append({"case": name, "expected_new_advancement": advances, "matches_expectation": matches,
                        "before": before, "after": after})
    receipt = {"base": args.base, "candidate": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "host": platform.node(), "python": platform.python_version(),
               "completed": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "scope": "CPU subprocesses with mocked cluster, window, telemetry and timing actions",
               "passed": passed, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: receipt[key] for key in ("base", "candidate", "passed", "seconds", "max_rss_kib")}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
