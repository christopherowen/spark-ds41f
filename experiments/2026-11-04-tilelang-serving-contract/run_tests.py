#!/usr/bin/env python3
"""Validate source ownership contracts inside a disposable CPU-only image."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "corrected"), required=True)
    args = parser.parse_args()
    inputs, output = Path("/input"), Path("/output")
    source = Path("/opt/spark-ds41f/candidate/vllm")
    base = json.loads((inputs / "base.json").read_text())
    report = {"mode": args.mode, "cpu_only": True, "started": time.time(),
              "python": sys.version, "files": {}, "expected_result_observed": False}
    output.mkdir(exist_ok=True)
    try:
        if sha(inputs / "cpu_declarations.py") != base["cpu_declarations_sha256"]:
            raise ValueError("CPU declaration harness identity mismatch")
        for name, expected in base["patches"].items():
            if sha(inputs / "vllm" / name) != expected:
                raise ValueError("patch identity mismatch: " + name)
        tests = []
        for name, record in base["files"].items():
            target, generated = source / name, inputs / "generated" / name
            if record["original_sha256"] is not None:
                if sha(target) != record["original_sha256"]:
                    raise ValueError("image source mismatch: " + name)
            if sha(generated) != record["corrected_sha256"]:
                raise ValueError("generated source mismatch: " + name)
            if record["test"] or args.mode == "corrected":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generated, target)
            expected = (record["corrected_sha256"] if record["test"] or
                        args.mode == "corrected" else record["original_sha256"])
            if sha(target) != expected:
                raise ValueError("tested source mismatch: " + name)
            report["files"][name] = sha(target)
            if record["test"]:
                tests.append(str(target))
        env = dict(os.environ)
        for name in base["retired_environment"]:
            env.pop(name, None)
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=f"{inputs}:{source}",
                   VLLM_TARGET_DEVICE="cpu", CUDA_VISIBLE_DEVICES="-1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        common = [sys.executable, "-m", "pytest", "--noconftest", "-c", "/dev/null",
                  "-p", "cpu_declarations", "-q"]
        collect_command = common + ["--collect-only"] + tests
        collect = subprocess.run(collect_command, cwd=source, env=env, text=True,
                                 capture_output=True, timeout=120)
        collection_log = collect.stdout + collect.stderr
        (output / "collection.txt").write_text(collection_log)
        report.update(collection_command=collect_command, collection_exit=collect.returncode)
        match = re.search(r"(\d+) tests? collected", collection_log)
        if collect.returncode != 0 or match is None:
            raise RuntimeError("CPU test collection failed; see collection.txt")
        collected = int(match.group(1))
        junit = output / "tests.xml"
        command = common + tests + [f"--junitxml={junit}"]
        report["command"] = command
        result = subprocess.run(command, cwd=source, env=env, text=True,
                                capture_output=True, timeout=180)
        (output / "tests.txt").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr, end="", flush=True)
        suites = ET.parse(junit).getroot()
        suites = [suites] if suites.tag == "testsuite" else list(suites)
        counts = {k: sum(int(s.get(k, 0)) for s in suites)
                  for k in ("tests", "failures", "errors", "skipped")}
        expected_failures = base["expected_control_failures"] if args.mode == "control" else 0
        report.update(exit=result.returncode, collected=collected, counts=counts,
                      expected={"tests": collected, "failures": expected_failures,
                                "errors": 0, "skipped": 0})
        report["expected_result_observed"] = (
            collected >= 50 and counts == report["expected"]
            and result.returncode == (1 if expected_failures else 0)
        )
    except Exception as error:
        report["error"] = str(error)
    finally:
        report["seconds"] = time.time() - report["started"]
        (output / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return 0 if report["expected_result_observed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
