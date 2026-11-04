#!/usr/bin/env python3
"""Run the submission ownership regressions inside a disposable candidate image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "corrected"), required=True)
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    identity = json.loads((root / "base.json").read_text())
    patch = root / "vllm/0001-rollback-failed-submissions.patch"
    source = Path("/opt/spark-ds41f/candidate/vllm")
    target = source / "vllm/v1/engine/async_llm.py"
    test = source / "tests/v1/engine/test_request_submission_cleanup.py"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "started": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_only": True,
        "source_base": identity["source_base"],
        "patch_sha256": digest(patch),
        "original_async_llm_sha256": digest(target),
    }
    if report["patch_sha256"] != identity["source_patch_sha256"]:
        raise RuntimeError("patch identity mismatch")
    if report["original_async_llm_sha256"] != identity["original_async_llm_sha256"]:
        raise RuntimeError("candidate host source does not match the recorded base")
    # The runtime image deliberately has no Git client. The host generates
    # these files by applying the published patch to the verified image source;
    # check every resulting byte before installing into this disposable container.
    generated = root / "generated"
    corrected = generated / "async_llm.py"
    regression = generated / "test_request_submission_cleanup.py"
    if digest(corrected) != identity["fixed_async_llm_sha256"]:
        raise RuntimeError("generated correction identity mismatch")
    if digest(regression) != identity["test_sha256"]:
        raise RuntimeError("generated regression test identity mismatch")
    test.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(regression, test)
    if args.mode == "corrected":
        shutil.copyfile(corrected, target)
    expected = identity[
        "fixed_async_llm_sha256"
        if args.mode == "corrected"
        else "original_async_llm_sha256"
    ]
    report["tested_async_llm_sha256"] = digest(target)
    report["test_sha256"] = digest(test)
    if report["tested_async_llm_sha256"] != expected:
        raise RuntimeError("tested source identity mismatch")
    if report["test_sha256"] != identity["test_sha256"]:
        raise RuntimeError("regression test identity mismatch")
    environment = dict(os.environ)
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(source),
        VLLM_TARGET_DEVICE="cpu",
        CUDA_VISIBLE_DEVICES="-1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    junit = args.output / f"{args.mode}.xml"
    command = [
        sys.executable, "-m", "pytest", "--noconftest", "-c", "/dev/null",
        "-q", str(test), f"--junitxml={junit}",
    ]
    report["command"] = command
    print(json.dumps(report), flush=True)
    result = subprocess.run(
        command, cwd=source, env=environment, text=True, capture_output=True,
        timeout=180,
    )
    (args.output / f"{args.mode}.log").write_text(result.stdout + result.stderr)
    report["exit"] = result.returncode
    report["seconds"] = time.time() - report["started"]
    if junit.exists():
        suite = ET.parse(junit).getroot()
        suites = [suite] if suite.tag == "testsuite" else list(suite)
        report["counts"] = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    report["expected"] = {
        "tests": 10,
        "failures": 8 if args.mode == "control" else 0,
        "errors": 0,
        "skipped": 0,
    }
    report["expected_result_observed"] = (
        report.get("counts") == report["expected"]
        and result.returncode == (1 if args.mode == "control" else 0)
    )
    (args.output / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result.stdout + result.stderr, end="")
    print(json.dumps(report), flush=True)
    return 0 if report["expected_result_observed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
