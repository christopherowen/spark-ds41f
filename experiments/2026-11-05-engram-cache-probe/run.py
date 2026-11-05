#!/usr/bin/env python3
"""Repeat the native C planner on synthetic row IDs; never access a GPU or NVMe."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("probe.c").resolve()
    report = {"scope": "synthetic native planner only; no NVMe, CUDA, serving or boot measurement",
              "platform": platform.platform(), "python": sys.version,
              "started_unix": time.time(), "probe_sha256": sha(source),
              "runner_sha256": sha(__file__), "commands": [], "runs": []}

    def command(argv):
        started = time.time()
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=180)
        report["commands"].append({"argv": list(map(str, argv)), "exit": result.returncode,
                                   "elapsed_s": time.time() - started,
                                   "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"command failed: {argv}")
        return result.stdout

    try:
        report["compiler"] = command([args.cc, "--version"])
        for label, include in (("base", args.base), ("candidate", args.candidate)):
            report[label + "_header_sha256"] = sha(include / "_row_plan.h")
            flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"] if args.sanitize else ["-O3"]
            command([args.cc, *flags, "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-I", include.resolve(),
                     source, "-o", (args.out / label).resolve()])
        cases = [(16, 75, True), (64, 100, True), (65, 100, True),
                 (96, 100, True), (96, 75, True), (96, 0, True),
                 (128, 75, True), (129, 100, True), (512, 75, True), (6144, 75, True),
                 (16384, 75, True), (16384, 0, True), (16384, 0, False)]
        for tp in (3, 4):
            for tokens, hot, populated in cases:
                pair = {}
                for label in ("base", "candidate"):
                    result = json.loads(command([(args.out / label).resolve(), tokens, tp, hot,
                                                 int(populated), 2 if args.sanitize else 11]))
                    pair[label] = result
                assert pair["base"]["ids_digest_fnv64"] == pair["candidate"]["ids_digest_fnv64"]
                report["runs"].append(pair)
    finally:
        report["finished_unix"] = time.time()
        (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(report["runs"]), "receipt": str(args.out / "receipt.json")}))


if __name__ == "__main__":
    main()
