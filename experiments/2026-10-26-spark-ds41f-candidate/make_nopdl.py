#!/usr/bin/env python3
"""Candidate 7 without PDL, for #589's serving A/B on the candidate's own image.

usage: make_nopdl.py VLLM_REPO

PDL (0520-0525, next-candidate 577091997..97e204cd9) changes 13 package files and adds
tilelang/pdl.py; no later commit of the series touches those 13 files, and nothing else
imports pdl. So candidate 7 without PDL is the image with the 13 files mounted at
577091997 (pdl.py stays, unused). Writes nopdl/overlay/ and the arms rc-tp4-nopdl.json and
rc-tp4-nopdl-dev.json; refuses when the commits no longer satisfy that.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = "experiments/2026-10-26-spark-ds41f-candidate"
BEFORE, AFTER, HEAD = "577091997", "97e204cd9", "c579b3fe1"
IMAGE_ROOT = "/opt/spark-ds41f/candidate/vllm"


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    changed = git(repo, "diff", "--name-status", BEFORE, AFTER, "--", "vllm").split("\n")
    changed = [line.split("\t") for line in changed if line]
    modified = [path for status, path in changed if status == "M"]
    added = [path for status, path in changed if status != "M"]
    if added != ["vllm/models/deepseek_v4_1/tilelang/pdl.py"] or len(modified) != 13:
        raise SystemExit(f"PDL's files are not 13 modified plus pdl.py: {changed}")
    later = set(git(repo, "diff", "--name-only", AFTER, HEAD, "--", "vllm").split())
    if later & set(modified):
        raise SystemExit(f"later commits touch PDL's files: {sorted(later & set(modified))}")
    # Imports of the DS4.1 module (FlashInfer's own enable_pdl predates it).
    pattern = r"(import pdl\b|from \.pdl\b|tilelang\.pdl\b|tilelang import .*\bpdl\b)"
    found = subprocess.run(["git", "-C", repo, "grep", "-l", "-P", pattern, HEAD, "--", "vllm"],
                           capture_output=True, text=True).stdout.split()
    importers = {path.split(":", 1)[1] for path in found} - set(modified) - set(added)
    if importers:
        raise SystemExit(f"other files name pdl: {sorted(importers)}")
    overlay = HERE / "nopdl/overlay"
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in modified:
        target = overlay / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(git(repo, "show", f"{BEFORE}:{path}"), encoding="utf-8")
        mounts.append([f"{{checkout}}/{FOLDER}/nopdl/overlay/{path}", f"{IMAGE_ROOT}/{path}", "ro"])
    arm = {
        "notes": ("Candidate 7 without PDL: rc-tp4.json with the 13 package files PDL (0520-0525) "
                  f"changes mounted at their version before it (vLLM {BEFORE}, nopdl/overlay, "
                  "make_nopdl.py); #589's serving A/B on the candidate's image."),
        "extends": {"config": f"{FOLDER}/rc-tp4.json"},
        "set": {"container": {"mounts": mounts}},
    }
    (HERE / "rc-tp4-nopdl.json").write_text(json.dumps(arm, indent=1) + "\n", encoding="utf-8")
    dev = {
        "notes": "Candidate 7 without PDL with the dev-mode collective RPC, for the screen's engine grid.",
        "extends": {"config": f"{FOLDER}/rc-tp4-nopdl.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    }
    (HERE / "rc-tp4-nopdl-dev.json").write_text(json.dumps(dev, indent=1) + "\n", encoding="utf-8")
    print(f"{len(modified)} files at {BEFORE}: nopdl/overlay; rc-tp4-nopdl.json, rc-tp4-nopdl-dev.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
