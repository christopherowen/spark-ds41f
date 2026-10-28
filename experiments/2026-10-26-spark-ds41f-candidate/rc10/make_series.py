#!/usr/bin/env python3
"""Candidate 10's vLLM series: candidate 9's, then the candidate-10 commits after it.

usage: make_series.py VLLM_REPO

Candidate 9's series (../rc9/vllm/series) ends at e2b2b0b59. Every commit
from there to the candidate head is written to vllm/ as one patch, numbered from 0537 in
application order. vllm/series lists candidate 9's patches, their paths relative to this
folder, then these. Nothing that exists is overwritten.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc9/vllm/series"
BASE = "e2b2b0b59"  # candidate 9's head (tree 2b0931a0)
HEAD = "58938a074"  # vLLM mxfp8-block-scales
PATCHES = [
    ("58938a074", 537),  # MXFP8 GEMMs load the block-scale grid at SFB[row // 32]
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    order = git(repo, "rev-list", "--reverse", f"{BASE}..{HEAD}").split()
    if order != [git(repo, "rev-parse", commit) for commit, _ in PATCHES]:
        raise SystemExit(f"{BASE}..{HEAD} is not the listed commits in the listed order")
    out = HERE / "vllm"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(exist_ok=True)
    names = []
    for commit, number in PATCHES:
        # format-patch prints the path of the file it writes.
        path = Path(git(repo, "format-patch", "--zero-commit", "--no-signature", "--numbered",
                        "--start-number", str(number), "-1", commit, "-o", str(out)))
        if path.parent != out or not path.name.startswith(f"{number:04d}-"):
            raise SystemExit(f"{commit}: unexpected patch file {path}")
        names.append(path.name)
    lines = [
        "# Candidate 10: candidate 9's series (../../rc9/vllm/series), then vLLM mxfp8-block-scales",
        f"# from {BASE} to {HEAD} (../README.md, Candidate 10). Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        target = (PREVIOUS.parent / line.strip()).resolve()
        if not target.is_file():
            raise SystemExit(f"candidate 9's series names a missing patch: {line}")
        lines.append(os.path.relpath(target, out))
    lines += names
    (out / "series").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 2} patches, {len(names)} new: vllm/series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
