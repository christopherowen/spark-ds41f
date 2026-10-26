#!/usr/bin/env python3
"""Candidate 8's vLLM series: candidate 7's, then next-candidate's commits after it.

usage: make_series.py VLLM_REPO

Candidate 7's series (../vllm/series) ends at next-candidate c579b3fe1. Every commit from
there to the candidate head is written to vllm/ as one patch, numbered from 0533 in
application order. vllm/series lists candidate 7's patches, their paths relative to this
folder, then these. Nothing that exists is overwritten.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "vllm/series"
BASE = "c579b3fe1"  # candidate 7's head (tree fbd941eb)
HEAD = "e846e64dc"
PATCHES = [
    ("e846e64dc", 533),  # a modular MoE kernel's output never lives in the workspace arena
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
        "# Candidate 8: candidate 7's series (../../vllm/series), then vLLM next-candidate",
        f"# from {BASE} to {HEAD} (../README.md, Candidate 8). Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        target = (PREVIOUS.parent / line.strip()).resolve()
        if not target.is_file():
            raise SystemExit(f"candidate 7's series names a missing patch: {line}")
        lines.append(os.path.relpath(target, out))
    lines += names
    (out / "series").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 2} patches, {len(names)} new: vllm/series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
