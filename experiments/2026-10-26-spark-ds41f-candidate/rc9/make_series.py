#!/usr/bin/env python3
"""Candidate 9's vLLM series: candidate 8's, then the candidate-9 commits after it.

usage: make_series.py VLLM_REPO

Candidate 8's series (../rc8/vllm/series) ends at next-candidate e846e64dc. Every commit
from there to the candidate head is written to vllm/ as one patch, numbered from 0534 in
application order. vllm/series lists candidate 8's patches, their paths relative to this
folder, then these. Nothing that exists is overwritten.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc8/vllm/series"
BASE = "e846e64dc"  # candidate 8's head (tree e63ab9e9)
HEAD = "e2b2b0b59"  # vLLM candidate-9-fixes
PATCHES = [
    ("31312f358", 534),  # tests: each test restores the root logging levels
    ("1fca420fa", 535),  # MoE: a lease prepared for the caller's output holds no output buffer
    ("e2b2b0b59", 536),  # entrypoints: the SageMaker routes' import keeps the root logging
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
        "# Candidate 9: candidate 8's series (../../rc8/vllm/series), then vLLM candidate-9-fixes",
        f"# from {BASE} to {HEAD} (../README.md, Candidate 9). Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        target = (PREVIOUS.parent / line.strip()).resolve()
        if not target.is_file():
            raise SystemExit(f"candidate 8's series names a missing patch: {line}")
        lines.append(os.path.relpath(target, out))
    lines += names
    (out / "series").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 2} patches, {len(names)} new: vllm/series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
