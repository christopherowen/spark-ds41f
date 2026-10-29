#!/usr/bin/env python3
"""Candidate 13's vLLM series: candidate 12's, then the candidate-13 commits after it.

usage: make_series.py VLLM_REPO

Candidate 12's series (../rc12/vllm/series) ends at d984b2e66. Every commit
listed below is written to vllm/ as one patch, numbered from 0566 in
application order. vllm/series lists candidate 12's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc12/vllm/series"
BASE = "d984b2e66"  # vLLM candidate-12 (tree 4976b446)
HEAD = "13793229f"  # vLLM scale-codes-one-form
PATCHES = [
    ("13793229f", 566),  # [DS4.1] Expert scale codes: a narrow expert's 2-bit codes replace its 4-bit codes
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    listed = git(repo, "rev-list", "--reverse", f"{BASE}..{HEAD}").split()
    if [git(repo, "rev-parse", c) for c, _ in PATCHES] != listed:
        print(f"PATCHES is not {BASE}..{HEAD} in order", file=sys.stderr)
        return 1
    out = HERE / "vllm"
    out.mkdir(exist_ok=True)
    names = []
    for commit, number in PATCHES:
        written = git(repo, "format-patch", "--start-number", str(number), "-1", commit,
                      "-o", str(out))
        names.append(Path(written).name)
    lines = [
        "# Candidate 13: candidate 12's series (../../rc12/vllm/series), then vLLM",
        f"# scale-codes-one-form from {BASE} to {HEAD} (../README.md, Candidate 13).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc12/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
