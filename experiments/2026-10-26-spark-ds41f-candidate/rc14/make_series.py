#!/usr/bin/env python3
"""Candidate 14's vLLM series: candidate 13's, then the candidate-14 commits after it.

usage: make_series.py VLLM_REPO

Candidate 13's series (../rc13/vllm/series) ends at 13793229f. Every commit
listed below is written to vllm/ as one patch, numbered from 0567 in
application order. vllm/series lists candidate 13's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc13/vllm/series"
BASE = "13793229f"  # vLLM scale-codes-one-form, candidate 13 (tree e1a74af7)
HEAD = "d5df88d83"  # vLLM engram-drafted-prefill
PATCHES = [
    ("d5df88d83", 567),  # [DS4.1] Engram host hash: a prefilling request with drafts is left to the GPU
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
        "# Candidate 14: candidate 13's series (../../rc13/vllm/series), then vLLM",
        f"# engram-drafted-prefill from {BASE} to {HEAD} (../README.md, Candidate 14).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc13/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
