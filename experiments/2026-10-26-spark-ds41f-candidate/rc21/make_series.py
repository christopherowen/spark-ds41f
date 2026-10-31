#!/usr/bin/env python3
"""Candidate 21's vLLM series: candidate 20's, then the 3 DSpark draft-step commits (0613..0615).

usage: make_series.py VLLM_REPO

Candidate 20's series (../rc20/vllm/series) ends at 1f18be144. Every commit
listed below is written to vllm/ as one patch, numbered from 0613 in
application order. vllm/series lists candidate 20's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc20/vllm/series"
BASE = "1f18be144"  # vLLM candidate-20 head
HEAD = "eb29b4794"  # vLLM rc21 head
PATCHES = [
    ('d56b97b2d', 613),  # perf(deepseek_v4_1): add TP4 and TP3 DSpark main_proj decode tile configs (#741)
    ('e282cce4b', 614),  # Slice anchor-first draft hidden states and use single-stage decode GEMM for Markov head (#756)
    ('eb29b4794', 615),  # Eliminate redundant host work during DSpark full CUDA graph replay (#763)
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
        written = git(repo, "format-patch", "--zero-commit", "--no-signature", "--numbered",
                      "--start-number", str(number), "-1", commit, "-o", str(out))
        names.append(Path(written).name)
    lines = [
        "# Candidate 21: candidate 20's series (../../rc20/vllm/series), then vLLM",
        f"# candidate-21 from {BASE} to {HEAD} (../README.md, Candidate 21).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc20/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
