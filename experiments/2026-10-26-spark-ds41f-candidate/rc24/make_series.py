#!/usr/bin/env python3
"""Candidate 24's vLLM series: candidate 23's, then the 3 worker/sampler/spec-decode commits (0622..0624).

usage: make_series.py VLLM_REPO

Candidate 23's series (../rc23/vllm/series) ends at 1c77d4e4b. Every commit
listed below is written to vllm/ as one patch, numbered from 0622 in
application order. vllm/series lists candidate 23's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc23/vllm/series"
BASE = "1c77d4e4b"  # vLLM candidate-23 head
HEAD = "7d3263647"  # vLLM rc24 head
PATCHES = [
    ('353f40894', 622),  # refactor(v1,worker): move V2 GPU worker Triton reference kernels out of runtime modules (#770)
    ('c04ab0284', 623),  # V1 Sampler: Fuse temperature scaling into Gumbel-max sampling kernel (#775)
    ('7d3263647', 624),  # perf(v1,spec_decode): skip speculative drafting in steps where no request can use drafts (#786)
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
        "# Candidate 24: candidate 23's series (../../rc23/vllm/series), then vLLM",
        f"# candidate-24 from {BASE} to {HEAD} (../README.md, Candidate 24).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc23/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
