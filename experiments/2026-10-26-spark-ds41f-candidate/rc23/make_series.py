#!/usr/bin/env python3
"""Candidate 23's vLLM series: candidate 22's, then the 3 DSpark input/norm-cast & PDL commits (0619..0621).

usage: make_series.py VLLM_REPO

Candidate 22's series (../rc22/vllm/series) ends at 95280f597. Every commit
listed below is written to vllm/ as one patch, numbered from 0619 in
application order. vllm/series lists candidate 22's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc22/vllm/series"
BASE = "95280f597"  # vLLM candidate-22 head
HEAD = "1c77d4e4b"  # vLLM rc23 head
PATCHES = [
    ('1275bb8c6', 619),  # Consolidate multi-group DSpark input preparation into a single launch (#761)
    ('4f82d23ee', 620),  # DS4.1: Fuse DSpark context main_norm and MXFP8 per-token cast (#773)
    ('1c77d4e4b', 621),  # perf(dsv41,gemm): disable warp specialization and chain PDL in bf16_gemm_partials (#788)
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
        "# Candidate 23: candidate 22's series (../../rc22/vllm/series), then vLLM",
        f"# candidate-23 from {BASE} to {HEAD} (../README.md, Candidate 23).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc22/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
