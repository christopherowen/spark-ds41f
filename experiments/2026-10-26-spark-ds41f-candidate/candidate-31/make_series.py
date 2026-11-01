#!/usr/bin/env python3
"""candidate-28's vLLM series: candidate-27's, then the 3 mHC / MoE / L2-prefetch commits (0634..0636).

usage: make_series.py VLLM_REPO

Candidate 27's series (../candidate-27/vllm/series) ends at 68ffa3c2b. Every commit
listed below is written to vllm/ as one patch, numbered from 0634 in
application order. vllm/series lists candidate-27's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "candidate-27/vllm/series"
BASE = "68ffa3c2b"  # vLLM candidate-27 head
HEAD = "78171359d"  # vLLM candidate-28 head
PATCHES = [
    ('3ed232ef6', 634),  # dsv41: optimize mHC finalize, project_streams, in-register aux collapse, and fused post_collapse_norm (#790 + #821/#835)
    ('afa03acc5', 635),  # Defer grouped_fp8_fp4_gemm barrier init past empty-CTA return and use 2 stages for decode down GEMM (#792)
    ('78171359d', 636),  # Configure GB10 persisting-L2 set-aside and clamp V4.1 prefetch window budgets (#766, flagless)
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
        "# candidate-28: candidate-27's series (../../candidate-27/vllm/series), then vLLM",
        f"# candidate-28 from {BASE} to {HEAD} (../README.md, candidate-28).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../candidate-27/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
