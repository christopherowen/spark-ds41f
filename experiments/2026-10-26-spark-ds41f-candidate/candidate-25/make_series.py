#!/usr/bin/env python3
"""candidate-25's vLLM series: candidate 24's, then the 3 TileLang metadata/sparse_mla/vocab commits (0625..0627).

usage: make_series.py VLLM_REPO

Candidate 24's series (../rc24/vllm/series) ends at 7d3263647. Every commit
listed below is written to vllm/ as one patch, numbered from 0625 in
application order. vllm/series lists candidate 24's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc24/vllm/series"
BASE = "7d3263647"  # vLLM candidate-24 head
HEAD = "e625eb146"  # vLLM candidate-25 head
PATCHES = [
    ('d95ec7d09', 625),  # dsv41: single-CTA group-skipping metadata.pages and cross-layer index page cache (#809)
    ('f1b47517a', 626),  # dsv41: preload Sinks, Positions, and CosSin before PDL wait in sparse_mla (#811)
    ('e625eb146', 627),  # dsv41: bind row_offsets bounds and predicate exception column in packed vocab (#812)
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
        "# candidate-25: candidate 24's series (../../rc24/vllm/series), then vLLM",
        f"# candidate-25 from {BASE} to {HEAD} (../README.md, candidate-25).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc24/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
