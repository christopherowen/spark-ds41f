#!/usr/bin/env python3
"""candidate-26's vLLM series: candidate-25's, then the 3 TileLang PDL prewait commits (0628..0630).

usage: make_series.py VLLM_REPO

Candidate 25's series (../candidate-25/vllm/series) ends at 65864cb57 (patch_head 0ec630907). Every commit
listed below is written to vllm/ as one patch, numbered from 0628 in
application order. vllm/series lists candidate-25's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "candidate-25/vllm/series"
BASE = "65864cb57"  # vLLM candidate-25 head
HEAD = "0ec630907"  # vLLM candidate-26 head
PATCHES = [
    ('acaac7baa', 628),  # dsv41: preload dispatch metadata before PDL wait in moe_swiglu_cast and FC2 (#803)
    ('3048e227b', 629),  # dsv41: preload static weights and step metadata before PDL wait in front/rope/indexer/mhc (#804)
    ('0ec630907', 630),  # dsv41: preload step metadata and norm weight before PDL wait in compressor (#808)
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
        "# candidate-26: candidate-25's series (../../candidate-25/vllm/series), then vLLM",
        f"# candidate-26 from {BASE} to {HEAD} (../README.md, candidate-26).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../candidate-25/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
