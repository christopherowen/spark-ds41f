#!/usr/bin/env python3
"""Candidate 22's vLLM series: candidate 21's, then the 3 DSpark draft-step commits (0616..0618).

usage: make_series.py VLLM_REPO

Candidate 21's series (../rc21/vllm/series) ends at 1f18be144. Every commit
listed below is written to vllm/ as one patch, numbered from 0616 in
application order. vllm/series lists candidate 21's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc21/vllm/series"
BASE = "6efba0add"  # vLLM candidate-21 head
HEAD = "95280f597"  # vLLM rc22 head
PATCHES = [
    ('efcfabdae', 616),  # perf(dsv41,glp): skip per-step mark_rows and eager glp_steer when no steered request is active (#783)
    ('270db89c9', 617),  # perf(dsv41,vocab): borrow project_head_greedy_pairs and vocab_parallel_pairs scratch buffers (#784)
    ('95280f597', 618),  # perf(worker): elide redundant decode-step logits gathers, H2D, and index expansions (#785)
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
        "# Candidate 22: candidate 21's series (../../rc21/vllm/series), then vLLM",
        f"# candidate-22 from {BASE} to {HEAD} (../README.md, Candidate 22).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc21/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
