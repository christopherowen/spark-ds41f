#!/usr/bin/env python3
"""candidate-34's vLLM series: candidate-33's 0001..0663, then 0664..0667 (552 patches total).

usage: make_series.py VLLM_REPO

Candidate 33's series (../candidate-33/vllm/series) ends at 10bef730e. Every commit
listed below is written to vllm/ as one patch, numbered from 0664 in
application order. vllm/series lists candidate-33's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "candidate-33/vllm/series"
BASE = "f1f43b1e5"  # vLLM candidate-33 head
HEAD = "679ccdd19"  # vLLM candidate-34 head
PATCHES = [
    ('7106b0f50', 664),  # perf(cudagraph): cover odd c2 and full-depth speculative batches in dense_varlen_decode_shapes (#740)
    ('774918a36', 665),  # perf(dsv41): use 16-row MoE decode tiles below 1 tile/expert and 3-tile 2D prefill swizzle (#764, #765)
    ('f891d7a96', 666),  # perf(dsv41): run attention_front at prefill, share group metadata, trim TP3 rank-2 group, and use hybrid TP/SP fused_wqa_wkv (#744, #745, #776)
    ('679ccdd19', 667),  # perf(dsv41): fuse mhc.post into engram_post_gate and chain Engram gates on PDL (#833)
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
        "# candidate-34: candidate-33's series (../../candidate-33/vllm/series), then vLLM",
        f"# candidate-34 from {BASE} to {HEAD} (../README.md, candidate-34).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../candidate-33/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
