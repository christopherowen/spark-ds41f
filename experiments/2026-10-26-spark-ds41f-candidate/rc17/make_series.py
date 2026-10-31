#!/usr/bin/env python3
"""Candidate 17's vLLM series: candidate 14's, then the candidate-17 commits after it.

usage: make_series.py VLLM_REPO

Candidate 14's series (../rc14/vllm/series) ends at d5df88d83. Every commit
listed below is written to vllm/ as one patch, numbered from 0568 in
application order. vllm/series lists candidate 14's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc14/vllm/series"
BASE = "d5df88d83"  # vLLM engram-drafted-prefill, candidate 14 (spark-ds41f:1030)
HEAD = "b8574bf22"  # vLLM copy-drafts-rework
PATCHES = [
    ('e88ab7137', 568),  # [V2] Display carveout test: expand one CPU row so loading does not fault 64 MiB on unified memory
    ('15a749af6', 569),  # [Direct IO] Bind contiguous weight arena and Order-13 buddy sponge
    ('8a8eef155', 570),  # [MoE] Reuse static MoE output buffer across layers in forward context
    ('49821ec24', 571),  # [Worker] Flush allocator cache after CUDA graph capture and V2 kernel warmup
    ('2d9d00205', 572),  # [DS4.1] L2 prefetch: the owner and Engram layers' windows placed where their lines survive
    ('8a78ab172', 573),  # Route local SP rows and gather MXFP8 with MoE routes
    ('7755079c6', 574),  # Kernel preparation: omit unreachable MoE, router, and Markov head variants
    ('6cd4cfd66', 575),  # [DS4.1] Clamp dead verification rows out of acceptance estimator observation
    ('407868693', 576),  # [DS4.1] Reuse q_cast in indexer wq_b and fuse RoPE into index query MXFP4 quantize (#706)
    ('e308d91c6', 577),  # [DS4.1] Fuse global KV post-projection kernels in _prepare_global_kv (#707)
    ('b6d8bdc4e', 578),  # [DS4.1] DSpark: cast context main_x once and fuse kv_norm + rope + cache_writer (#708)
    ('1c7f6c0b4', 579),  # [DS4.1] Fuse mHC post+collapse and deduplicate CED boundary position/ID gathers (#710)
    ('f4cfaa4dd', 580),  # [DS4.1] Remove dead static SP/mega_moe branches and default multi-rank DSpark TP flags on (#711)
    ('4162e7026', 581),  # Pre-build direct_io and engram_rows io_uring helpers into VLLM_NATIVE_CACHE_DIR
    ('7c79a276e', 582),  # Tensor-parallelize the TileLang vision tower and aligner across TP ranks
    ('4ad7b8a7d', 583),  # [Spec] Copy drafts: a verification lookahead apart from DSpark's block
    ('7f3fcf1a4', 584),  # [DS4.1] Decode rows for the verification lookahead
    ('3eb6663e9', 585),  # [Spec] copy_match and copy_select (TileLang), the copy drafter
    ('dd23505d4', 586),  # [V2] Copy drafts in the runner: wide verification and the copy state's report
    ('4729775f3', 587),  # [Spec] Copy drafts benchmark: the copy kernels and verification at the lookahead
    ('cb0862256', 588),  # [V2] Copy drafts: warmup steps run without adaptive verification
    ('a772de95b', 589),  # [DS4.1] Copy drafts: the decode arena holds every request at the reorder threshold
    ('9bc755df6', 590),  # [V2] Copy drafts tests: the stand-ins carry the fields copy drafts added
    ('f561a330c', 591),  # [V2] Copy drafts tests: the sink runner's batch carries its drafts a request
    ('8b53931d3', 592),  # [Spec] Rework copy drafts: async alloc before drafter, block-width non-copy kernels, slot draft limits, and warp-coalesced match/select
    ('b8574bf22', 593),  # Preserve initial unit confidence across mid-prefill chunks in adaptive verification
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
        "# Candidate 17: candidate 14's series (../../rc14/vllm/series), then vLLM",
        f"# candidate-17 from {BASE} to {HEAD} (../README.md, Candidate 17).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc14/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
