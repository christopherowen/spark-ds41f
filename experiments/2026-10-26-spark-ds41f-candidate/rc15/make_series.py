#!/usr/bin/env python3
"""Candidate 15's vLLM series: candidate 14's, then the candidate-15 commits after it.

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
HEAD = "2945b1eb2"  # vLLM candidate-15
PATCHES = [
    ('e01a47375', 568),  # [Spec] Copy drafts: a verification lookahead apart from DSpark's block
    ('cdc125259', 569),  # [DS4.1] Decode rows for the verification lookahead
    ('424bc46c9', 570),  # [Spec] copy_match and copy_select (TileLang), the copy drafter
    ('2dcceafdb', 571),  # [V2] Copy drafts in the runner: wide verification and the copy state's report
    ('5eff21f21', 572),  # [Spec] Copy drafts benchmark: the copy kernels and verification at the lookahead
    ('6fb0ebac2', 573),  # [V2] Copy drafts: warmup steps run without adaptive verification
    ('520e73a3b', 574),  # [DS4.1] Copy drafts: the decode arena holds every request at the reorder threshold
    ('e28693599', 575),  # [V2] Copy drafts tests: the stand-ins carry the fields copy drafts added (num_verification_tokens, num_draft_steps, max_decode_rows, num_draft_tokens_per_req)
    ('437540ee4', 576),  # [V2] Copy drafts tests: the sink runner's batch carries its drafts a request, and the logprob test's verification takes the runner's arguments by name (the sink chunk is no longer last)
    ('3bc67d86f', 577),  # [V2] Display carveout test: expand one CPU row so loading does not fault 64 MiB on unified memory
    ('602db9d57', 578),  # [Direct IO] Bind contiguous weight arena and Order-13 buddy sponge
    ('4a1dfc177', 579),  # [MoE] Reuse static MoE output buffer across layers in forward context
    ('4d77e5b06', 580),  # [Worker] Flush allocator cache after CUDA graph capture and V2 kernel warmup
    ('c2571eb73', 581),  # [DS4.1] L2 prefetch: the owner and Engram layers' windows placed where their lines survive
    ('74e72c82e', 582),  # Route local SP rows and gather MXFP8 with MoE routes
    ('016a89326', 583),  # [Scheduler] Prefill-decode interleave: run K graphed decode steps between full prefill chunks
    ('b62468342', 584),  # [Scheduler] Context-aware prefill step-time budgeting on an absolute 2,048-token grid
    ('2c2afabfe', 585),  # Kernel preparation: omit unreachable MoE, router, and Markov head variants
    ('b1ecb91ab', 586),  # [DS4.1] Adaptive verification: unify seeded per-request budgeting, piecewise step cost, device dead-row clamp, and continuation-value cutoff
    ('ecea426f8', 587),  # [DS4.1] DSpark speculator: unify online IRLS confidence calibration, greedy shard runner-up margin, and concurrency-bounded draft depth
    ('4b192a3f8', 588),  # [DS4.1] Reuse q_cast in indexer wq_b and fuse RoPE into index query MXFP4 quantize (#706)
    ('b2ae69a75', 589),  # [DS4.1] Fuse global KV post-projection kernels in _prepare_global_kv (#707)
    ('139394805', 590),  # [DS4.1] DSpark: cast context main_x once and fuse kv_norm + rope + cache_writer (#708)
    ('6133b90ad', 591),  # [DS4.1] Fuse mHC post+collapse and deduplicate CED boundary position/ID gathers (#710)
    ('6e49b2128', 592),  # [DS4.1] Remove dead static SP/mega_moe branches and default multi-rank DSpark TP flags on (#711)
    ('c843dd18d', 593),  # Pre-build direct_io and engram_rows io_uring helpers into VLLM_NATIVE_CACHE_DIR
    ('2945b1eb2', 594),  # Tensor-parallelize the TileLang vision tower and aligner across TP ranks
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
        "# Candidate 15: candidate 14's series (../../rc14/vllm/series), then vLLM",
        f"# candidate-15 from {BASE} to {HEAD} (../README.md, Candidate 15).",
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
