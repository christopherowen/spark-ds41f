#!/usr/bin/env python3
"""Candidate 20's vLLM series: candidate 17's, then the 19 retained candidate-18 commits (0594..0612).

usage: make_series.py VLLM_REPO

Candidate 17's series (../rc17/vllm/series) ends at 37a0a07bb. Every commit
listed below is written to vllm/ as one patch, numbered from 0594 in
application order. vllm/series lists candidate 17's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc17/vllm/series"
BASE = "37a0a07bb"  # vLLM candidate-17 head
HEAD = "1f18be144"  # vLLM rc20-exact head
PATCHES = [
    ('ae2e49bc1', 594),  # perf(moe): reduce peak memory and allocator slack during expert weight preparation (#714)
    ('984bf3a3a', 595),  # perf(moe,tilelang): compact 2-bit and 4-bit expert scale codes in one storage buffer (#727)
    ('c08a190dd', 596),  # perf(compressor): share transient capacity scratch buffers across compressor layers (#715)
    ('2312ca8f7', 597),  # feat(worker): log per-component GPU and host memory breakdown at startup (#716)
    ('429b21ba3', 598),  # perf(engram): share disk staged_rows buffer and trim transient hash/wkv buffers (#718)
    ('156d0939b', 599),  # perf(sample): specialize step kernel cold-boot declarations to aligned logit strides (#719)
    ('e6183b5e3', 600),  # perf(loader,ds41): speed up direct loader shard routing, bounce copies, and expert lookup (#720)
    ('8d8a42418', 601),  # perf(config): avoid eager model/tokenizer/quant imports during ModelConfig init (#721)
    ('2790c85c7', 602),  # perf(worker): eliminate redundant post-capture non-graph work in warmup_kernels (#722)
    ('fe06cad12', 603),  # perf(cudagraph): warm up once per descriptor family in CudaGraphManager.capture (#723)
    ('3d4ec3971', 604),  # perf(tilelang): streamline _prepare_kernels registration, warm_declared, and bind_declared (#725)
    ('f63f56451', 605),  # perf(vocab,sample): add 128-row packed vocab head tile and fused greedy epilogue argmax (#729)
    ('871172001', 606),  # perf(mhc): add single-launch project_and_finalize_streams and barrier-free finalize (#730)
    ('c8eb05b4f', 607),  # perf(gemm,linear): pack decode Q-B and indexer Q-B FP8 weights into 128x128 contiguous tiles (#731)
    ('a31174812', 608),  # perf(worker,headless): run single-rank headless worker inline and trim host heap (#717)
    ('f72d89acd', 609),  # perf(attention,model): trim activation and MXFP8 cast lifetimes across attention and decoder layer (#713)
    ('58fa0bb20', 610),  # perf(dspark,sample): compute draft token confidence directly from TP-invariant local vocabulary shard (#733)
    ('f4bdc28ae', 611),  # perf(deepseek_v4_1): remove torch.library.custom_op wrappers and compiler indirection (#736)
    ('1f18be144', 612),  # Eliminate CopyDrafter async-scheduling event sync stall, pad unused drafts with -1, and cap live drafts to pending (#687/#619)
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
        "# Candidate 20: candidate 17's series (../../rc17/vllm/series), then vLLM",
        f"# candidate-20 from {BASE} to {HEAD} (../README.md, Candidate 20).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../rc17/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
