#!/usr/bin/env python3
"""Candidate 18's vLLM series: candidate 17's, then the candidate-18 commits after it.

usage: make_series.py VLLM_REPO [SPARKNET_REPO]

Candidate 17's series (../rc17/vllm/series) ends at b8574bf22. Every commit
listed below is written to vllm/ as one patch, numbered from 0594 in
application order. vllm/series lists candidate 17's patches, their paths
relative to this folder, then these. Nothing that exists is overwritten.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc17/vllm/series"
BASE = "b8574bf22"  # vLLM candidate-17 head
HEAD = "28565956d"  # vLLM rc18-stack head
PATCHES = [
    ('2f9056378', 594),  # perf(moe): reduce peak memory and allocator slack during expert weight preparation (#714)
    ('f6b6458c8', 595),  # perf(moe,tilelang): compact 2-bit and 4-bit expert scale codes in one storage buffer (#727)
    ('1e65ff7f4', 596),  # perf(compressor): share transient capacity scratch buffers across compressor layers (#715)
    ('c8159d8fc', 597),  # feat(worker): log per-component GPU and host memory breakdown at startup (#716)
    ('4c5b5d62b', 598),  # perf(engram): share disk staged_rows buffer and trim transient hash/wkv buffers (#718)
    ('76bfcc733', 599),  # perf(sample): specialize step kernel cold-boot declarations to aligned logit strides (#719)
    ('a4a9efea1', 600),  # perf(loader,ds41): speed up direct loader shard routing, bounce copies, and expert lookup (#720)
    ('675e86203', 601),  # perf(config): avoid eager model/tokenizer/quant imports during ModelConfig init (#721)
    ('1dac35fd3', 602),  # perf(worker): eliminate redundant post-capture non-graph work in warmup_kernels (#722)
    ('2de9d09a5', 603),  # perf(cudagraph): warm up once per descriptor family in CudaGraphManager.capture (#723)
    ('b80915d92', 604),  # perf(tilelang): streamline _prepare_kernels registration, warm_declared, and bind_declared (#725)
    ('a1a330fe6', 605),  # perf(vocab,sample): add 128-row packed vocab head tile and fused greedy epilogue argmax (#729)
    ('b82ff3402', 606),  # perf(mhc): add single-launch project_and_finalize_streams and barrier-free finalize (#730)
    ('eff13bd95', 607),  # perf(gemm,linear): pack decode Q-B and indexer Q-B FP8 weights into 128x128 contiguous tiles (#731)
    ('22b85ad9c', 608),  # perf(distributed,deepseek_v4_1): route SP reduce_scatter, exchange, all_to_all_rows, and large all_reduce to sparknet.oneshot (#735)
    ('1d3bd6bbd', 609),  # perf(worker,headless): run single-rank headless worker inline and trim host heap (#717)
    ('a5ad1fc76', 610),  # perf(attention,model): trim activation and MXFP8 cast lifetimes across attention and decoder layer (#713)
    ('0d6c4f9ad', 611),  # perf(attention): overlap indexer query/weights_proj and global KV cache writes on aux_stream (#732)
    ('fd9cae574', 612),  # perf(dspark,sample): compute draft token confidence directly from TP-invariant local vocabulary shard (#733)
    ('ef39f5d43', 613),  # perf(deepseek_v4_1): remove torch.library.custom_op wrappers and compiler indirection (#736)
    ('28565956d', 614),  # [Spec] Eliminate CopyDrafter async-scheduling event sync stall, pad unused drafts with -1, and cap live drafts to pending (#687/#619)
]

SPARKNET_BASE = "ed6af32"
SPARKNET_PATCHES = [
    ('057aed4', 1),  # feat(oneshot): add RDMA exchange, reduce_scatter, all_to_all_rows, and sliced all_gather
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) not in (2, 3):
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
        "# Candidate 18: candidate 17's series (../../rc17/vllm/series), then vLLM",
        f"# candidate-18 from {BASE} to {HEAD} (../README.md, Candidate 18).",
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

    if len(sys.argv) == 3:
        srepo = sys.argv[2]
        sout = HERE / "sparknet"
        sout.mkdir(exist_ok=True)
        snames = []
        for commit, number in SPARKNET_PATCHES:
            written = git(srepo, "format-patch", "--zero-commit", "--no-signature", "--numbered",
                          "--start-number", str(number), "-1", commit, "-o", str(sout))
            snames.append(Path(written).name)
        sseries = sout / "series"
        sseries.write_text(
            "# Candidate 18 sparknet patches on top of v0.5.0 (ed6af327a895).\n"
            + "\n".join(snames) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
