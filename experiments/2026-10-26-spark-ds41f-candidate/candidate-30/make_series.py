#!/usr/bin/env python3
"""Regenerate the candidate-30 vLLM patch series on top of candidate-28.

Candidate-30 adds:
- 0636..0639: #823 switched-fabric TP3..TP8 collectives & padding
- 0640: #781 batched 2D strided loader reads
- 0641: #799 + #813 switched Engram lookup/sharding and decode GEMM tiles
- 0642: #825 Component B fused shared-expert clamped SwiGLU + MXFP8 cast
- 0643: #831 Components A & C fused DSpark context join+cast, columns PDL chaining, and vec16 SWAR vocab decode
"""

from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE / "vllm"
CANDIDATE_28_SERIES = HERE.parent / "candidate-28" / "vllm" / "series"

PATCHES: list[tuple[int, str, str]] = [
    (636, "bfa7c17bf33216442b945421c94f2c794c7bdfc0", "0636-feat-dsv41-extend-TileLang-rank_order_sum-kernel-to-.patch"),
    (637, "e7345b1c7056a24bb12b37fcd8140370a6a2dec4", "0637-feat-dsv41-extend-TileLang-mHC-post_pre-and-GLP-stee.patch"),
    (638, "8ee8ab800d8ccf5c4ff3d78c8aa0ec9339996cd0", "0638-test-dsv41-verify-SP-row-exchange-and-all-to-all-acr.patch"),
    (639, "6a3ba5e471c431f4031bb8f13198606d977225c6", "0639-feat-dsv41-pad-moe_intermediate_size-and-handle-TP-p.patch"),
    (640, "cfd49344e89afa729082d8271032d0335a9a07f7", "0640-perf-loader-batch-strided-reads-into-column-padded-2.patch"),
    (641, "6015db794fb8d1ed50ce917cf4312a8f68660ed9", "0641-perf-dsv41-optimize-switched-Engram-lookup-sharding-.patch"),
    (642, "874e5b254d37da45bcc1cf36dd8838ed9829e9ee", "0642-perf-dsv41-fuse-shared-expert-clamped-SwiGLU-with-MX.patch"),
    (643, "776a2cb2098731f1c7829d5644e4902f6edd745b", "0643-perf-dsv41-fuse-DSpark-context-join-cast-chain-colum.patch"),
]

def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])

def format_commit_patch(repo: Path, number: int, total: int, commit: str) -> bytes:
    return git_bytes(
        repo, "format-patch", "--stdout", "-1", commit,
        f"--subject-prefix=PATCH {number:04d}/{total:04d}",
        "--zero-commit", "--no-signature",
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-repo", type=Path, default=Path("/home/swank/projects/spark-ds41f/.work/vllm-candidate-30"))
    args = parser.parse_args()
    repo = args.vllm_repo.resolve()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    c28_lines = [line.strip() for line in CANDIDATE_28_SERIES.read_text().splitlines() if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("0636-")]
    total = len(c28_lines) + len(PATCHES)
    series_lines = [f"../../candidate-28/vllm/{line}" if not line.startswith("../") else line for line in c28_lines]
    for number, commit, filename in PATCHES:
        (SERIES_DIR / filename).write_bytes(format_commit_patch(repo, number, total, commit))
        series_lines.append(filename)
    (SERIES_DIR / "series").write_text("\n".join(series_lines) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
