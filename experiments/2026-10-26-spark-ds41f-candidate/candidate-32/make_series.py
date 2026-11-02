#!/usr/bin/env python3
"""Regenerate the candidate-32 vLLM patch series on top of candidate-30.

Candidate-32 adds:
- 0644: #832 Components 1 & 2 fused compressor indexed KV write and indexer.wk split-K epilogue
- 0645: #767 column-shard target fused_wqa_wkv across TP
- 0646: #839 shard compressor.fused_wkv_wgate across TP ranks with single all-gather
- 0647: #816 decouple b12x from TileLang backend and port V2 worker Triton holdouts
- 0648: #787 break in-<think> reasoning loops via dynamic ThinkingBudgetState
"""

from __future__ import annotations
import argparse, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE / "vllm"
CANDIDATE_30_SERIES = HERE.parent / "candidate-30" / "vllm" / "series"

PATCHES: list[tuple[int, str, str]] = [
    (644, "02ffc34827ee8fd3b0a189914f517e7d39edbce6", "0644-perf-dsv41-fuse-compressor-indexed-KV-write-and-inde.patch"),
    (645, "0b0dcd696b5f98725224e79238685de9062eaa55", "0645-perf-dsv41-column-shard-target-fused_wqa_wkv-across-.patch"),
    (646, "0d06fda0ab564d8e8d1aab554a606334779bc63a", "0646-perf-dsv41-shard-compressor.fused_wkv_wgate-across-T.patch"),
    (647, "aaeb41bd35f0cfe7c7dfe022654a9b4aa60b8bc2", "0647-perf-v1-dsv41-decouple-b12x-from-TileLang-backend-an.patch"),
    (648, "7710e73390a9a3d9ea72db7ed5db2b5de8612329", "0648-feat-v1-break-in-think-reasoning-loops-via-dynamic-T.patch"),
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
    parser.add_argument("--vllm-repo", type=Path, default=Path("/home/swank/projects/spark-ds41f/.work/vllm-candidate-32"))
    args = parser.parse_args()
    repo = args.vllm_repo.resolve()
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    c30_lines = [line.strip() for line in CANDIDATE_30_SERIES.read_text().splitlines() if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("0644-")]
    total = len(c30_lines) + len(PATCHES)
    series_lines = [f"../../candidate-30/vllm/{line}" if not line.startswith("../") else line for line in c30_lines]
    for number, commit, filename in PATCHES:
        (SERIES_DIR / filename).write_bytes(format_commit_patch(repo, number, total, commit))
        series_lines.append(filename)
    (SERIES_DIR / "series").write_text("\n".join(series_lines) + "\n", encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
