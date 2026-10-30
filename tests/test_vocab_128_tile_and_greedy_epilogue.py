"""Verify patch 0068 (128-row vocabulary LM-head tile and fused greedy epilogue argmax)."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT / "patches" / "vllm" / "0068-tilelang-vocab-128-tile-and-greedy-epilogue.patch"
)
EXP_PATCH = (
    ROOT
    / "experiments"
    / "2026-10-30-vocab-128-tile-and-greedy-epilogue"
    / "patches"
    / "vllm"
    / "0068-tilelang-vocab-128-tile-and-greedy-epilogue.patch"
)


def test_patch_applies_and_matches_experiment():
    assert PATCH.exists()
    assert EXP_PATCH.exists()
    assert PATCH.read_text() == EXP_PATCH.read_text()
    series = (ROOT / "patches" / "vllm" / "series").read_text().splitlines()
    assert "0068-tilelang-vocab-128-tile-and-greedy-epilogue.patch" in series
    text = PATCH.read_text()
    assert "ROW_TILES = (16, 32, 64, 128)" in text
    assert "def packed_vocab_greedy_partials(" in text
    assert "def finish_vocab_greedy_pairs(" in text
    assert "def project_head_greedy_pairs(" in text
    assert "def vocab_parallel_pairs_from_head(" in text
    vllm_dir = ROOT / ".work" / "upstreams" / "vllm"
    if vllm_dir.exists():
        subprocess.run(
            ["git", "apply", "--check", str(PATCH)],
            cwd=vllm_dir,
            check=True,
        )
