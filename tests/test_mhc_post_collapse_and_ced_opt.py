"""Verify patch 0045 (fused mhc.post_collapse and CED boundary gather deduplication)."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "vllm" / "0045-tilelang-mhc-post-collapse-and-ced-opt.patch"
EXP_PATCH = (
    ROOT
    / "experiments"
    / "2026-10-30-mhc-post-collapse-and-ced-opt"
    / "patches"
    / "vllm"
    / "0045-tilelang-mhc-post-collapse-and-ced-opt.patch"
)


def test_patch_applies_and_matches_experiment():
    assert PATCH.exists()
    assert EXP_PATCH.exists()
    assert PATCH.read_text() == EXP_PATCH.read_text()
    series = (ROOT / "patches" / "vllm" / "series").read_text().splitlines()
    assert "0045-tilelang-mhc-post-collapse-and-ced-opt.patch" in series
    text = PATCH.read_text()
    assert "def post_collapse_streams(" in text
    assert "def post_collapse(" in text
    assert "ced_positions=ced_positions" in text
    assert "ced_input_ids=ced_input_ids" in text
    vllm_dir = ROOT / ".work" / "upstreams" / "vllm"
    if vllm_dir.exists():
        subprocess.run(
            ["git", "apply", "--check", str(PATCH)],
            cwd=vllm_dir,
            check=True,
        )
