"""Verify patch 0045 (DS4.1 dead optionality cleanup and multi-rank DSpark TP defaults)."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "vllm" / "0045-ds41-dead-optionality-cleanup.patch"
EXP_PATCH = (
    ROOT
    / "experiments"
    / "2026-10-30-ds41-dead-optionality-cleanup"
    / "patches"
    / "vllm"
    / "0045-ds41-dead-optionality-cleanup.patch"
)


def test_patch_applies_and_matches_experiment():
    assert PATCH.exists()
    assert EXP_PATCH.exists()
    assert PATCH.read_text() == EXP_PATCH.read_text()
    series = (ROOT / "patches" / "vllm" / "series").read_text().splitlines()
    assert "0045-ds41-dead-optionality-cleanup.patch" in series
    text = PATCH.read_text()
    assert '-def _use_sequence_parallel(vllm_config):' in text
    assert 'os.environ.get("SPARK_DS41F_DSPARK_VOCAB_PARALLEL", "1")' in text
    assert 'os.environ.get("SPARK_DS41F_DSPARK_MAIN_PROJ_TP", "1")' in text
    assert "SPARK_DS41F_INDEXER_SPLIT_CHECK" not in text
    vllm_dir = ROOT / ".work" / "upstreams" / "vllm"
    if vllm_dir.exists():
        subprocess.run(
            ["git", "apply", "--check", str(PATCH)],
            cwd=vllm_dir,
            check=True,
        )
