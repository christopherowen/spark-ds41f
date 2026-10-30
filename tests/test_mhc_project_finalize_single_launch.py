from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPO_ROOT
    / "patches"
    / "vllm"
    / "0068-tilelang-mhc-project-finalize-single-launch.patch"
)
EXPERIMENT_README = (
    REPO_ROOT
    / "experiments"
    / "2026-10-30-mhc-project-finalize-single-launch"
    / "README.md"
)


def test_mhc_project_finalize_patch_exists_and_declares_symbols():
    assert PATCH_PATH.exists(), f"Missing patch: {PATCH_PATH}"
    content = PATCH_PATH.read_text(encoding="utf-8")

    for expected in (
        "FINALIZE_FUSE_MAX_ROWS = 1",
        "def project_and_finalize_streams(",
        "def mhc_project_finalize(",
        "T.shfl_down(mix_val[0], MULT)",
        "T.shfl_down(mix_val[0], 2 * MULT)",
        '"mhc_project_finalize",',
        "def test_project_and_finalize_bits_match_split(",
        "def test_single_row_entry_launches_fused_kernel(",
    ):
        assert expected in content, f"Expected {expected!r} in {PATCH_PATH.name}"


def test_experiment_readme_documents_benchmarks_and_verification():
    assert EXPERIMENT_README.exists(), f"Missing README: {EXPERIMENT_README}"
    text = EXPERIMENT_README.read_text(encoding="utf-8")
    assert "project_and_finalize_streams" in text
    assert "FINALIZE_FUSE_MAX_ROWS" in text
    assert "0 violations" in text
