from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPO_ROOT
    / "patches"
    / "vllm"
    / "0068-tilelang-contiguous-fp8-decode-weights.patch"
)
EXPERIMENT_README = (
    REPO_ROOT
    / "experiments"
    / "2026-10-30-tile-contiguous-fp8-decode-weights"
    / "README.md"
)


def test_tile_contiguous_fp8_patch_exists_and_declares_symbols():
    assert PATCH_PATH.exists(), f"Missing patch: {PATCH_PATH}"
    content = PATCH_PATH.read_text(encoding="utf-8")

    for expected in (
        "TILE_WEIGHT_N = 128",
        "TILE_WEIGHT_K = 128",
        "def pack_weight_tiles(",
        "def unpack_weight_tiles(",
        "TILE_CONTIGUOUS_FP8",
        "tile_contiguous: bool = False",
        "def test_tile_contiguous_fp8_matches_row_major(",
    ):
        assert expected in content, f"Expected {expected!r} in {PATCH_PATH.name}"


def test_experiment_readme_documents_benchmarks_and_verification():
    assert EXPERIMENT_README.exists(), f"Missing README: {EXPERIMENT_README}"
    text = EXPERIMENT_README.read_text(encoding="utf-8")
    assert "pack_weight_tiles" in text
    assert "TILE_CONTIGUOUS_FP8" in text or "(8192, 1280)" in text
    assert "0 PDL violations" in text
