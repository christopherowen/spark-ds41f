# SPDX-License-Identifier: Apache-2.0
"""Tests for fusing global KV post-projection kernels in _prepare_global_kv."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "patches" / "vllm" / "0045-tilelang-global-kv-post-fusion.patch"
RESULTS_PATH = (
    REPO_ROOT / "experiments" / "2026-10-30-global-kv-post-fusion" / "results.json"
)


class GlobalKvPostFusionTests(unittest.TestCase):
    def test_patch_is_listed_in_series_and_contains_fused_kernels(self) -> None:
        self.assertTrue(PATCH_PATH.is_file(), f"Missing patch: {PATCH_PATH}")
        series = (REPO_ROOT / "patches" / "vllm" / "series").read_text(encoding="utf-8")
        self.assertIn(PATCH_PATH.name, series.splitlines())
        patch_text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("def mxfp4_norm_rope_quantize_v41(", patch_text)
        self.assertIn("def norm_rope_write_index_keys(", patch_text)
        self.assertIn("rope_dim: int = 0", patch_text)
        self.assertIn(
            "self._write_index_keys(key, index_meta.slot_mapping[:rows], positions=positions)",
            patch_text,
        )
        self.assertIn("self.k_norm.declare_kernels = lambda: None", patch_text)

    def test_benchmark_results_confirm_bit_exact_and_speedup(self) -> None:
        self.assertTrue(RESULTS_PATH.is_file(), f"Missing results: {RESULTS_PATH}")
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["kv_source_layers"], 11)
        for ratio_key in ("ratio_1", "ratio_2"):
            for rows, entry in data["rows"][ratio_key].items():
                self.assertTrue(entry["bit_exact"], f"{ratio_key} rows={rows} not bit-exact")
                self.assertGreater(
                    entry["speedup"], 2.5, f"{ratio_key} rows={rows} speedup < 2.5x"
                )


if __name__ == "__main__":
    unittest.main()
