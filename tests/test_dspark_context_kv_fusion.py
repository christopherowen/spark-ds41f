# SPDX-License-Identifier: Apache-2.0
"""Tests for DSpark context KV fused front and shared main_x MXFP8 cast."""

from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPO_ROOT / "patches" / "vllm" / "0045-tilelang-dspark-context-kv-fusion.patch"
)
RESULTS_PATH = (
    REPO_ROOT / "experiments" / "2026-10-30-dspark-context-kv-fusion" / "results.json"
)


class DSparkContextKvFusionTests(unittest.TestCase):
    def test_patch_is_listed_in_series_and_contains_fusions(self) -> None:
        self.assertTrue(PATCH_PATH.is_file(), f"Missing patch: {PATCH_PATH}")
        series = (REPO_ROOT / "patches" / "vllm" / "series").read_text(encoding="utf-8")
        self.assertIn(PATCH_PATH.name, series.splitlines())
        patch_text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("def context_kv_front_kernel(", patch_text)
        self.assertIn("def _block32_split_mxfp8(", patch_text)
        self.assertIn("def insert_raw_context_kv(", patch_text)
        self.assertIn("projection.tilelang_mxfp8_input = True", patch_text)

    def test_benchmark_results_confirm_bit_exact_and_speedup(self) -> None:
        self.assertTrue(RESULTS_PATH.is_file(), f"Missing results: {RESULTS_PATH}")
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["num_draft_layers"], 3)
        for rows, entry in data["rows"].items():
            self.assertTrue(entry["bit_exact"], f"rows={rows} not bit-exact")
            self.assertGreater(entry["speedup"], 1.65, f"rows={rows} speedup < 1.65x")


if __name__ == "__main__":
    unittest.main()
