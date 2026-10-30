# SPDX-License-Identifier: Apache-2.0
"""Tests for reusing q_cast in indexer.wq_b and fusing index-query RoPE + MXFP4 quantize."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPO_ROOT / "patches" / "vllm" / "0045-tilelang-indexer-query-qcast-rope-quantize.patch"
)
BUNDLE_DIR = (
    REPO_ROOT
    / "experiments"
    / "2026-10-30-indexer-query-qcast-rope-quantize"
    / "bundles"
    / "dgx4-local"
)


class IndexerQueryQCastRopeQuantizeTests(unittest.TestCase):
    def test_patch_is_listed_in_series_and_non_empty(self) -> None:
        self.assertTrue(PATCH_PATH.is_file(), f"Missing patch: {PATCH_PATH}")
        series = (REPO_ROOT / "patches" / "vllm" / "series").read_text(encoding="utf-8")
        self.assertIn(PATCH_PATH.name, series.splitlines())
        patch_text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("self.wq_b.tilelang_mxfp8_input = True", patch_text)
        self.assertIn("def slice_mxfp8(", patch_text)
        self.assertIn("rope_heads: int = 0", patch_text)

    def test_indexer_wqb_reads_q_cast_and_fuses_rope_in_attention_ast(self) -> None:
        attn_py = BUNDLE_DIR / "vllm" / "models" / "deepseek_v4_1" / "attention.py"
        source = attn_py.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn("self.wq_b.tilelang_mxfp8_input = True", source)
        self.assertIn("kernels.slice_mxfp8(q_cast, first, last)", source)
        self.assertIn("positions=positions[first:last]", source)
        self.assertNotIn('self._rope(iq, positions[first:last], "index_query")', source)
        # Ensure _INDEX_SPLIT_CHECK remains intact and receives q_cast too.
        self.assertIn("_INDEX_SPLIT_CHECK", source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_index_query":
                arg_names = [a.arg for a in node.args.args]
                self.assertIn("q_cast", arg_names)
                break
        else:
            self.fail("_index_query not found in attention.py")

    def test_mxfp4_quantize_v41_supports_fused_rope(self) -> None:
        indexer_py = (
            BUNDLE_DIR / "vllm" / "models" / "deepseek_v4_1" / "tilelang" / "indexer.py"
        )
        source = indexer_py.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("rope_heads: int = 0", source)
        self.assertIn("T.ieee_fmaf(first, cosine, -(second * sine))", source)
        self.assertIn("T.ieee_fmaf(first, sine, second * cosine)", source)


if __name__ == "__main__":
    unittest.main()
