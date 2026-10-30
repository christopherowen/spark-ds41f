"""Tests for context-aware step-time budgeting on an absolute 2,048-token grid."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments/2026-10-30-context-aware-chunk-grid"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import context_grid_policy  # noqa: E402


class ContextAwareChunkGridTest(unittest.TestCase):
    def test_solo_prefill_uses_full_threshold_unchanged(self):
        for threshold in (16288, 8144, 6096):
            for context in (0, 16288, 32768, 131072, 900000):
                self.assertEqual(
                    context_grid_policy.quantize_chunk_threshold(
                        context,
                        0,
                        threshold=threshold,
                        step_target_s=1.0,
                        grid=2048,
                    ),
                    threshold,
                )
                self.assertEqual(
                    context_grid_policy.aligned_prefill_chunk(
                        context,
                        32768,
                        threshold=threshold,
                        num_runnable_decodes=0,
                        step_target_s=1.0,
                        grid=2048,
                    ),
                    threshold - context % threshold,
                )

    def test_low_and_deep_context_quantize_to_2048_grid_tiers(self):
        # Low context (c < 64K): T(c) = 8,192 across runnable decode counts 1..15.
        for n_dec in (1, 4, 8, 11, 15):
            for context in (0, 8192, 32768, 57344):
                tier = context_grid_policy.quantize_chunk_threshold(
                    context,
                    n_dec,
                    threshold=16288,
                    step_target_s=1.0,
                    grid=2048,
                )
                self.assertEqual(tier, 8192)
                chunk = context_grid_policy.aligned_prefill_chunk(
                    context,
                    32768,
                    threshold=16288,
                    num_runnable_decodes=n_dec,
                    step_target_s=1.0,
                    grid=2048,
                )
                self.assertEqual(chunk, 8192)

        # Deep context (c > 500K): T(c) = 4,096 across runnable decode counts 1..15.
        for n_dec in (1, 4, 8, 15):
            for context in (524288, 655360, 901120):
                tier = context_grid_policy.quantize_chunk_threshold(
                    context,
                    n_dec,
                    threshold=16288,
                    step_target_s=1.0,
                    grid=2048,
                )
                self.assertEqual(tier, 4096)
                chunk = context_grid_policy.aligned_prefill_chunk(
                    context,
                    32768,
                    threshold=16288,
                    num_runnable_decodes=n_dec,
                    step_target_s=1.0,
                    grid=2048,
                )
                self.assertEqual(chunk, 4096)

    def test_prefix_cache_hits_and_solo_to_mixed_transitions_snap_to_grid(self):
        # Prefix-cache hit at 1,536 tokens ends at 8,192 (6,656 tokens).
        chunk = context_grid_policy.aligned_prefill_chunk(
            1536,
            32768,
            threshold=16288,
            num_runnable_decodes=4,
            step_target_s=1.0,
            grid=2048,
        )
        self.assertEqual(chunk, 8192 - 1536)
        self.assertEqual((1536 + chunk) % 2048, 0)

        # Solo first chunk (16,288) followed by active-decode second chunk snaps to 22,528.
        mixed = context_grid_policy.aligned_prefill_chunk(
            16288,
            32768,
            threshold=16288,
            num_runnable_decodes=4,
            step_target_s=1.0,
            grid=2048,
        )
        self.assertEqual(mixed, 22528 - 16288)
        self.assertEqual((16288 + mixed) % 2048, 0)

    def test_experiment_results_and_arms(self):
        results = json.loads((EXP_DIR / "results.json").read_text())

        # Solo prefills are 100% unchanged.
        for shape_key, cases in results["solo_prefills"].items():
            for name, entry in cases.items():
                self.assertEqual(entry["ttft_delta_pct"], 0.0, f"{shape_key}/{name}")
                self.assertEqual(
                    entry["production_chunks"],
                    entry["context_grid_chunks"],
                    f"{shape_key}/{name}",
                )

        # Mixed workloads cap max single-step stall and p99 ITL.
        workloads = results["workloads"]
        for name in ("stall-15", "stall-11", "stall-7-tp3", "stall-3-tp3", "stall-1m"):
            prod = workloads[name]["production"]
            grid = workloads[name]["context_grid"]
            combo = workloads[name]["context_grid_interleave_k2"]
            self.assertLess(grid["max_chunk_step_s"], prod["max_chunk_step_s"], name)
            self.assertLess(combo["max_chunk_step_s"], prod["max_chunk_step_s"], name)
            self.assertLess(combo["stream_itl_p99_s"], prod["stream_itl_p99_s"], name)
            self.assertLess(combo["stream_itl_mean_s"], prod["stream_itl_mean_s"], name)
            self.assertGreater(
                combo["decode_tok_s_during_prefill"],
                prod["decode_tok_s_during_prefill"] * 2.0,
                name,
            )

        # Deep-context 900K stall-1m and soak cut max step stall by >50%.
        self.assertLess(
            workloads["stall-1m"]["context_grid_interleave_k2"]["max_chunk_step_s"],
            workloads["stall-1m"]["production"]["max_chunk_step_s"] * 0.5,
        )
        self.assertLess(
            workloads["soak"]["context_grid_interleave_k2"]["max_chunk_step_s"],
            workloads["soak"]["production"]["max_chunk_step_s"] * 0.5,
        )

        # Arm JSON files exist and carry --prefill-step-target-s and --prefill-chunk-grid.
        for filename in (
            "context-grid-tp4.json",
            "context-grid-interleave-k2-tp4.json",
            "context-grid-interleave-k2-tp3.json",
        ):
            payload = json.loads((EXP_DIR / filename).read_text())
            serve_args = payload["set"]["serve_args"]
            self.assertIn("--prefill-step-target-s", serve_args, filename)
            self.assertIn("--prefill-chunk-grid", serve_args, filename)

    def test_vllm_overlay_and_bundle_ast(self):
        overlay_files = (
            EXP_DIR / "overlay/vllm/config/scheduler.py",
            EXP_DIR / "overlay/vllm/engine/arg_utils.py",
            EXP_DIR / "overlay/vllm/v1/core/sched/scheduler.py",
            EXP_DIR
            / "bundles/context-grid-tests/tests/v1/core/test_aligned_prefill_chunks.py",
        )
        for path in overlay_files:
            self.assertTrue(path.is_file(), str(path))
            ast.parse(path.read_text(), filename=str(path))

        sched_src = (EXP_DIR / "overlay/vllm/v1/core/sched/scheduler.py").read_text()
        self.assertIn("def _context_aware_chunk_threshold(", sched_src)
        self.assertIn("prefill_step_target_s", sched_src)
        self.assertIn("prefill_chunk_grid", sched_src)


if __name__ == "__main__":
    unittest.main()
