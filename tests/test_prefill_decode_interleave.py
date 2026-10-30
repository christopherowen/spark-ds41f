"""Unit tests for K-step graphed decode interleaving between full prefill chunks."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "2026-10-30-prefill-decode-interleave"
STALLS_DIR = ROOT / "experiments" / "2026-10-26-concurrency-stalls"
if str(STALLS_DIR) not in sys.path:
    sys.path.insert(0, str(STALLS_DIR))

import stallsim  # noqa: E402
from stallsim import Engine, Lineage, Policy, Shape  # noqa: E402


def _load_interleave_policy():
    spec = importlib.util.spec_from_file_location(
        "interleave_policy", EXP_DIR / "interleave_policy.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["interleave_policy"] = module
    spec.loader.exec_module(module)
    return module


class PrefillDecodeInterleaveTest(unittest.TestCase):
    def test_overlay_and_bundle_files_exist_and_parse(self):
        rel_paths = [
            "vllm/config/scheduler.py",
            "vllm/engine/arg_utils.py",
            "vllm/v1/core/sched/scheduler.py",
        ]
        for rel in rel_paths:
            overlay_file = EXP_DIR / "overlay" / rel
            bundle_file = EXP_DIR / "bundles" / "interleave-tests" / "overlay" / rel
            self.assertTrue(overlay_file.is_file(), f"Missing {overlay_file}")
            self.assertTrue(bundle_file.is_file(), f"Missing {bundle_file}")
            source = overlay_file.read_text()
            self.assertEqual(source, bundle_file.read_text())
            ast.parse(source)

        test_bundle_file = (
            EXP_DIR
            / "bundles"
            / "interleave-tests"
            / "tests_v1_core_test_prefill_yield.py"
        )
        self.assertTrue(test_bundle_file.is_file())
        test_src = test_bundle_file.read_text()
        ast.parse(test_src)
        self.assertIn(
            "test_interleave_k_graphed_decode_steps_between_full_prefill_chunks",
            test_src,
        )
        self.assertIn(
            "test_interleave_works_at_full_slot_occupancy_stall_15",
            test_src,
        )
        self.assertIn(
            "test_interleave_admits_waiting_short_prefill_during_decode_step",
            test_src,
        )

    def test_vllm_scheduler_overlay_implements_interleave_and_short_admission(self):
        sched_cfg_src = (
            EXP_DIR / "overlay" / "vllm" / "config" / "scheduler.py"
        ).read_text()
        self.assertIn("prefill_decode_interleave_steps", sched_cfg_src)

        arg_utils_src = (
            EXP_DIR / "overlay" / "vllm" / "engine" / "arg_utils.py"
        ).read_text()
        self.assertIn("--prefill-decode-interleave-steps", arg_utils_src)
        self.assertIn("prefill_decode_interleave_steps", arg_utils_src)

        sched_src = (
            EXP_DIR / "overlay" / "vllm" / "v1" / "core" / "sched" / "scheduler.py"
        ).read_text()
        self.assertIn("self.prefill_decode_interleave_steps", sched_src)
        self.assertIn("self._interleave_decode_steps_remaining", sched_src)
        self.assertIn("has_inflight_multi_chunk_prefill", sched_src)
        self.assertIn("interleave_decode_step", sched_src)
        self.assertIn("interleave_admit_short", sched_src)

    def test_stallsim_interleaves_k_pure_decode_steps_and_keeps_chunk_boundaries(self):
        shape = Shape(
            name="toy",
            budget=1024,
            max_seqs=4,
            pool_blocks=10_000,
            retention_interval=512,
            fill_blocks=0,
            dec_knots=[[1, 0.010], [4, 0.020]],
            F=0.050,
            m0=50e-6,
            m1=0.0,
            short_e=0.002,
            short_s=0.0005,
            alpha=1.0,
            fe0=0.0,
            fe1=0.0,
            engram_first_s=0.0,
            host_s=0.0,
            accept_knots=[[1, 2.0], [4, 2.0]],
        )
        # 3 active decode streams + 1 long prefill = 4/4 sequence slots (full occupancy).
        engine = Engine(shape, Policy(interleave_decode_steps=2), seed=1)
        for i in range(3):
            engine.submit("stream", 0.0, 20, 5000, Lineage())
        threshold = engine.threshold
        doc = engine.submit("doc", 0.1, 3 * threshold + 200, 1, Lineage())
        engine.run(30.0)

        self.assertIsNotNone(doc.first)
        doc_start = doc.admitted
        doc_end = doc.prefilled
        steps_during = [
            s for s in engine.steps
            if s.start >= doc_start - 1e-9 and s.start < doc_end - 1e-9
        ]
        # 4 prefill chunks, with 2 pure decode steps (prefill_tokens == 0) between
        # each of the 3 non-final chunks -> 6 pure decode steps during prefill.
        prefill_steps = [s for s in steps_during if s.prefill_tokens > 0]
        pure_decode_steps = [s for s in steps_during if s.prefill_tokens == 0]
        self.assertEqual(len(prefill_steps), 4)
        self.assertEqual(
            [s.prefill_tokens for s in prefill_steps],
            [threshold, threshold, threshold, 200],
        )
        self.assertEqual(len(pure_decode_steps), 6)

    def test_interleave_policy_results_match_expected_improvements(self):
        mod = _load_interleave_policy()
        computed = mod.evaluate_all()
        saved = json.loads((EXP_DIR / "results.json").read_text())
        self.assertEqual(computed, saved)

        # 1. stall-15 (100% slot occupancy): prefill-yield-short does nothing,
        # while interleave-k2 gives >2.5x decode throughput, >55% lower mean ITL,
        # ~0.115s p50 ITL, <10% doc TTFT overhead, and 0 extra MoE passes.
        s15 = saved["scenarios"]["stall-15"]["policies"]
        self.assertEqual(
            s15["prefill-yield-short"]["decode_tok_s_during_prefill"],
            s15["production"]["decode_tok_s_during_prefill"],
        )
        self.assertGreaterEqual(s15["interleave-k2"]["decode_throughput_speedup_x"], 2.5)
        self.assertGreaterEqual(
            s15["interleave-k2"]["stream_itl_mean_reduction_pct"], 55.0
        )
        self.assertLessEqual(s15["interleave-k2"]["stream_itl_p50_s"], 0.12)
        self.assertLess(s15["interleave-k2"]["doc_ttft_overhead_pct"], 10.0)
        self.assertEqual(s15["interleave-k2"]["extra_moe_passes"], 0)

        # 2. stall-11: interleave-k2 strictly beats prefill-yield-short on BOTH
        # decode throughput/ITL AND short-probe TTFT.
        s11 = saved["scenarios"]["stall-11"]["policies"]
        self.assertGreater(
            s11["interleave-k2"]["decode_tok_s_during_prefill"],
            s11["prefill-yield-short"]["decode_tok_s_during_prefill"],
        )
        self.assertLess(
            s11["interleave-k2"]["stream_itl_mean_s"],
            s11["prefill-yield-short"]["stream_itl_mean_s"],
        )
        self.assertLess(
            s11["interleave-k2"]["probe_ttft_mean_s"],
            s11["prefill-yield-short"]["probe_ttft_mean_s"],
        )
        self.assertEqual(s11["interleave-k2"]["extra_moe_passes"], 0)

        # 3. stall-1m: ~3x decode throughput, ~65% lower mean ITL, <10% doc
        # overhead, and short probe TTFT < 2.5s throughout the 1M prefill.
        s1m = saved["scenarios"]["stall-1m"]["policies"]
        self.assertGreaterEqual(s1m["interleave-k2"]["decode_throughput_speedup_x"], 2.8)
        self.assertGreaterEqual(
            s1m["interleave-k2"]["stream_itl_mean_reduction_pct"], 64.0
        )
        self.assertLess(s1m["interleave-k2"]["doc_ttft_overhead_pct"], 10.0)
        self.assertLess(s1m["interleave-k2"]["probe_ttft_mean_s"], 2.5)
        self.assertEqual(s1m["interleave-k2"]["extra_moe_passes"], 0)

    def test_arm_configs_are_valid(self):
        for fname, expected_k, expected_shape in (
            ("interleave-k1-tp4.json", "1", "tp4-ring"),
            ("interleave-k2-tp4.json", "2", "tp4-ring"),
            ("interleave-k2-tp3.json", "2", "tp3-triangle"),
        ):
            cfg = json.loads((EXP_DIR / fname).read_text())
            self.assertEqual(cfg["shape"], expected_shape)
            self.assertEqual(
                cfg["serve_options"]["--prefill-decode-interleave-steps"],
                expected_k,
            )


if __name__ == "__main__":
    unittest.main()
