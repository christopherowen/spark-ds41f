"""Tests for piecewise CUDA-graph / 16-row GEMM tile / dead-row step-cost pricing
in DSpark AdaptiveVerificationManager (experiments/2026-10-30-dspark-row-cost)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-10-30-dspark-row-cost"
sys.path.insert(0, str(EXPERIMENT))

import row_cost_policy as policy  # noqa: E402


def load_spark():
    loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader("spark", loader)
    spark = importlib.util.module_from_spec(spec)
    loader.exec_module(spark)
    return spark


class TestDSparkPiecewiseRowCost(unittest.TestCase):
    def test_backward_compatible_linear_step_costs_when_unconfigured(self) -> None:
        scores = [0.85, 0.60, 0.40, 0.20, 0.10]
        costs = policy.step_costs(2, scores, 0.035)
        expected = [1.0 + 0.035 * (2 + k) for k in range(len(scores) + 1)]
        for actual, want in zip(costs, expected):
            self.assertAlmostEqual(actual, want, places=12)

    def test_cudagraph_capture_snapping(self) -> None:
        self.assertEqual(policy.padded_capture_size(1), 1)
        self.assertEqual(policy.padded_capture_size(4), 4)
        # No 5-row graph in TP4 ladder: rows=5 snaps to the 6-row CUDA graph.
        self.assertEqual(policy.padded_capture_size(5), 6)
        self.assertEqual(policy.padded_capture_size(6), 6)
        self.assertEqual(policy.padded_capture_size(9), 12)
        self.assertEqual(policy.padded_capture_size(17), 20)
        self.assertEqual(policy.padded_capture_size(33), 40)
        self.assertEqual(policy.padded_capture_size(65), 72)
        # Eager sizes above the capture limit stay unpadded.
        self.assertEqual(policy.padded_capture_size(120), 120)

    def test_gemm_16_and_lm_head_64_tile_boundary_pricing(self) -> None:
        rc = 0.045
        for r_before, r_after, expected_equiv_jump in (
            # 16 -> 17 pads to 20 (+2.0 graph-padded equiv) + 2nd 16-row GEMM tile (+1.0)
            # + 1 live MoE row (+0.5) = +3.5 * row_cost
            (16, 17, 3.5),
            # 32 -> 33 pads to 40 (+4.0 graph-padded equiv) + 3rd 16-row GEMM tile (+1.0)
            # + 1 live MoE row (+0.5) = +5.5 * row_cost
            (32, 33, 5.5),
            # 48 -> 49 pads to 56 (+4.0 graph-padded equiv) + 4th 16-row GEMM tile (+1.0)
            # + 1 live MoE row (+0.5) = +5.5 * row_cost
            (48, 49, 5.5),
            # 64 -> 65 pads to 72 (+4.0 graph-padded equiv) + 5th 16-row GEMM tile (+1.0)
            # + 2nd 64-row LM-head tile (+1.0) + 1 live MoE row (+0.5) = +6.5 * row_cost
            (64, 65, 6.5),
        ):
            costs = policy.step_costs(
                r_before,
                [1.0],
                rc,
                capture_sizes=policy.TP4_CAPTURE_SIZES,
            )
            self.assertAlmostEqual(
                costs[1] - costs[0], expected_equiv_jump * rc, places=12
            )
        # In-tile slots after the jump only pay live MoE cost (0.5 * row_cost).
        in_tile_costs = policy.step_costs(
            17,
            [1.0, 1.0, 1.0],
            rc,
            capture_sizes=policy.TP4_CAPTURE_SIZES,
        )
        for k in range(3):
            self.assertAlmostEqual(
                in_tile_costs[k + 1] - in_tile_costs[k], 0.5 * rc, places=12
            )

    def test_c1_marginal_threshold_prunes_7pct_and_4pct_tail_slots(self) -> None:
        # Recorded 1030-tp4 single-stream/c1 mean survival curve:
        c1_surv = [0.6545, 0.3548, 0.1569, 0.0720, 0.0386]
        # At legacy row_cost=0.02, the 4th-draft threshold is ~4.2%, admitting pos 3 (7.2%).
        thresh_020 = policy.marginal_survival_threshold(1, 1, c1_surv[:3], 0.02)
        self.assertLess(thresh_020, 0.045)
        self.assertGreater(c1_surv[3], thresh_020)

        # At row_cost=0.045, the linear 4th-draft threshold rises to ~8.26% (~9.2% at 5th)
        # and the piecewise 4 -> 6 CUDA-graph jump threshold rises to ~12.39%.
        thresh_045_lin = policy.marginal_survival_threshold(1, 1, c1_surv[:3], 0.045)
        thresh_045_pw = policy.marginal_survival_threshold(
            1, 1, c1_surv[:3], 0.045, capture_sizes=policy.TP4_CAPTURE_SIZES
        )
        self.assertGreater(thresh_045_lin, c1_surv[3])
        self.assertGreater(thresh_045_pw, 0.12)

        # Legacy row_cost=0.02 admits 4 drafts (rows=5 -> padded to 6-row CUDA graph).
        legacy_budget, _ = policy.select_draft_budget(1, 1, c1_surv, 0.02)
        self.assertEqual(legacy_budget, 4)

        # Piecewise row_cost=0.045 with dead_rows_tau=0.25 prunes tail slots < 0.25
        # and keeps c1 in the 3-row CUDA graph.
        tuned_budget, _ = policy.select_draft_budget(
            1,
            1,
            c1_surv,
            0.045,
            capture_sizes=policy.TP4_CAPTURE_SIZES,
            dead_rows_tau=0.25,
        )
        self.assertEqual(tuned_budget, 2)

    def test_c16_snaps_to_32_row_gemm_tile_boundary_instead_of_spilling(self) -> None:
        # 16 decode requests (16 non-draft rows): 16 strong first drafts (0.78) bring
        # the batch to rows=32 (exact 32-row CUDA graph, 2nd 16-row GEMM tile),
        # followed by 4 moderate second drafts (0.50) that beat linear row_cost=0.035
        # (threshold ~0.47) and spill rows=36 into the 40-row CUDA graph and 3rd
        # 16-row GEMM tile, but lose under piecewise tile pricing.
        scores = [0.78] * 16 + [0.50] * 4
        lin_budget, _ = policy.select_draft_budget(16, 16, scores, 0.035)
        pw_budget, _ = policy.select_draft_budget(
            16,
            16,
            scores,
            0.045,
            capture_sizes=policy.TP4_CAPTURE_SIZES,
            dead_rows_tau=0.25,
        )
        # Linear row_cost=0.035 spills 4 rows past 32 into the 40-row graph (rows=36).
        self.assertEqual(16 + lin_budget, 36)
        # Piecewise row_cost=0.045 snaps cleanly to the 32-row graph/tile boundary.
        self.assertEqual(16 + pw_budget, 32)

    def test_overlay_contains_piecewise_step_costs_and_bundle_tests(self) -> None:
        overlay_src = (
            EXPERIMENT
            / "overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        ).read_text()
        bundle_src = (
            EXPERIMENT
            / "bundles/row-cost-tests/overlay/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        ).read_text()
        self.assertEqual(overlay_src, bundle_src)
        self.assertIn("def padded_capture_sizes(", overlay_src)
        self.assertIn("def step_costs(", overlay_src)
        self.assertIn("GRAPH_PAD_COST_FRACTION", overlay_src)
        self.assertIn("GEMM_TILE_ROW_EQUIV", overlay_src)
        self.assertIn("LM_HEAD_TILE_ROW_EQUIV", overlay_src)

    def test_saved_results_match_replayed_trace_and_arms_resolve(self) -> None:
        saved = json.loads((EXPERIMENT / "results.json").read_text())
        computed = policy.build_summary()
        self.assertEqual(saved, computed)

        for w_name, w_data in saved["workloads"].items():
            pw = w_data["policies"]["piecewise_rc045_tau025"]
            self.assertGreater(pw["row_yield_vs_baseline_pct"], 0.0, w_name)
            self.assertGreater(pw["tps_vs_baseline_pct"], 0.0, w_name)

        spark = load_spark()
        for arm_file in (
            "piecewise-row045-tau025-tp4.json",
            "piecewise-row045-tau025-tp4-dev.json",
            "linear-row045-tau025-tp4.json",
            "linear-row045-tau025-tp4-dev.json",
        ):
            cluster = spark.read_cluster(
                str((EXPERIMENT / arm_file).relative_to(ROOT))
            )
            self.assertEqual(
                cluster["environment"]["SPARK_DS41F_DSPARK_DEAD_ROWS_TAU"], "0.25"
            )
            args = cluster["serve_args"]
            spec = json.loads(args[args.index("--speculative-config") + 1])
            self.assertEqual(spec["adaptive_verification_row_cost"], 0.045)


if __name__ == "__main__":
    unittest.main()
