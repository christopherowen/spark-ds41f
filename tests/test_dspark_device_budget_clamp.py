"""Unit tests for eliminating 1-step async budget staleness leaks between CPU
graph budgeting and GPU dead-row cutting in DSpark adaptive verification.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "2026-10-30-dspark-device-budget-clamp"


def _load_staleness_policy():
    spec = importlib.util.spec_from_file_location(
        "staleness_policy", EXP_DIR / "staleness_policy.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["staleness_policy"] = module
    spec.loader.exec_module(module)
    return module


class DSparkDeviceBudgetClampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _load_staleness_policy()

    def test_host_budget_filters_sub_tau_draft_slots(self) -> None:
        conf = [[0.82, 0.60, 0.55, 0.55, 0.55]]
        num_tokens_unfiltered, budget_unfiltered = self.policy.host_budget_num_tokens(
            conf,
            [5],
            [1],
            row_cost=0.02,
            dead_rows_tau=0.0,
        )
        self.assertEqual(budget_unfiltered, 5)
        self.assertEqual(num_tokens_unfiltered, 6)

        num_tokens_filtered, budget_filtered = self.policy.host_budget_num_tokens(
            conf,
            [5],
            [1],
            row_cost=0.02,
            dead_rows_tau=0.35,
        )
        self.assertEqual(budget_filtered, 2)
        self.assertEqual(num_tokens_filtered, 3)

    def test_c1_high_to_low_transition_clamps_verified_draft_counts(self) -> None:
        # Step t-1 was high confidence on request 0 -> host budgets 5 drafts.
        # Request 1 is a prefill chunk with 0 scheduled drafts.
        prev_conf = [
            [0.95, 0.94, 0.93, 0.92, 0.91],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        _, draft_budget = self.policy.host_budget_num_tokens(
            prev_conf,
            [5, 0],
            [1, 32],
            row_cost=0.02,
            dead_rows_tau=0.35,
            is_sampling_request=[True, False],
        )
        self.assertEqual(draft_budget, 5)

        # Step t hits a hard branch on request 0: only draft 0 survives tau=0.35.
        curr_conf = [
            [0.80, 0.30, 0.30, 0.30, 0.30],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        admitted = self.policy.assign_device_draft_budget(
            curr_conf, [5, 0], draft_budget
        )
        self.assertEqual(admitted, [5, 0])

        cut = self.policy.compute_device_dead_row_cut(
            curr_conf,
            admitted,
            dead_rows_tau=0.35,
            is_decoding=[True, False],
        )
        self.assertEqual(cut, [1, 9])

        verified = self.policy.clamp_verified_draft_counts(admitted, cut)
        self.assertEqual(verified, [1, 0])

    def test_live_observed_rejected_eliminates_dead_row_false_rejections(self) -> None:
        # Request 0: capacity=5, dead_row_cut=2, both live drafts accepted
        #   (num_sampled=3, raw num_rejected=3) -> live_observed_rejected = 0.
        # Request 1: capacity=5, dead_row_cut=2, 1 live draft accepted and 1 live
        #   draft rejected (num_sampled=2, raw num_rejected=4) -> 1.
        # Request 2: capacity=5, dead_row_cut=0, 0 live drafts
        #   (num_sampled=1, raw num_rejected=5) -> 0.
        # Request 3: capacity=1, dead_row_cut=4, 1 admitted draft accepted
        #   (num_sampled=2, raw num_rejected=0) -> min(0, 3) = 0.
        dead_row_cut = [2, 2, 0, 4]
        num_sampled = [3, 2, 1, 2]
        raw_rejected = [3, 4, 5, 0]

        live_rejected = self.policy.compute_live_observed_rejected(
            num_sampled, raw_rejected, dead_row_cut
        )
        self.assertEqual(live_rejected, [0, 1, 0, 0])

        obs_stale, lbl_stale = self.policy.evaluate_estimator_labels(
            num_sampled, raw_rejected
        )
        # Without live-rejected clamping, request 0 grades dead step 2 as a false
        # rejection (observed=True, label=0.0) and request 2 grades dead step 0
        # as a false rejection.
        self.assertTrue(obs_stale[0][2])
        self.assertEqual(lbl_stale[0][2], 0.0)
        self.assertTrue(obs_stale[2][0])
        self.assertEqual(lbl_stale[2][0], 0.0)

        obs_clamped, lbl_clamped = self.policy.evaluate_estimator_labels(
            num_sampled, live_rejected
        )
        # With live-rejected clamping:
        # Request 0 observes only steps 0 and 1 (both label=1.0); step 2 is not observed.
        self.assertEqual(obs_clamped[0], [True, True, False, False, False])
        self.assertEqual(lbl_clamped[0][:2], [1.0, 1.0])
        # Request 1 observes step 0 (label=1.0) and step 1 (label=0.0).
        self.assertEqual(obs_clamped[1], [True, True, False, False, False])
        self.assertEqual(lbl_clamped[1][:2], [1.0, 0.0])
        # Request 2 (dead_row_cut=0) observes no drafts.
        self.assertEqual(obs_clamped[2], [False, False, False, False, False])
        # Request 3 (capacity=1, accepted) observes only step 0 (label=1.0).
        self.assertEqual(obs_clamped[3], [True, False, False, False, False])

        # Composes with sink draft-zero rejection masking.
        with_sink = self.policy.compute_live_observed_rejected(
            num_sampled,
            raw_rejected,
            dead_row_cut,
            sink_rejected_mask=[False, True, False, False],
        )
        self.assertEqual(with_sink, [0, 0, 0, 0])

    def test_overlay_and_bundle_parity_and_ast_contracts(self) -> None:
        overlay_av = (
            EXP_DIR
            / "overlay"
            / "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        )
        overlay_mr = EXP_DIR / "overlay" / "vllm/v1/worker/gpu/model_runner.py"
        bundle_av = (
            EXP_DIR
            / "bundles/device-clamp-tests/overlay"
            / "vllm/v1/worker/gpu/spec_decode/adaptive_verification.py"
        )
        bundle_mr = (
            EXP_DIR
            / "bundles/device-clamp-tests/overlay"
            / "vllm/v1/worker/gpu/model_runner.py"
        )

        self.assertEqual(overlay_av.read_text(), bundle_av.read_text())
        self.assertEqual(overlay_mr.read_text(), bundle_mr.read_text())

        av_src = overlay_av.read_text()
        mr_src = overlay_mr.read_text()
        ast.parse(av_src)
        ast.parse(mr_src)

        self.assertIn("valid = valid & (survival_probability >= dead_rows_tau)", av_src)
        self.assertIn("torch.minimum(capacities, cut[:num_reqs], out=capacities)", av_src)
        self.assertIn(
            "self.adaptive_verification.get_verified_draft_counts(\n"
            "                input_batch.num_reqs, self._dead_row_cut\n"
            "            )",
            mr_src,
        )
        self.assertIn(
            "observed_rejected = torch.minimum(\n"
            "                        observed_rejected, live_rejected\n"
            "                    )",
            mr_src,
        )

        recorded = json.loads((EXP_DIR / "results.json").read_text())
        self.assertEqual(recorded, self.policy.build_experiment_summary())


if __name__ == "__main__":
    unittest.main()
