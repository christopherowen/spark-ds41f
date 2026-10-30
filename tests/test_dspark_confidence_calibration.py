"""Unit and contract tests for online calibration of DSpark confidence_head."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments" / "2026-10-30-dspark-confidence-calibration"
OVERLAY_DIR = EXP_DIR / "overlay" / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
BUNDLE_DIR = EXP_DIR / "bundles" / "calibration-tests"


def _load_calibration_policy_module():
    module_path = EXP_DIR / "calibration_policy.py"
    spec = importlib.util.spec_from_file_location("dspark_calibration_policy", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDSparkConfidenceCalibrationMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_calibration_policy_module()

    def test_cold_start_identity_calibration_preserves_head_probabilities(self) -> None:
        calibrator = self.mod.OnlineConfidenceCalibrator(
            max_num_reqs=4, num_speculative_steps=5
        )
        self.assertEqual(calibrator.slope, 1.0)
        self.assertEqual(calibrator.intercepts, [0.0, 0.0, 0.0, 0.0, 0.0])

        raw = [
            [0.78, 0.72, 0.71, 0.71, 0.73],
            [0.92, 0.65, 0.40, 0.18, 0.05],
        ]
        probs = [list(row) for row in raw]
        calibrator.calibrate_confidence_probs(probs, [2, 0])

        for r in range(2):
            for k in range(5):
                self.assertAlmostEqual(probs[r][k], raw[r][k], places=6)

        # Verify persistent slot indexing (batch row 0 -> slot 2, batch row 1 -> slot 0).
        for k in range(5):
            self.assertAlmostEqual(calibrator.predictions[2][k], raw[0][k], places=6)
            self.assertAlmostEqual(calibrator.predictions[0][k], raw[1][k], places=6)
            self.assertAlmostEqual(
                calibrator.features[2][k], self.mod.logit(raw[0][k]), places=6
            )

    def test_calibrate_and_step_skip_negative_padded_slots(self) -> None:
        calibrator = self.mod.OnlineConfidenceCalibrator(
            max_num_reqs=4, num_speculative_steps=5, refit_interval=2
        )
        calibrator.features[1] = [1.25] * 5
        calibrator.predictions[1] = [0.77] * 5

        probs = [
            [0.80, 0.75, 0.70, 0.65, 0.60],
            [0.50, 0.50, 0.50, 0.50, 0.50],
        ]
        # Batch row 1 is a CUDA-graph-padded row with req_state_idx == -1.
        calibrator.calibrate_confidence_probs(probs, [0, -1])
        self.assertEqual(calibrator.features[1], [1.25] * 5)
        self.assertEqual(calibrator.predictions[1], [0.77] * 5)

        calibrator.step([0, -1], num_sampled=[3, 5], num_rejected=[3, 1])
        # Only slot 0 is graded: accepted = 2, admitted = 5 -> positions 0, 1, 2 graded.
        self.assertEqual(calibrator.counts, [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_per_token_sample_idx_mapping_resolves_slot_correctly(self) -> None:
        calibrator = self.mod.OnlineConfidenceCalibrator(
            max_num_reqs=4, num_speculative_steps=5
        )
        probs = [
            [0.81, 0.74, 0.69, 0.63, 0.58],
            [0.64, 0.55, 0.47, 0.39, 0.31],
        ]
        # 2 requests * 5 speculative steps = 10 entries in sample_idx_mapping.
        per_token_mapping = [3] * 5 + [1] * 5
        calibrator.calibrate_confidence_probs(probs, per_token_mapping)
        for k in range(5):
            self.assertAlmostEqual(calibrator.predictions[3][k], probs[0][k], places=6)
            self.assertAlmostEqual(calibrator.predictions[1][k], probs[1][k], places=6)

    def test_online_irls_corrects_overconfident_tail_survival_across_workloads(self) -> None:
        summary = self.mod.evaluate_all()
        saved = json.loads((EXP_DIR / "results.json").read_text())
        self.assertEqual(summary, saved)

        for name in ("single-stream/c1", "penalties-sampled/c8", "chat-Mixed/c16"):
            entry = summary[name]
            self.assertLess(
                entry["survival_mae_calibrated"],
                0.025,
                msg=f"{name} calibrated survival MAE too high: {entry}",
            )
            self.assertLess(
                entry["survival_mae_calibrated"],
                entry["survival_mae_static"] * 0.30,
            )
            self.assertEqual(
                entry["admitted_depth_calibrated"],
                entry["admitted_depth_empirical_optimal"],
                msg=f"{name} admitted depth mismatch: {entry}",
            )
            # Calibrated tail survival at positions 2, 3, 4 must move closer to empirical
            # survival and strictly below the overconfident static cumprod.
            for k in (2, 3, 4):
                self.assertLess(
                    entry["calibrated_survival"][k],
                    entry["static_survival"][k],
                )

        for name in ("single-stream/c1", "penalties-sampled/c8"):
            intercepts = summary[name]["fitted_intercepts"]
            self.assertLess(intercepts[2], -1.0)
            self.assertLess(intercepts[3], -0.9)
            self.assertLess(intercepts[4], -0.35)

    def test_online_adaptation_across_code_to_sampled_prose_domain_shift(self) -> None:
        calibrator = self.mod.OnlineConfidenceCalibrator(
            max_num_reqs=16, num_speculative_steps=5, refit_interval=10
        )
        rng = random.Random(42)
        slots = list(range(16))
        head_row = [0.78, 0.72, 0.71, 0.71, 0.73]

        # Phase 1: high-acceptance structured code (positions 0..4 accept at ~85% cond).
        for _ in range(80):
            batch = [
                [
                    self.mod.sigmoid(self.mod.logit(head_row[k]) + rng.gauss(0.0, 0.2))
                    for k in range(5)
                ]
                for _ in range(16)
            ]
            calibrator.calibrate_confidence_probs(batch, slots)
            sampled, rejected = [], []
            for _r in range(16):
                acc = 0
                for _k in range(5):
                    if rng.random() < 0.85:
                        acc += 1
                    else:
                        break
                sampled.append(acc + 1)
                rejected.append(5 - acc)
            calibrator.step(slots, sampled, rejected)

        code_probe = [list(head_row)]
        calibrator.calibrate_confidence_probs(code_probe, [0])
        code_survival = self.mod.cumprod(code_probe[0])

        # Phase 2: domain shift to high-entropy sampled prose (cond [0.58, 0.52, 0.40, 0.35, 0.35]).
        prose_cond = [0.58, 0.52, 0.40, 0.35, 0.35]
        for _ in range(120):
            batch = [
                [
                    self.mod.sigmoid(self.mod.logit(head_row[k]) + rng.gauss(0.0, 0.2))
                    for k in range(5)
                ]
                for _ in range(16)
            ]
            calibrator.calibrate_confidence_probs(batch, slots)
            sampled, rejected = [], []
            for _r in range(16):
                acc = 0
                for k in range(5):
                    if rng.random() < prose_cond[k]:
                        acc += 1
                    else:
                        break
                sampled.append(acc + 1)
                rejected.append(5 - acc)
            calibrator.step(slots, sampled, rejected)

        prose_probe = [list(head_row)]
        calibrator.calibrate_confidence_probs(prose_probe, [0])
        prose_survival = self.mod.cumprod(prose_probe[0])

        self.assertGreater(code_survival[3], 0.40)
        self.assertLess(prose_survival[3], 0.10)

    def test_zero_observation_tail_positions_keep_intercept_unchanged(self) -> None:
        calibrator = self.mod.OnlineConfidenceCalibrator(
            max_num_reqs=4, num_speculative_steps=5, refit_interval=1
        )
        probs = [[0.78, 0.72, 0.71, 0.71, 0.73] for _ in range(4)]
        calibrator.calibrate_confidence_probs(probs, [0, 1, 2, 3])
        # Every request rejects at position 0 (num_sampled=1, num_rejected=5).
        # Positions 1..4 are never reached or graded (k <= accepted is False for k >= 1).
        calibrator.step([0, 1, 2, 3], [1, 1, 1, 1], [5, 5, 5, 5])
        self.assertLess(calibrator.intercepts[0], 0.0)
        self.assertEqual(calibrator.intercepts[1:], [0.0, 0.0, 0.0, 0.0])


class TestDSparkConfidenceCalibrationOverlayContracts(unittest.TestCase):
    def test_overlay_files_parse_and_expose_calibration_entry_points(self) -> None:
        ae_path = OVERLAY_DIR / "acceptance_estimator.py"
        spec_path = OVERLAY_DIR / "speculator.py"
        dspark_path = OVERLAY_DIR / "dspark" / "speculator.py"

        for path in (ae_path, spec_path, dspark_path):
            self.assertTrue(path.is_file(), f"Missing overlay file: {path}")
            ast.parse(path.read_text(), filename=str(path))

        ae_src = ae_path.read_text()
        self.assertIn("def _calibrate_confidence_kernel(", ae_src)
        self.assertIn("def for_confidence_head(", ae_src)
        self.assertIn("def reset_for_confidence_head(", ae_src)
        self.assertIn("def calibrate_confidence_probs(", ae_src)
        self.assertIn("_INIT_CONFIDENCE_HEAD_SLOPE = 1.0", ae_src)
        self.assertIn("_INIT_CONFIDENCE_HEAD_BIAS = 0.0", ae_src)

        spec_src = spec_path.read_text()
        self.assertIn("OnlineAcceptanceEstimator.for_confidence_head(", spec_src)

        dspark_src = dspark_path.read_text()
        self.assertIn("def _needs_logits_for_acceptance(", dspark_src)
        self.assertIn("self.acceptance_estimator.calibrate_confidence_probs(", dspark_src)
        self.assertIn("self.acceptance_estimator.reset_for_confidence_head()", dspark_src)
        # Ensure load_draft_model no longer disables use_acceptance_estimator when
        # use_confidence_head is True.
        self.assertNotIn("self.use_acceptance_estimator = False", dspark_src)

    def test_bundle_overlay_matches_canonical_overlay_byte_for_byte(self) -> None:
        rel_paths = [
            Path("vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py"),
            Path("vllm/v1/worker/gpu/spec_decode/speculator.py"),
            Path("vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"),
        ]
        for rel in rel_paths:
            canonical = EXP_DIR / "overlay" / rel
            bundled = BUNDLE_DIR / "overlay" / rel
            self.assertTrue(bundled.is_file(), f"Missing bundled overlay file: {bundled}")
            self.assertEqual(
                canonical.read_bytes(),
                bundled.read_bytes(),
                f"Overlay drift between {canonical} and {bundled}",
            )


if __name__ == "__main__":
    unittest.main()
