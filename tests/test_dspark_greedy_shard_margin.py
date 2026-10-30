from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT / "experiments/2026-10-30-dspark-greedy-shard-margin/margin_policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location("margin_policy", POLICY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DSparkGreedyShardMarginTests(unittest.TestCase):
    def test_shard_runner_up_and_cross_rank_margin_exactness(self) -> None:
        policy = _load_policy()
        rows, vocab_per_rank, ranks = 3, 4096, 4
        full_vocab = vocab_per_rank * ranks
        base = [[0.0] * full_vocab for _ in range(rows)]
        bias = [[0.0] * full_vocab for _ in range(rows)]

        # Row 0: clear winner on rank 2, runner-up on rank 1.
        base[0][2 * vocab_per_rank + 17] = 18.0
        bias[0][2 * vocab_per_rank + 17] = 2.0  # sum = 20.0
        base[0][1 * vocab_per_rank + 99] = 15.0
        bias[0][1 * vocab_per_rank + 99] = 1.5  # sum = 16.5

        # Row 1: winner and runner-up in different 2048-blocks of the SAME shard (rank 0).
        base[1][100] = 22.0
        bias[1][100] = 0.0
        base[1][2048 + 300] = 19.25
        bias[1][2048 + 300] = 0.0

        # Row 2: exact tie across ranks -> margin must be 0.0 and lowest index wins.
        base[2][500] = 30.0
        bias[2][500] = 0.0
        base[2][3 * vocab_per_rank + 500] = 30.0
        bias[2][3 * vocab_per_rank + 500] = 0.0

        shard_pairs = []
        for r in range(ranks):
            s_base = [row[r * vocab_per_rank : (r + 1) * vocab_per_rank] for row in base]
            s_bias = [row[r * vocab_per_rank : (r + 1) * vocab_per_rank] for row in bias]
            pairs = policy.reduce_shard_pair_with_runner_up(
                s_base, s_bias, r * vocab_per_rank, track_runner_up=True
            )
            self.assertEqual(len(pairs), rows)
            self.assertTrue(all(len(p) == 4 for p in pairs))
            self.assertTrue(all(p[3] == -3.0 for p in pairs))
            shard_pairs.append(pairs)

        gathered = []
        for row_idx in range(rows):
            row_flat: list[float] = []
            for r in range(ranks):
                row_flat.extend(shard_pairs[r][row_idx])
            gathered.append(row_flat)

        selected, margins = policy.select_first_argmax_with_margin(
            gathered, pair_width=4
        )
        self.assertEqual(selected, [2 * vocab_per_rank + 17, 100, 500])
        self.assertTrue(all(m >= 0.0 for m in margins))
        self.assertAlmostEqual(margins[0], 3.5, places=5)
        self.assertAlmostEqual(margins[1], 2.75, places=5)
        self.assertEqual(margins[2], 0.0)

    def test_calibrated_confidence_monotonic_in_margin_and_step_bias(self) -> None:
        policy = _load_policy()
        reqs, n_spec = 4, 5
        total = reqs * n_spec
        base_logits = [0.0] * total
        margins = [0.2 * i for i in range(total)]
        step_biases = [0.0, -0.15, -0.30, -0.45, -0.60]

        uncalibrated = policy.calibrate_confidence_probabilities(base_logits)
        for p in uncalibrated:
            self.assertAlmostEqual(p, 0.5, places=7)

        with_margin = policy.calibrate_confidence_probabilities(
            base_logits, margins, margin_weight=0.25
        )
        self.assertTrue(all(p >= u for p, u in zip(with_margin, uncalibrated)))
        self.assertTrue(all(with_margin[i + 1] > with_margin[i] for i in range(total - 1)))

        with_step_bias = policy.calibrate_confidence_probabilities(
            base_logits, margins=None, margin_weight=0.0, step_biases=step_biases
        )
        for req in range(reqs):
            row = with_step_bias[req * n_spec : (req + 1) * n_spec]
            self.assertTrue(all(row[k] < row[k - 1] for k in range(1, n_spec)))

    def test_suite_profiles_confirm_zero_extra_bytes_and_exposure_bias(self) -> None:
        policy = _load_policy()
        summary = policy.evaluate_suite_profiles()
        profiles = summary["profiles"]
        self.assertIn("tp4_c16", profiles)
        self.assertIn("tp4_sampled_c16", profiles)
        for profile in profiles.values():
            self.assertEqual(profile["tp4_allgather_bytes_per_req_per_step"], 64)
            self.assertEqual(profile["extra_collective_bytes"], 0)
            self.assertEqual(profile["recommended_step_biases"][0], 0.0)
            self.assertLess(profile["recommended_step_biases"][-1], 0.0)


if __name__ == "__main__":
    unittest.main()
