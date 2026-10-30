import importlib.util
from pathlib import Path
import unittest


def _load_draft_depth_policy():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "2026-10-30-dspark-concurrency-draft-depth"
        / "draft_depth_policy.py"
    )
    spec = importlib.util.spec_from_file_location("draft_depth_policy", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDSparkConcurrencyDraftDepth(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _load_draft_depth_policy()

    def test_parse_schedule_formats_and_monotonic_clamping(self) -> None:
        self.assertEqual(self.policy.parse_max_steps_by_batch("", 5), ())
        self.assertEqual(
            self.policy.parse_max_steps_by_batch("8:4, 16:3", 5),
            ((8, 4), (16, 3)),
        )
        # Non-monotonic input is clamped to non-increasing steps with batch size.
        self.assertEqual(
            self.policy.parse_max_steps_by_batch({4: 3, 8: 4, 16: 2}, 5),
            ((4, 3), (16, 2)),
        )
        # Steps above num_speculative_steps are elided; steps < 1 clamp to 1.
        self.assertEqual(
            self.policy.parse_max_steps_by_batch([(2, 9), (8, 0)], 5),
            ((8, 1),),
        )

    def test_active_steps_and_work_savings_across_concurrency(self) -> None:
        schedule = self.policy.parse_max_steps_by_batch("8:4,16:3", 5)
        self.assertEqual(
            [self.policy.active_speculative_steps(b, 5, schedule) for b in (1, 4, 8, 16)],
            [5, 5, 4, 3],
        )
        c1 = self.policy.estimate_draft_work_savings(1, 5, schedule)
        c8 = self.policy.estimate_draft_work_savings(8, 5, schedule)
        c16 = self.policy.estimate_draft_work_savings(16, 5, schedule)
        self.assertEqual(c1["saved_rows"], 0)
        self.assertEqual(c8["saved_rows"], 8)
        self.assertEqual(c8["saved_markov_steps"], 1)
        self.assertAlmostEqual(float(c8["row_reduction_fraction"]), 0.20)
        self.assertEqual(c16["saved_rows"], 32)
        self.assertEqual(c16["saved_markov_steps"], 2)
        self.assertAlmostEqual(float(c16["row_reduction_fraction"]), 0.40)


if __name__ == "__main__":
    unittest.main()
