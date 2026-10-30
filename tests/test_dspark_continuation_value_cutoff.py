import importlib.util
import math
from pathlib import Path
import unittest


def _load_continuation_policy():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "2026-10-30-dspark-continuation-value-cutoff"
        / "continuation_policy.py"
    )
    spec = importlib.util.spec_from_file_location("continuation_policy", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDSparkContinuationValueCutoff(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = _load_continuation_policy()

    def test_monotonicity_along_draft_positions(self) -> None:
        cases = [
            [0.95, 0.25, 0.98, 0.97, 0.96],
            [0.40, 0.99, 0.99, 0.99, 0.99],
            [0.85, 0.80, 0.15, 0.10, 0.05],
            [1.00, 1.00, 1.00, 1.00, 1.00],
        ]
        for confs in cases:
            for valid_count in range(1, len(confs) + 1):
                for alpha in (0.0, 0.05, 0.15, 0.25, 0.50):
                    scores = self.policy.effective_survival_scores(
                        confs, valid_count, alpha
                    )
                    valid_scores = scores[:valid_count]
                    for idx in range(len(valid_scores) - 1):
                        self.assertGreaterEqual(
                            valid_scores[idx] + 1e-12,
                            valid_scores[idx + 1],
                        )
                    for idx in range(valid_count, len(scores)):
                        self.assertEqual(scores[idx], 0.0)

    def test_dead_row_cut_preserves_bridge_token_but_prunes_dead_tail(self) -> None:
        # Survival: [0.9, 0.72, 0.36, 0.324, 0.2916] -> draft 2 is 0.36 < 0.50,
        # but suffix sum behind draft 2 is 0.324 + 0.2916 = 0.6156.
        bridge_confs = [0.9, 0.8, 0.5, 0.9, 0.9]
        dead_tail_confs = [0.9, 0.8, 0.5, 0.1, 0.1]

        self.assertEqual(
            self.policy.dead_row_cut(bridge_confs, 5, tau=0.5, continuation_alpha=0.0),
            2,
        )
        self.assertEqual(
            self.policy.dead_row_cut(bridge_confs, 5, tau=0.5, continuation_alpha=0.25),
            3,
        )
        self.assertEqual(
            self.policy.dead_row_cut(dead_tail_confs, 5, tau=0.5, continuation_alpha=0.25),
            2,
        )

    def test_budget_selection_prefers_high_yield_continuation(self) -> None:
        batch_confs = [
            [0.90, 0.48, 0.95, 0.95, 0.95],  # bridge with strong continuation
            [0.90, 0.50, 0.05, 0.05, 0.05],  # slightly higher step 1, dead tail
        ]
        _, myopic_admitted = self.policy.select_batch_draft_budget(
            batch_confs,
            scheduled_drafts=[5, 5],
            num_non_draft_tokens=[1, 1],
            row_cost=0.01,
            continuation_alpha=0.0,
            max_total_logits=5,  # 2 bonus + 3 draft budget
        )
        _, cont_admitted = self.policy.select_batch_draft_budget(
            batch_confs,
            scheduled_drafts=[5, 5],
            num_non_draft_tokens=[1, 1],
            row_cost=0.01,
            continuation_alpha=0.15,
            max_total_logits=5,
        )
        self.assertEqual(myopic_admitted, [1, 2])
        self.assertEqual(cont_admitted, [2, 1])

    def test_row_cost_cutoff_recovers_bridge_sequence(self) -> None:
        bridge = [[0.90, 0.20, 0.95, 0.95]]
        dead_tail = [[0.90, 0.20, 0.10, 0.10]]
        myopic_budget, _ = self.policy.select_batch_draft_budget(
            bridge, [4], [1], row_cost=0.12, continuation_alpha=0.0
        )
        cont_budget, _ = self.policy.select_batch_draft_budget(
            bridge, [4], [1], row_cost=0.12, continuation_alpha=0.15
        )
        dead_budget, _ = self.policy.select_batch_draft_budget(
            dead_tail, [4], [1], row_cost=0.12, continuation_alpha=0.15
        )
        self.assertEqual(myopic_budget, 1)
        self.assertGreaterEqual(cont_budget, 2)
        self.assertEqual(dead_budget, 1)


if __name__ == "__main__":
    unittest.main()
