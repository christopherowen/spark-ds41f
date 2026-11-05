"""Independent finite-choice checks for the frozen-trace upper bound."""

import itertools
import unittest

import audit


class BoundTests(unittest.TestCase):
    def test_aggregate_bound_beats_per_step_ratio_trap(self):
        choices = [[(1, 1), (100, 110)], [(100, 1000)]]
        result = audit.aggregate_oracle(choices)
        self.assertAlmostEqual(result["rate"], 200 / 1110)
        self.assertGreater(result["rate"], 101 / 1001)

    def test_oracle_matches_exhaustive_cross_step_choices(self):
        choices = [[(2, 10), (3, 12), (6, 19)], [(4, 15), (5, 22)],
                   [(1, 4), (4, 12), (7, 30)], [(0, 2), (2, 7)]]
        exact = max(sum(t for t, _ in selected) / sum(c for _, c in selected)
                    for selected in itertools.product(*choices))
        self.assertAlmostEqual(audit.aggregate_oracle(choices)["rate"], exact)

    def test_invalid_or_empty_choices_refused(self):
        for choices in ([], [[]], [[(1, 0)]], [[(1, float("nan"))]], [[(-1, 1)]]):
            with self.assertRaises(ValueError):
                audit.aggregate_oracle(choices)

    def test_oracle_ordering_and_single_cell_no_cost_choice(self):
        steps = [[([.9, .8, .7, .6, .5], 4), ([.1] * 5, 0)],
                 [([.1] * 5, 0), ([.9, .8, .7, .6, .5], 4)]]
        result = audit.compare(steps, {4: 12, 6: 15, 12: 30})
        oracles = result["frozen_trace_oracles"]
        best = max(p["modeled_delta_pct"] for p in result["policies"].values())
        self.assertGreaterEqual(oracles["budget_only"]["headroom_pct"] + 1e-10, best)
        self.assertGreaterEqual(oracles["budget_and_allocation"]["headroom_pct"] + 1e-10,
                                oracles["budget_only"]["headroom_pct"])
        single = audit.compare(steps, {6: 15})
        self.assertTrue(all(p["modeled_delta_pct"] == 0 for p in single["policies"].values()))
        self.assertEqual(single["frozen_trace_oracles"]["budget_only"]["headroom_pct"], 0)


if __name__ == "__main__":
    unittest.main()
