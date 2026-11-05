"""Checks for the offline evidence screen, not inference correctness tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("budget_screen", Path(__file__).with_name("replay.py"))
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def cell(requests, rows, cost, context=4096):
    return {"shape": {"kind": "decode", "num_reqs": requests, "num_tokens": rows, "context_len": context}, "step_ms": {"mean": cost}}


class ReplayEvidenceTests(unittest.TestCase):
    def test_never_pool_request_counts_or_contexts(self):
        grid = {"engine": [cell(1, 2, 10), cell(4, 8, 20), cell(1, 3, 30, 65536)]}
        self.assertEqual(screen.grid_cells(grid, 1, 4096), {2: 10})
        self.assertEqual(screen.grid_cells(grid, 16, 4096), {})

    def test_duplicate_or_nonfinite_cost_fails(self):
        with self.assertRaises(ValueError):
            screen.grid_cells({"engine": [cell(1, 2, 10), cell(1, 2, 11)]}, 1, 4096)
        for cost in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                screen.grid_cells({"engine": [cell(1, 2, cost)]}, 1, 4096)

    def test_unmeasured_optimum_is_reported_not_interpolated(self):
        rows = [([.9, .8, .1, .1, .1], 2)]
        result = screen.score(rows, {2: 10, 6: 30})
        self.assertFalse(result["unrestricted_control_covered"])
        for policy in ("linear035_at_measured_cells", "grid_at_measured_cells"):
            self.assertIn(result[policy]["draft_rows"] + 1, (2, 6))

    def test_fresh_confidence_prefix_and_common_ranking(self):
        rows = [([.9, .8, .7, .6, .5], 3), ([.1, .1, .1, .1, .1], 0)]
        result = screen.score(rows, {4: 10})
        # Both extra slots go to the first request. Two mandatory tokens plus
        # two matching drafts, not one draft allocated equally to each request.
        for policy in ("linear035_at_measured_cells", "grid_at_measured_cells"):
            self.assertEqual(result[policy]["matched_tokens"], 4)

    def test_missing_grid_produces_no_result(self):
        self.assertIsNone(screen.score([([.5] * 5, 0)], {}))

    def test_corrupt_acceptance_fails_before_any_cost_comparison(self):
        records = [
            {"t": 0, "b": 1, "r": "a", "s": [10], "d": [20, 30, 40, 50, 60], "c": [.5] * 5},
            {"t": 1, "b": 1, "r": "a", "s": [20, 30], "d": [40, 50, 60, 70, 80], "c": [.5] * 5, "v": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("".join(json.dumps(r) + "\n" for r in records))
            with self.assertRaisesRegex(ValueError, "acceptance does not replay"):
                screen.analyze(trace, [])

    def test_terminal_matching_tail_is_not_assumed_rejected(self):
        records = [[{"t": 0, "b": 1, "r": "a", "s": [10], "d": [20] * 5, "c": [.5] * 5}]]
        streams, anchors = screen.historical.committed_streams(records)
        self.assertEqual(screen.historical.decode_steps(records, streams, anchors, 5, 1, 16), [])


if __name__ == "__main__":
    unittest.main()
