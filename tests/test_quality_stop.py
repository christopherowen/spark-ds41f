"""Sequential quality rules: the paired mixture likelihood ratio and the Wald slip test."""

import importlib.util
import math
from pathlib import Path
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "quality_stop", Path(__file__).resolve().parents[1] / "scripts/quality_stop.py"
)
quality_stop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quality_stop)


class PairedTest(unittest.TestCase):
    def test_one_discordant_win_is_no_evidence(self):
        log_e, stop = quality_stop.paired_e(1, 0)
        self.assertAlmostEqual(log_e, 0.0)
        self.assertFalse(stop)

    def test_one_way_wins_cross_five_percent_at_eight(self):
        self.assertFalse(quality_stop.paired_e(7, 0)[1])
        self.assertTrue(quality_stop.paired_e(8, 0)[1])

    def test_symmetric_in_direction(self):
        self.assertEqual(quality_stop.paired_e(8, 0), quality_stop.paired_e(0, 8))

    def test_exact_value(self):
        self.assertAlmostEqual(math.exp(quality_stop.paired_e(3, 1)[0]), 16 / 20)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            quality_stop.paired_e(-1, 0)


class WaldTest(unittest.TestCase):
    def test_states(self):
        self.assertEqual(quality_stop.wald([True] * 10, 0.1, 0.3)[0], "SLIP")
        self.assertEqual(quality_stop.wald([False] * 30, 0.1, 0.3)[0], "BASELINE")
        self.assertEqual(quality_stop.wald([], 0.1, 0.3)[0], "CONTINUE")

    def test_rejects_non_booleans(self):
        with self.assertRaises(ValueError):
            quality_stop.wald([1], 0.1, 0.3)


if __name__ == "__main__":
    unittest.main()
