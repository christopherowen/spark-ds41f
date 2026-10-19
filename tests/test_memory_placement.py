"""The placement probe must refuse allocations that encroach on the guard."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/2026-10-19-memory-placement/bundles/placement/placement.py'
spec = importlib.util.spec_from_file_location('placement', path)
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


class MemoryPlanTest(unittest.TestCase):
    def test_twenty_buffers_plus_guard_and_workspace_fit(self):
        self.assertEqual(placement.memory_plan(80 << 30)['count'], 20)
        with self.assertRaisesRegex(RuntimeError, 'reserve'):
            placement.memory_plan(44 << 30)
        for count, size in ((1, 8), (20, 7), (20, 0)):
            with self.assertRaises(ValueError):
                placement.memory_plan(80 << 30, count, size)
