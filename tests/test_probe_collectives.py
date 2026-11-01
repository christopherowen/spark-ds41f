"""CPU-only fixtures for the TileLang collective's packed-row contract."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('collective_probe', ROOT / 'scripts/probe_collectives.py')
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PackedShapeTest(unittest.TestCase):
    def test_exact_payload_becomes_packed_rows_without_inflation(self):
        self.assertTrue(callable(getattr(probe, 'packed_shape', None)),
                        'probe must generate packed 2D shapes')
        for itemsize in (2, 4):
            pack = 16 // itemsize
            for elements in (pack, 17 * pack, 1024, 2 * 1024 * 1024,
                             5120, 17 * 5120, 5242880):
                with self.subTest(itemsize=itemsize, elements=elements):
                    rows, width = probe.packed_shape(elements, itemsize)
                    self.assertEqual(rows * width, elements)
                    self.assertEqual(width * itemsize % 16, 0)
                    # Mirrors all_reduce_rank_order's padded rows and zero_ slice:
                    # both the start address offset and fill size must be packs.
                    for world in (2, 3, 4, 5, 6, 7, 8):
                        padded = -(-rows // world) * world
                        self.assertEqual(rows * width * itemsize % 16, 0)
                        self.assertEqual((padded - rows) * width * itemsize % 16, 0)
            self.assertEqual(probe.packed_shape(17 * 5120, itemsize), (17, 5120))


    def test_scalar_payloads_are_explicitly_unsupported_not_rounded(self):
        for itemsize in (2, 4):
            for elements in (0, 1, 17, -8):
                with self.subTest(elements=elements, itemsize=itemsize):
                    with self.assertRaisesRegex(ValueError, 'whole 16-byte packs'):
                        probe.packed_shape(elements, itemsize)
        with self.assertRaises(ValueError):
            probe.packed_shape(64, 3)


    def test_correctness_cases_cover_tiny_model_rows_and_exact_dispatch_packs(self):
        self.assertTrue(callable(getattr(probe, 'correctness_shapes', None)),
                        'correctness must enumerate packed model-compatible shapes')
        limits = (2097152, 4194304, 1048576)
        for itemsize in (2, 4):
            shapes = probe.correctness_shapes(itemsize, limits)
            self.assertEqual(shapes, sorted(set(shapes)))
            self.assertTrue({(1, 16 // itemsize), (17, 16 // itemsize),
                             (1, 5120), (17, 5120), (205, 5120)} <= set(shapes))
            payloads = {rows * width * itemsize for rows, width in shapes}
            for limit in limits:
                self.assertTrue({limit - 16, limit, limit + 16} <= payloads)
            self.assertIn(1024 * itemsize, payloads)
            self.assertIn(2 * 1024 * 1024 * itemsize, payloads)
            self.assertLessEqual(max(payloads), max(2 * 1024 * 1024 * itemsize, max(limits) + 16))
            for rows, width in shapes:
                self.assertGreater(rows, 0)
                self.assertEqual(width * itemsize % 16, 0)


if __name__ == '__main__':
    unittest.main()
