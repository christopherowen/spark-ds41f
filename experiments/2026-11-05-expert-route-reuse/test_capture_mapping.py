"""CPU contracts for exact scheduler/consumer row joins; no model or GPU."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('route_capture', Path(__file__).parent / 'capture/route_capture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
map_rows = module.map_rows
LIVE = [0, 3, 7, 8, 11, 383]
DEAD = [-1] * 6


class MappingTests(unittest.TestCase):
    def metadata(self):
        return dict(starts=[0, 3, 5], full_positions=[100, 101, 102, 50, 51, -1],
                    full_padding=[False] * 5 + [True],
                    request_names=['a', 'b'], prefill_lengths=[100, 100])

    def test_decode_prefill_and_padding(self):
        got = map_rows([LIVE] * 5 + [DEAD], [0, 0, 0, 1, 1, -1],
                       [100, 101, 102, 50, 51, -1], **self.metadata())
        self.assertEqual([r['kind'] for r in got], ['anchor', 'draft', 'draft', 'prefill', 'prefill', 'padding'])
        self.assertEqual(got[4]['request'], 'b')
        self.assertEqual(got[4]['full_row'], 4)

    def test_ced_subset_joins_by_coordinate(self):
        got = map_rows([LIVE, LIVE, DEAD], [0, 1, -1], [102, 51, -1], **self.metadata())
        self.assertEqual([r['full_row'] for r in got], [2, 4, None])
        self.assertEqual([r['kind'] for r in got], ['draft', 'prefill', 'padding'])
        # The first compact row is a draft; it is not re-labelled an anchor.

    def test_adaptive_gpu_starts_are_authoritative(self):
        data = self.metadata()
        data.update(starts=[0, 2, 5], full_positions=[100, 101, 50, 51, 52, -1])
        got = map_rows([LIVE, LIVE], [0, 1], [101, 50], **data)
        self.assertEqual([r['full_row'] for r in got], [1, 2])
        self.assertEqual([r['kind'] for r in got], ['draft', 'prefill'])

    def test_dead_rows_are_retained(self):
        data = self.metadata(); data['full_padding'][1] = True
        got = map_rows([DEAD], [0], [101], **data)
        self.assertEqual(got[0]['kind'], 'dead')
        self.assertEqual(got[0]['request'], 'a')

    def test_live_routes_on_masked_rows_fail(self):
        data = self.metadata(); data['full_padding'][1] = True
        with self.assertRaisesRegex(ValueError, 'still routes'):
            map_rows([LIVE], [0], [101], **data)

    def test_missing_coordinate_fails_without_guessing(self):
        with self.assertRaisesRegex(ValueError, 'no exact scheduler'):
            map_rows([LIVE], [1], [100], **self.metadata())
        with self.assertRaisesRegex(ValueError, 'no exact scheduler'):
            map_rows([DEAD], [1], [100], **self.metadata())

    def test_duplicate_positions_fail(self):
        data = self.metadata(); data['full_positions'][1] = 100
        with self.assertRaisesRegex(ValueError, 'duplicate scheduler'):
            map_rows([LIVE], [0], [100], **data)

    def test_invalid_native_ids_fail(self):
        for ids in ([0] * 6, [0, 3, 7, 8, 11, 384], LIVE[:5]):
            with self.assertRaises(ValueError):
                map_rows([ids], [0], [100], **self.metadata())

    def test_mapping_does_not_reorder_slots(self):
        ids = [list(reversed(LIVE))]
        got = map_rows(ids, [0], [100], **self.metadata())
        self.assertEqual(got[0]['ids'], list(reversed(LIVE)))
        self.assertEqual(ids, [list(reversed(LIVE))])

    def test_shape_and_upper_bound_mismatches_fail(self):
        with self.assertRaisesRegex(ValueError, 'metadata differ'):
            map_rows([LIVE], [], [100], **self.metadata())
        data = self.metadata(); data['starts'] = [0, 3, 7]
        with self.assertRaisesRegex(ValueError, 'outside allocated'):
            map_rows([LIVE], [0], [100], **data)


if __name__ == '__main__':
    unittest.main()
