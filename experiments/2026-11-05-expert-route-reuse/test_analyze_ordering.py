import unittest
from analyze_ordering import cta_order, ordering_metrics, scale_bounds


class OrderingTest(unittest.TestCase):
    def test_partial_and_oversized_panels_are_permutations(self):
        for nx in (1, 3, 9, 12, 40):
            for ny in (1, 2, 3, 7, 17):
                expected = {(x, y) for x in range(nx) for y in range(ny)}
                for panel in (0, 1, 2, 3, 4, 8):
                    coords = list(cta_order(nx, ny, panel))
                    self.assertEqual(len(coords), len(set(coords)))
                    self.assertEqual(set(coords), expected)

    def test_low_overlap_has_no_repeated_weight_payload(self):
        for panel in (0, 1, 2, 3, 4, 8):
            out = ordering_metrics([0, 1, 2, -1], 512, 576, 2, panel)
            self.assertEqual(out['repeated_weight_payload_bytes'], 0)
            self.assertEqual(out['active_ctas'], 12)

    def test_interleaving_tradeoff_and_payload_are_explicit(self):
        plain = ordering_metrics([0, 0, 0], 512, 576, 2, 0)
        swizzled = ordering_metrics([0, 0, 0], 512, 576, 2, 3)
        self.assertEqual(plain['requested_weight_payload_bytes'], swizzled['requested_weight_payload_bytes'])
        self.assertEqual(plain['weight_reuse_grid_index_distance']['median'], 4)
        self.assertEqual(swizzled['weight_reuse_grid_index_distance']['median'], 1)
        self.assertEqual(plain['same_input_tile_reuse_grid_index_distance']['median'], 1)
        self.assertEqual(swizzled['same_input_tile_reuse_grid_index_distance']['median'], 3)

    def test_scale_bounds_cannot_remove_fp4_payload(self):
        for bound in scale_bounds():
            self.assertLess(bound['all_4_to_all_2_payload_saving_ceiling'], 0.016)
            self.assertLess(bound['remove_all_scales_payload_saving_ceiling'], 0.033)
            self.assertLess(bound['remove_unused_4bit_tail_payload_saving_ceiling'], 0.0012)


if __name__ == '__main__':
    unittest.main()
