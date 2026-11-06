from collections import Counter
import unittest
from screen_statistics import arm_order, interval


class ScreenStatisticsTests(unittest.TestCase):
    def test_two_arm_order_is_not_always_control_first(self):
        orders = [arm_order([0, 1], i) for i in range(48)]
        self.assertEqual(Counter(tuple(x) for x in orders), {(0, 1): 24, (1, 0): 24})

    def test_six_arms_fill_every_position_equally(self):
        orders = [arm_order(list(range(6)), i) for i in range(48)]
        for position in range(6):
            self.assertEqual(Counter(order[position] for order in orders), dict.fromkeys(range(6), 8))
        self.assertTrue(all(sorted(order) == list(range(6)) for order in orders))

    def test_interval_rejects_wrong_count_and_nonfinite_values(self):
        self.assertEqual(interval([0.] * 8), [0., 0.])
        self.assertEqual(interval([2.] * 8), [2., 2.])
        for values in ([0.] * 7, [0.] * 7 + [float('nan')]):
            with self.assertRaises(ValueError):
                interval(values)


if __name__ == '__main__':
    unittest.main()
