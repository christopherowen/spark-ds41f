import math
import random
import unittest

from reference import bf16, decode, score, select_clusters, topk, upper_bound


class BoundTests(unittest.TestCase):
    def test_random_mxfp4_and_signed_weights(self):
        rng = random.Random(4104)
        for case in range(40):
            scales = (96, 97, 120, 126, 127, 128, 143, 144)
            def vector():
                sf = [rng.choice(scales) for _ in range(4)]
                return [decode(rng.randrange(16), sf[d // 32]) for d in range(128)]
            queries = [vector() for _ in range(4)]
            keys = [vector() for _ in range(8)]
            weights = [bf16(rng.uniform(-.1, .1)) for _ in queries]
            bound = upper_bound(queries, weights, keys)
            matrix_bound = upper_bound(queries, weights, keys, split=True)
            orders = [list(range(128)), list(reversed(range(128)))]
            permuted = list(range(128))
            rng.shuffle(permuted)
            orders.append(permuted)
            for key in keys:
                for order in orders:
                    observed = score(queries, weights, key, order)
                    self.assertGreaterEqual(bound, observed, case)
                    self.assertGreaterEqual(matrix_bound, observed, case)

    def test_certified_endpoints_are_exact_bf16(self):
        for scale in range(96, 145):
            for code in range(16):
                value = decode(code, scale)
                self.assertEqual(bf16(value), value)

    def test_split_bound_cancellation(self):
        for scale in (96, 127, 144):
            q = [decode(7 if d % 2 else 15, scale) for d in range(128)]
            keys = [[decode(7, scale)] * 128]
            self.assertGreaterEqual(upper_bound([q], [bf16(2**20)], keys, split=True),
                                    score([q], [bf16(2**20)], keys[0]))

    def test_shared_prefix_superset_preserves_selection(self):
        rng = random.Random(4179)
        keys = [[decode(rng.randrange(16), 127) for _ in range(128)] for _ in range(65)]
        for extent in (0, 1, 7, 37, 64, 65):
            queries = [[decode(rng.randrange(16), 127) for _ in range(128)] for _ in range(2)]
            weights = [bf16(.03), bf16(-.01)]
            scores = [score(queries, weights, key) for key in keys[:extent]]
            bounds = [upper_bound(queries, weights, keys[c:c+8], split=True)
                      for c in range(0, len(keys), 8)]
            selected, _ = select_clusters(scores, bounds, [0, 2, 4], 4, 8)
            self.assertEqual(selected, topk(scores, 4))

    def test_single_point_intervals_and_mixed_sign_queries(self):
        rng = random.Random(4117)
        for _ in range(24):
            q = [decode(rng.randrange(16), rng.choice((96, 127, 144))) for _ in range(128)]
            key = [decode(rng.randrange(16), rng.choice((96, 127, 144))) for _ in range(128)]
            queries = [q, list(reversed(q))]
            weights = [bf16(.03), bf16(-.02)]
            bound = upper_bound(queries, weights, [key] * 64)
            self.assertGreaterEqual(bound, score(queries, weights, key))

    def test_zero_and_cancellation(self):
        keys = [[(-1.)**d for d in range(128)], [0.] * 128]
        for weights in ([0.], [1.], [-1.]):
            queries = [[1.] * 128]
            bound = upper_bound(queries, weights, keys)
            for key in keys:
                self.assertGreaterEqual(bound, score(queries, weights, key))

    def test_unsupported_weights_keep_all(self):
        for weight in (math.nan, math.inf, -math.inf, 2**21):
            self.assertEqual(upper_bound([[1.]*128], [weight], [[1.]*128]), math.inf)

    def test_threshold_ties_keep_earliest_positions(self):
        scores = [1.] * 128 + [0.] * 128
        actual, kept = select_clusters(scores, [1., 1., 0., 0.], [1], 32, 64)
        self.assertEqual(actual, topk(scores, 32))
        self.assertEqual(len(kept), 128)

    def test_empty_and_short_seed(self):
        for scores in ([], [3.], [4., 3., 2., 1., 0.]):
            bounds = [max(scores)] if scores else []
            actual, _ = select_clusters(scores, bounds, [], 8, 64)
            self.assertEqual(actual, topk(scores, 8))

    def test_selection_random_and_partial_tail(self):
        rng = random.Random(4128)
        for length in (1, 37, 63, 64, 65, 127, 128, 129, 1025):
            scores = [float(rng.randrange(-4, 5)) for _ in range(length)]
            bounds = [max(scores[p:p+64]) for p in range(0, length, 64)]
            for k in (1, 8, 64, 512):
                actual, _ = select_clusters(scores, bounds, [len(bounds)-1], k, 64)
                self.assertEqual(actual, topk(scores, k))


if __name__ == '__main__':
    unittest.main()
