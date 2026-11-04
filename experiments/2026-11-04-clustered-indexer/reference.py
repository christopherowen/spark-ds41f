"""CPU arithmetic model and exact-selection oracle; no optional dependencies."""
import math
import struct


def f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', f32(value)))[0]
    if bits & 0x7f800000 != 0x7f800000:
        bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def decode(code, scale):
    values = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    if not 96 <= scale <= 144:
        raise ValueError('outside the certified normal arithmetic range')
    return math.copysign(math.ldexp(values[code & 7], scale - 127), -1 if code & 8 else 1)


def dot(q, key, order=None):
    total = 0.
    for d in range(len(q)) if order is None else order:
        total = f32(total + f32(q[d] * key[d]))
    return total


def score(queries, weights, key, order=None):
    total = 0.
    for q, w in zip(queries, weights):
        term = bf16(f32(max(bf16(dot(q, key, order)), 0.) * w))
        total = f32(total + term)
    return bf16(total)


def upper_bound(queries, weights, keys, *, split=False):
    lower = [min(column) for column in zip(*keys)]
    upper = [max(column) for column in zip(*keys)]
    total = 0.
    for q, w in zip(queries, weights):
        if not math.isfinite(w) or abs(w) > 2**20:
            return math.inf
        products = [f32(x * (hi if x >= 0 else lo)) for x, lo, hi in zip(q, lower, upper)]
        absolute = [f32(abs(x) * max(abs(lo), abs(hi))) for x, lo, hi in zip(q, lower, upper)]
        estimate, norm = 0., 0.
        if split:
            for endpoint, positive in ((upper, True), (lower, False)):
                for x, end in zip(q, endpoint):
                    part = max(x, 0.) if positive else min(x, 0.)
                    estimate = f32(estimate + f32(part * end))
            for a in absolute:
                norm = f32(norm + a)
        else:
            for p, a in zip(products, absolute):
                estimate, norm = f32(estimate + p), f32(norm + a)
        bound = max(f32(f32(estimate + f32(norm / 256.)) + 2**-50), 0.)
        term = bf16(f32(bf16(bound) * max(w, 0.)))
        total = f32(total + term)
    return bf16(total)


def topk(scores, k):
    selected = sorted(range(len(scores)), key=lambda p: (-scores[p], p))[:k]
    return sorted(selected)


def select_clusters(scores, bounds, seed_ids, k, cluster_size):
    seed_positions = [p for c in sorted(seed_ids)
                      for p in range(c * cluster_size, min((c + 1) * cluster_size, len(scores)))]
    seed_scores = [scores[p] for p in seed_positions]
    threshold = sorted(seed_scores, reverse=True)[k - 1] if len(seed_scores) >= k else -math.inf
    kept = [p for c, bound in enumerate(bounds) if bound >= threshold
            for p in range(c * cluster_size, min((c + 1) * cluster_size, len(scores)))]
    winners = sorted(kept, key=lambda p: (-scores[p], p))[:k]
    return sorted(winners), kept
