#!/usr/bin/env python3
"""Exact laws of DSpark's block verification with dead rows (one-hot drafts), in FP64.

usage: exact_law.py [--vocab V] [--drafts K] [--seed S] [--kinds a,b,...]

For every dead-row cut c (0 to K - 1 live drafts of K) and every kind of dead-row
logits, the law of the committed tokens is computed exactly: summed over every draft
sequence (V^K, each draft drawn from q) and over every acceptance length (the uniform
draws integrated out). Two procedures:

- scan, cap: next-candidate's. Verify all K drafts, the rows past the cut included,
  then commit at most c + 1 tokens (the token at c is draft c when more than c drafts
  were accepted, else the residual or bonus there).
- truncated: block-verify-cut's. Verify the first c drafts as the whole block; the
  token at c is the bonus of row c.

Row s verifies draft s; rows 0 to c are live (the target p), the others hold the kind's
logits. Where non-finite values decide, the kernels' arithmetic is followed: a step's
log ratio is min(log_p + log_m, 0) with a NaN sum giving 0 (min.f32), and a threshold
whose denominator is not positive or is NaN is 1. A row's log-softmax is Eq. 5's
(Barron et al.): NaN for a row with NaN or with no finite maximum but an infinite one
it shares, 0 at a unique infinite maximum.

Every live row holds the same target, so the truth at every position is p. For each
position j <= c the script prints P(reach j), the same for both procedures wherever
the identity below holds, and the total variation between the law of the token at j
among the outcomes that reach j and p.

Block verification's thresholds satisfy prod_{j >= i} (1 - h_j) = 1 - p_i for any
probabilities the rows after i hold, so a block reaches step i with probability p_i,
a product of live rows' probabilities only. With finite dead rows the capped scan
therefore has the truncated block's law; a NaN draft probability breaks the identity
(its threshold is 1 and the ratio after it is 1 again).
"""

import argparse
import itertools
import math

import numpy as np

KINDS = ("random", "certain", "impossible", "inf", "noise", "nan")


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    if np.isnan(z).any():
        return np.full_like(z, np.nan)
    m = z.max()
    if m == np.inf:
        ties = int((z == m).sum())
        return np.where(z == m, 0.0 if ties == 1 else np.nan, -np.inf)
    if m == -np.inf:
        return np.full_like(z, np.nan)
    return z - m - math.log(np.exp(z - m).sum())


def log_ratio(log_p, log_m):
    s = log_p + log_m
    return 0.0 if math.isnan(s) else min(s, 0.0)


def threshold(log_p, log_m):
    s = log_p + log_m
    denom = math.nan if math.isnan(s) else -math.expm1(s)
    if not denom > 0:
        return 1.0
    return math.exp(log_p) * -math.expm1(log_m) / denom


def acceptance_law(log_m):
    """P(tau = t), t = 0 .. len(log_m): block verification's count from each step's
    draft log probability, with the kernels' thresholds."""
    k = len(log_m)
    log_p, cumulative = 0.0, []
    for lm in log_m:
        log_p = log_ratio(log_p, lm)
        cumulative.append(log_p)
    h = [threshold(cumulative[s], log_m[s + 1]) if s + 1 < k
         else math.exp(cumulative[s]) for s in range(k)]
    h = [min(max(x, 0.0), 1.0) for x in h]
    law = np.zeros(k + 1)
    for t in range(k + 1):
        law[t] = (h[t - 1] if t else 1.0) * np.prod([1 - h[j] for j in range(t, k)])
    return law


def without(p, token):
    r = p.copy()
    r[token] = 0.0
    return r / r.sum()


def dead_row(kind, noise, draft):
    """A dead row's logits; ``draft`` is the one it verifies (None for the bonus row)."""
    if kind == "nan":
        return np.full(noise.shape, np.nan)
    z = noise * (1e4 if kind == "noise" else 4.0)
    if draft is not None and kind in ("certain", "impossible", "inf"):
        z = z.copy()
        z[draft] = {"certain": 80.0, "impossible": -np.inf, "inf": np.inf}[kind]
    return z


def committed(log_rows, drafts, cut, procedure):
    """P(reach j) and P(token j = x, reach j) for j <= cut, given the drafts."""
    k = len(drafts)
    probs = [np.exp(r) for r in log_rows]
    v = probs[0].shape[0]
    reach = np.zeros(cut + 1)
    joint = np.zeros((cut + 1, v))
    steps = k if procedure == "scan" else min(k, cut)
    law = acceptance_law([log_rows[s][drafts[s]] for s in range(steps)])
    for tau, w in enumerate(law):
        if w == 0:
            continue
        last = min(tau, cut)
        for j in range(last):
            reach[j] += w
            joint[j, drafts[j]] += w
        reach[last] += w
        if tau > cut:
            joint[last, drafts[last]] += w  # the scan accepted it: the draft itself
        elif tau == steps:
            joint[last] += w * probs[last]  # the bonus
        else:
            joint[last] += w * without(probs[last], drafts[last])  # the residual
    return reach, joint


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vocab", type=int, default=5)
    parser.add_argument("--drafts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    v, k = args.vocab, args.drafts
    z = rng.normal(size=v) * 2.0
    p = np.exp(log_softmax(z))
    q = np.exp(log_softmax(z + rng.normal(size=v)))
    print(f"vocab {v}, {k} drafts, seed {args.seed}")
    print(f"target p {np.round(p, 4).tolist()}")
    print(f"drafts q {np.round(q, 4).tolist()} (one-hot acceptance {float(q @ p):.3f})")
    noise = [rng.normal(size=v) for _ in range(k + 1)]
    for kind in args.kinds.split(","):
        for cut in range(k):
            reach = {name: np.zeros(cut + 1) for name in ("scan", "truncated")}
            joint = {name: np.zeros((cut + 1, v)) for name in ("scan", "truncated")}
            for drafts in itertools.product(range(v), repeat=k):
                w = float(np.prod(q[list(drafts)]))
                log_rows = [
                    log_softmax(z) if s <= cut
                    else log_softmax(dead_row(kind, noise[s], drafts[s] if s < k else None))
                    for s in range(k + 1)
                ]
                for name in reach:
                    r, j = committed(log_rows, drafts, cut, name)
                    reach[name] += w * r
                    joint[name] += w * j
            cells = []
            for j in range(cut + 1):
                tv = {name: 0.5 * np.abs(joint[name][j] / reach[name][j] - p).sum()
                      for name in reach}
                cells.append(f"j{j} reach {reach['scan'][j]:.4f}/{reach['truncated'][j]:.4f}"
                             f" tv {tv['scan']:.1e}/{tv['truncated']:.1e}")
            print(f"{kind:>10} cut {cut}: " + "; ".join(cells))
    print("cells: reach scan/truncated, total variation from p scan/truncated")


if __name__ == "__main__":
    main()
