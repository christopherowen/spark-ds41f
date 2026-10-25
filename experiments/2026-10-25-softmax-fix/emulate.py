#!/usr/bin/env python3
"""CPU emulation of the logprob and verification arithmetic, before and after the
softmax fix, against an FP64 reference: the README's error sizes.

usage: uvx --with numpy python emulate.py [--rows N] [--seed S]

FP32 is emulated with numpy float32, each operation rounded as the kernels round it:

- today (the Triton kernels and the release's TileLang ports of them): every column's
  exp(x - max) summed in 1024-column blocks (a thread's eight columns in register
  order, the warp at XOR distances 16 to 1, then (w0 + w2) + (w1 + w3)), the block
  sums added to 0 in block order, log(sum), and (x - max) - lse; verification's lse
  is max + log(sum) and its threshold p (1 - m) / (p (1 - m) + 1 - p);
- fixed (vLLM branch softmax-fix): the same sums without the maximum's column, the
  ties beyond the first added once, log1p(S), and shifted - log1p(S) with the
  shifted logit exactly 0 at the maximum; the threshold from expm1 of the logs.

ex2.approx is emulated as exp rounded to FP32 (the hardware's is within 2 ULPs), libdevice's
logf/log1pf/expm1f as correctly rounded. The reference is Eq. 5 in FP64 on the same
FP32 logits. One JSON line per experiment.
"""

import argparse
import json
import math

import numpy as np

F = np.float32
LOG2E = F(1.4426950408889634)
BLOCK = 1024
THREADS = 128


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32)


def exp32(x):
    """tl.exp: the product with log2(e) rounded, then 2^t rounded."""
    t = f32(f32(x) * LOG2E)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return f32(np.exp2(t.astype(np.float64)))


def block_sum(terms):
    """One 1024-column block in Triton's order (one element a thread, 128 apart)."""
    t = terms.reshape(8, THREADS)  # element j of thread tx is column j * 128 + tx
    acc = t[0].copy()
    for j in range(1, 8):
        acc = f32(acc + t[j])
    lanes = acc.reshape(4, 32)
    for d in (16, 8, 4, 2, 1):
        lanes = f32(lanes + lanes[:, np.arange(32) ^ d])
    w = lanes[:, 0]
    return f32(f32(w[0] + w[2]) + f32(w[1] + w[3]))


def row_sums(terms):
    padded = np.zeros(-(-terms.size // BLOCK) * BLOCK, dtype=np.float32)
    padded[: terms.size] = terms
    se = F(0)
    for b in range(padded.size // BLOCK):
        se = f32(se + block_sum(padded[b * BLOCK : (b + 1) * BLOCK]))
    return se


def today(row, ids):
    m = row.max()
    with np.errstate(invalid="ignore"):
        se = row_sums(exp32(f32(row - m)))
        lse = f32(math.log(float(se))) if se > 0 else F(-np.inf)
        return f32(f32(row[ids] - m) - lse)


def fixed(row, ids):
    m = row.max()
    tie = row == m
    with np.errstate(invalid="ignore"):
        terms = np.where(tie, F(0), exp32(f32(row - m)))
        s = row_sums(terms)
        n = int(tie.sum())
        if n > 1:
            s = f32(s + F(n - 1))
        lse = f32(math.log1p(float(s)))
        shifted = np.where(row[ids] == m, F(0), f32(row[ids] - m))
        return f32(shifted - lse)


def reference(row, ids):
    r = row.astype(np.float64)
    m = r.max()
    tie = r == m
    s = np.exp(r[~tie] - m).sum() + (tie.sum() - 1)
    shifted = np.where(r[ids] == m, 0.0, r[ids] - m)
    return shifted - math.log1p(s), s


def rel(a, ref):
    return abs(float(a) - ref) / abs(ref) if ref != 0 else (0.0 if a == 0 else math.inf)


def paper_example():
    row = f32([9.0, -9.0, -9.0])
    ids = np.array([0, 1])
    ref, s = reference(row, ids)
    return {
        "experiment": "paper z = [9, -9, -9]",
        "reference": ref.tolist(),
        "today": today(row, ids).tolist(),
        "fixed": fixed(row, ids).tolist(),
        "one_minus_p_reference": s / (1 + s),
        "one_minus_p_fixed": float(-np.expm1(np.float64(fixed(row, ids)[0]))),
    }


def margins(vocab, seed):
    """The top token's log probability by margin over the others' log-sum-exp, the
    argmax at the first, middle and last column (today's loss depends on how many
    blocks come after the maximum's)."""
    g = np.random.default_rng(seed)
    others = f32(g.normal(0.0, 2.0, vocab))
    lse_others = float(np.logaddexp.reduce(others.astype(np.float64)))
    out = []
    for margin in (4, 8, 10, 12, 13, 14, 15, 16, 16.3, 16.6, 16.9, 17.5, 18, 20, 25, 30,
                   36.7, 40, 60, 80, 87, 95, 100):
        for where in ("first", "middle", "last"):
            col = {"first": 0, "middle": vocab // 2, "last": vocab - 1}[where]
            row = others.copy()
            row[col] = F(lse_others + margin)
            ids = np.array([col])
            ref, s = reference(row, ids)
            t, x = today(row, ids)[0], fixed(row, ids)[0]
            out.append({"margin": margin, "argmax": where, "reference": float(ref[0]),
                        "today": float(t), "fixed": float(x),
                        "today_rel_error": rel(t, ref[0]), "fixed_rel_error": rel(x, ref[0])})
    return {"experiment": "margins", "vocab": vocab, "rows": out}


# The tests' allowance for the fixed sum's error over the release's (the two sums round
# independently; tests/v1/worker/test_gpu_tilelang_logprob_softmax.py S_ALLOWANCE).
S_ALLOWANCE = 4e-6


def random_rows(vocab, rows, seed):
    """Normal logits (sd 2, 4 and 8) and twenty random tokens plus the argmax: the
    largest error of each, and the largest excess of the fixed result's error over
    today's in the tests' allowance: an ULP of the output and of the log-sum-exp,
    and S_ALLOWANCE of S / (1 + S) (both round the same shifted logit; their sums of
    exponentials round independently)."""
    g = np.random.default_rng(seed)
    out = []
    for sd in (2.0, 4.0, 8.0):
        worst_today = worst_fixed = excess = 0.0
        worse = total = 0
        for _ in range(rows):
            row = f32(g.normal(0.0, sd, vocab))
            ids = np.concatenate([[int(row.argmax())], g.integers(0, vocab, 20)])
            ref, s = reference(row, ids)
            t, x = today(row, ids), fixed(row, ids)
            et = np.abs(t.astype(np.float64) - ref)
            ex = np.abs(x.astype(np.float64) - ref)
            worst_today, worst_fixed = max(worst_today, et.max()), max(worst_fixed, ex.max())
            allowance = (float(np.spacing(F(math.log1p(s))))
                         + np.spacing(np.abs(f32(ref))) + S_ALLOWANCE * s / (1 + s))
            worse += int((ex > et + allowance).sum())
            excess = max(excess, float(((ex - et) / allowance).max()))
            total += ex.size
        out.append({"sd": sd, "max_abs_error_today": worst_today,
                    "max_abs_error_fixed": worst_fixed,
                    "fixed_worse_beyond_allowance": worse,
                    "max_excess_in_allowances": excess, "outputs": total})
    return {"experiment": "random rows", "vocab": vocab, "rows": rows, "by_sd": out}


def threshold_today(lp, lm):
    p, m = exp32(F(lp)), exp32(F(lm))
    rest = f32(F(1) - m)
    residual = f32(p * rest)
    denom = f32(f32(np.float64(p) * np.float64(rest) + 1.0) - p)  # fma, then - p
    return f32(residual / denom) if denom > 0 else F(1)


def threshold_fixed(lp, lm):
    residual = f32(exp32(F(lp)) * f32(-np.expm1(np.float64(F(lm)))))
    denom = f32(-np.expm1(np.float64(f32(F(lp) + F(lm)))))
    return f32(np.float64(residual) / np.float64(denom)) if denom > 0 else F(1)


def threshold_reference(lp, lm):
    lp, lm = float(F(lp)), float(F(lm))
    den = -math.expm1(lp + lm)
    return math.exp(lp) * -math.expm1(lm) / den if den > 0 else 1.0


def thresholds():
    """Block verification's h = p (1 - m) / (1 - p m) over the logs of a prefix ratio
    and a next-draft probability, from confident to unsure."""
    grid = [0.0, -1e-9, -3e-8, -1e-7, -6e-7, -1e-6, -1e-5, -1e-4, -1e-3, -0.01, -0.1,
            -0.7, -3.0]
    out = []
    worst_today = worst_fixed = 0.0
    for lp in grid:
        for lm in grid:
            ref = threshold_reference(lp, lm)
            t, x = float(threshold_today(lp, lm)), float(threshold_fixed(lp, lm))
            et, ex = abs(t - ref), abs(x - ref)
            worst_today, worst_fixed = max(worst_today, et), max(worst_fixed, ex)
            if et > 1e-3 or ex > 1e-6:
                out.append({"log_p": lp, "log_m": lm, "reference": ref, "today": t,
                            "fixed": x})
    return {"experiment": "block threshold", "max_abs_error_today": worst_today,
            "max_abs_error_fixed": worst_fixed, "notable": out}


def verify_log_prob(vocab, seed):
    """The draft's log probability in verification: today's max + log(sum) rounded at
    the magnitude of the maximum, by logit scale (temperature-scaled logits reach
    50)."""
    g = np.random.default_rng(seed)
    out = []
    for top in (8.0, 20.0, 40.0):
        for margin in (10.0, 14.0, 17.0, 25.0):
            others = f32(g.normal(0.0, 2.0, vocab))
            others = f32(others - F(np.logaddexp.reduce(others.astype(np.float64)))
                         + F(top - margin))
            row = others.copy()
            row[7] = F(top)
            ref, s = reference(row, np.array([7]))
            m = row.max()
            with np.errstate(invalid="ignore"):
                total = row_sums(exp32(f32(row - m)))
            lse = f32(m + f32(math.log(float(total))))
            t = f32(row[7] - lse)
            x = fixed(row, np.array([7]))[0]
            out.append({"top_logit": top, "margin": margin, "reference": float(ref[0]),
                        "today": float(t), "fixed": float(x)})
    return {"experiment": "verification log probability", "rows": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--vocab", type=int, default=129280)
    args = parser.parse_args()
    for record in (paper_example(), margins(args.vocab, args.seed),
                   random_rows(args.vocab, args.rows, args.seed + 1), thresholds(),
                   verify_log_prob(args.vocab, args.seed + 2)):
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
