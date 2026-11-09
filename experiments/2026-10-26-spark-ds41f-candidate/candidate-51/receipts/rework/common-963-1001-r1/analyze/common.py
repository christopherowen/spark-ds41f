# Shared rules for candidate-51's own-mark analyses.
# Interval rule (as candidate-49's own marks): two container boots per arm; a cell is
# "faster" when the slower candidate boot beats the faster base boot, "slower" when the
# faster candidate boot is slower than the slower base boot, else "same".
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def verdict(base, cand):
    """base, cand: two per-boot values each (lower is better)."""
    b = [x for x in base if x is not None]
    c = [x for x in cand if x is not None]
    if len(b) < 2 or len(c) < 2:
        return "n/a"
    if max(c) < min(b):
        return "faster"
    if min(c) > max(b):
        return "slower"
    return "same"


def fmt(base, cand):
    b = "/".join(f"{x:.2f}" for x in base)
    c = "/".join(f"{x:.2f}" for x in cand)
    rb = sum(base) / len(base)
    rc = sum(cand) / len(cand)
    return f"{b} -> {c} ({rb / rc:.3f}x, {verdict(base, cand)})"


def ratio(base, cand):
    return round((sum(base) / len(base)) / (sum(cand) / len(cand)), 4)
