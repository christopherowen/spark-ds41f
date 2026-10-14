"""Small-sample statistics without numpy: Student t intervals, Welch comparisons,
nearest-rank percentiles."""

import math
import statistics

# Two-sided 95% Student t quantiles by degrees of freedom; linear in 1/df between rows.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def t95(df: float) -> float:
    """The 0.975 quantile of Student's t with ``df`` degrees of freedom (1.96 as df grows)."""
    if df <= 1:
        return _T95[1]
    keys = sorted(_T95)
    if df >= keys[-1]:
        # Interpolate in 1/df toward the normal quantile.
        return 1.960 + (_T95[120] - 1.960) * 120 / df
    for low, high in zip(keys, keys[1:]):
        if low <= df <= high:
            weight = (1 / low - 1 / df) / (1 / low - 1 / high)
            return _T95[low] + weight * (_T95[high] - _T95[low])
    return 1.960


def describe(values: list[float]) -> dict:
    """Mean, sample sd and the 95% t interval of the mean (absolute and % of the mean)."""
    values = [v for v in values if v is not None]
    n = len(values)
    if not n:
        return {"n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "min": min(values), "max": max(values),
            "ci95": half, "ci95_pct": None if half is None or not mean else 100 * half / abs(mean)}


def percentiles(values: list[float], points=(50, 90, 99)) -> dict:
    """Nearest-rank percentiles and the mean."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    out = {"n": len(values), "mean": statistics.fmean(values)}
    for p in points:
        rank = max(1, math.ceil(p / 100 * len(values)))
        out[f"p{p}"] = values[rank - 1]
    return out


def t_quantile(p: float, df: float) -> float:
    """Student's t quantile, from the normal one by the Cornish-Fisher expansion (within
    about 0.1% of the exact value for df >= 10)."""
    z = statistics.NormalDist().inv_cdf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def trim_slow(values: list[float], k: float = 5.0) -> tuple[list[float], int]:
    """Drop slow outliers (above the median by more than ``k`` scaled MADs): stalls
    outside the work being timed. Returns the kept values and how many were dropped."""
    if len(values) < 5:
        return list(values), 0
    middle = statistics.median(values)
    mad = statistics.median(abs(v - middle) for v in values) * 1.4826
    if not mad:
        return list(values), 0
    kept = [v for v in values if v <= middle + k * mad]
    return kept, len(values) - len(kept)


def welch(base: list[float], candidate: list[float], confidence: float = 0.95) -> dict | None:
    """Relative change of the candidate's mean against the base's, with its Welch
    interval at ``confidence`` (95% by default), both in % of the base mean; ``se_pct``
    is the change's standard error in the same units."""
    a = [v for v in base if v is not None]
    b = [v for v in candidate if v is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    if not ma:
        return None
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2 / (len(a) - 1) if va else 0) + (vb ** 2 / (len(b) - 1) if vb else 0)) \
        if (va or vb) else len(a) + len(b) - 2
    quantile = t95(df) if confidence == 0.95 else t_quantile(1 - (1 - confidence) / 2, max(df, 1.0))
    return {"change_pct": 100 * (mb / ma - 1), "ci95_pct": 100 * quantile * se / abs(ma),
            "se_pct": 100 * se / abs(ma), "df": df}


def verdict(change_pct: float, ci95_pct: float | None, higher_is_better: bool,
            tolerance_pct: float) -> str:
    """better / worse when the interval excludes zero; with no interval, against the
    tolerance and marked so; else same."""
    better = change_pct > 0 if higher_is_better else change_pct < 0
    if ci95_pct is None:
        if abs(change_pct) <= tolerance_pct:
            return "same (no interval)"
        return ("better" if better else "worse") + " (no interval)"
    if abs(change_pct) <= ci95_pct:
        return "same"
    return "better" if better else "worse"
