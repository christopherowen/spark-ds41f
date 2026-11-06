"""Strict performance preservation: compare independent boot-level observations.

A zero-crossing confidence interval is not evidence of preserved performance.
Use one scalar per independent boot, not per replay, with a predeclared margin.
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from servebench import stats


def compare_boots(reference, candidate, *, higher_is_better, margin_pct):
    if not math.isfinite(margin_pct) or margin_pct <= 0:
        raise ValueError("declare a positive material-regression bound")
    if min(len(reference), len(candidate)) < 2:
        return {
            "verdict": "inconclusive",
            "reason": "two independent boots per arm required",
        }
    if any(not math.isfinite(x) or x <= 0 for x in [*reference, *candidate]):
        raise ValueError("boot metrics must be finite and positive")
    comparison = stats.welch(reference, candidate, confidence=0.95)
    if comparison is None:
        return {"verdict": "inconclusive", "reason": "confidence interval unavailable"}
    change, half = comparison["change_pct"], comparison["ci95_pct"]
    if half is None or not math.isfinite(half):
        return {
            "verdict": "inconclusive",
            "reason": "finite confidence interval unavailable",
        }
    direction = -1 if higher_is_better else 1
    adverse = direction * change
    upper, lower = adverse + half, adverse - half
    pairings = [direction * 100 * (b / a - 1) for a in reference for b in candidate]
    preserved = upper <= margin_pct and max(pairings) <= margin_pct
    verdict = (
        "preserved"
        if preserved
        else "regression"
        if lower > margin_pct
        else "inconclusive"
    )
    return {
        "verdict": verdict,
        "margin_pct": margin_pct,
        "reference_boots": reference,
        "candidate_boots": candidate,
        "higher_is_better": higher_is_better,
        "change_pct": change,
        "ci95_half_width_pct": half,
        "adverse_ci95_lower_pct": lower,
        "adverse_ci95_upper_pct": upper,
        "all_pairings_adverse_pct": pairings,
        "reference_repeat_range_pct": 100 * (max(reference) / min(reference) - 1),
        "candidate_repeat_range_pct": 100 * (max(candidate) / min(candidate) - 1),
        "scope": "one observation per boot; no claim about missing workload cells",
    }
