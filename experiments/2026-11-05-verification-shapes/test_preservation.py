"""Reject wide zero-crossing intervals and preserve adverse boot pairings."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "preservation", Path(__file__).with_name("preservation.py")
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_zero_crossing_wide_interval_is_inconclusive(monkeypatch):
    monkeypatch.setattr(
        p.stats, "welch", lambda *a, **kw: {"change_pct": 0, "ci95_pct": 5}
    )
    assert (
        p.compare_boots([100, 100], [100, 100], higher_is_better=False, margin_pct=3)[
            "verdict"
        ]
        == "inconclusive"
    )


def test_tight_neutral_result_passes():
    result = p.compare_boots(
        [99.99, 100.01], [100.00, 100.02], higher_is_better=False, margin_pct=0.5
    )
    assert result["verdict"] == "preserved"
    assert len(result["all_pairings_adverse_pct"]) == 4


def test_direction_of_throughput_regression():
    result = p.compare_boots(
        [99.99, 100.01], [94.99, 95.01], higher_is_better=True, margin_pct=3
    )
    assert result["verdict"] == "regression"
    assert result["adverse_ci95_lower_pct"] > 3


def test_one_missing_reference_repeat_is_inconclusive():
    assert (
        p.compare_boots([100], [100, 101], higher_is_better=False, margin_pct=3)[
            "verdict"
        ]
        == "inconclusive"
    )


def test_no_bad_pairing_can_be_averaged_away(monkeypatch):
    monkeypatch.setattr(
        p.stats, "welch", lambda *a, **kw: {"change_pct": 0, "ci95_pct": 0.1}
    )
    assert (
        p.compare_boots([100, 100], [95, 105], higher_is_better=False, margin_pct=3)[
            "verdict"
        ]
        == "inconclusive"
    )


@pytest.mark.parametrize("values", [[float("nan"), 1], [0, 1], [float("inf"), 1]])
def test_invalid_observations_fail(values):
    with pytest.raises(ValueError):
        p.compare_boots(values, [1, 1], higher_is_better=False, margin_pct=3)
