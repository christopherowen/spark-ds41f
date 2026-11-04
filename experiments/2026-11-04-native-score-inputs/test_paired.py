"""The performance gate must reject increases and inconclusive results."""

import math

from paired import classify, extend


def test_rejects_slow_and_uncertain_cases():
    assert classify([math.log(1.01)] * 40)["status"] == "not_proved"
    assert classify([math.log(0.98), math.log(1.01)] * 20)["status"] == "not_proved"


def test_accepts_clear_gain_and_verified_program_identity():
    assert classify([math.log(0.98)] * 40)["status"] == "better"
    assert (
        classify([math.log(1.01)] * 40, identical_program=True)["status"]
        == "equal_program"
    )


def test_more_precision_cannot_average_away_an_observed_increase():
    assert not extend([math.log(1.01)] * 120, [math.log(0.90)] * 480)["passed"]
    first = [math.log(0.98), math.log(1.019)] * 60
    result = extend(first, [math.log(0.99)] * 480)
    assert result["primary"]["status"] == "not_proved"
    assert result["passed"]
