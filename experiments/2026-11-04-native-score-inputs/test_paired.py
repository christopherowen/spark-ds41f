"""The performance gate must reject increases and inconclusive results."""

import math

from paired import classify


def test_rejects_slow_and_uncertain_cases():
    assert classify([math.log(1.01)] * 40)["status"] == "not_proved"
    assert classify([math.log(0.98), math.log(1.01)] * 20)["status"] == "not_proved"


def test_accepts_clear_gain_and_verified_program_identity():
    assert classify([math.log(0.98)] * 40)["status"] == "better"
    assert (
        classify([math.log(1.01)] * 40, identical_program=True)["status"]
        == "equal_program"
    )
