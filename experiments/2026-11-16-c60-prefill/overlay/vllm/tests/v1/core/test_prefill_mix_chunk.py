# SPARK_DS41F_PREFILL_MIX_CHUNK caps the prefill chunk below 256K only while decodes
# are runnable with prefill yield; unset keeps the default chunks exactly.
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched import scheduler as scheduler_module

Scheduler = scheduler_module.Scheduler
THRESHOLD = 16288


def threshold_at(computed, decodes, *, mix, prefill_yield=True):
    owner = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=THRESHOLD),
        prefill_yield=prefill_yield,
    )
    old = Scheduler._MIX_CHUNK_TIER
    Scheduler._MIX_CHUNK_TIER = mix
    try:
        return Scheduler._context_aware_chunk_threshold(owner, computed, decodes)
    finally:
        Scheduler._MIX_CHUNK_TIER = old


@pytest.mark.parametrize("computed", [0, 16288, 65536, 262143, 262144, 400000, 524288, 900000])
@pytest.mark.parametrize("decodes", [0, 3])
@pytest.mark.parametrize("prefill_yield", [True, False])
def test_unset_keeps_the_release(computed, decodes, prefill_yield):
    owner = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=THRESHOLD),
        prefill_yield=prefill_yield,
    )
    expected = THRESHOLD
    if prefill_yield and decodes > 0 and computed >= 262144:
        expected = 2048 if computed >= 524288 else 4096
    assert threshold_at(computed, decodes, mix=0, prefill_yield=prefill_yield) == expected
    assert Scheduler._MIX_CHUNK_TIER == 0
    assert Scheduler._context_aware_chunk_threshold(owner, computed, decodes) == expected


@pytest.mark.parametrize("mix", [8192, 4096])
def test_mix_caps_below_256k_beside_decodes(mix):
    for computed in (0, 8192, 65536, 262143):
        assert threshold_at(computed, 3, mix=mix) == mix


@pytest.mark.parametrize("mix", [8192, 4096])
def test_solo_prefill_and_no_yield_keep_full_chunks(mix):
    assert threshold_at(65536, 0, mix=mix) == THRESHOLD
    assert threshold_at(65536, 3, mix=mix, prefill_yield=False) == THRESHOLD


@pytest.mark.parametrize("mix", [8192, 4096])
def test_deep_context_keeps_the_release_tiers(mix):
    assert threshold_at(262144, 3, mix=mix) == 4096
    assert threshold_at(524288, 3, mix=mix) == 2048


def test_a_cap_at_or_above_the_threshold_changes_nothing():
    assert threshold_at(65536, 3, mix=16384) == THRESHOLD
    assert threshold_at(65536, 3, mix=32768) == THRESHOLD


@pytest.mark.parametrize("value,expected", [
    (None, 0), ("", 0), ("8192", 8192), ("4096", 4096), ("2048", 2048),
    ("1024", 0), ("5000", 0), ("-4096", 0), ("abc", 0),
])
def test_the_variable_parser(value, expected):
    assert scheduler_module._prefill_mix_chunk_tier(value) == expected


def test_aligned_chunks_end_on_the_mix_grid():
    owner = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=THRESHOLD),
        prefill_yield=True,
    )
    old = Scheduler._MIX_CHUNK_TIER
    Scheduler._MIX_CHUNK_TIER = 8192
    try:
        assert Scheduler._aligned_prefill_chunk(owner, 0, 69418, 3) == 8192
        assert Scheduler._aligned_prefill_chunk(owner, 8192, 61226, 3) == 8192
        assert Scheduler._aligned_prefill_chunk(owner, 65536, 3882, 3) == 3882
        assert Scheduler._aligned_prefill_chunk(owner, 0, 69418, 0) == THRESHOLD
    finally:
        Scheduler._MIX_CHUNK_TIER = old
