# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill yield: a long prompt's aligned chunk gives its step to short prompts and to
held first tokens, and keeps its chunk boundaries."""

import functools
import json

import pytest

_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 8192,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}

BUDGET = 1024
SEQS = 16
THRESHOLD = BUDGET - SEQS


def _scheduler(
    monkeypatch,
    tmp_path,
    prefill_yield="short",
    limit_s=10.0,
    interleave_steps=0,
):
    utils = pytest.importorskip("tests.v1.core.utils")
    # DeepSeek V4.1's configuration sets align_prefill_chunks; this OPT model takes
    # it, and the yield option, here.
    monkeypatch.setattr(
        utils,
        "SchedulerConfig",
        functools.partial(
            utils.SchedulerConfig,
            align_prefill_chunks=True,
            prefill_yield=prefill_yield,
            prefill_yield_limit_s=limit_s,
            prefill_decode_interleave_steps=interleave_steps,
        ),
    )
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = utils.create_scheduler(
        model=str(tmp_path),
        max_num_seqs=SEQS,
        max_num_batched_tokens=BUDGET,
        long_prefill_token_threshold=THRESHOLD,
        max_model_len=8192,
        skip_tokenizer_init=True,
        device="cpu",
    )
    return utils, scheduler


def _request(utils, req_id, tokens):
    (request,) = utils.create_requests(num_requests=1, num_tokens=tokens, req_ids=[req_id])
    return request


def _steps(scheduler, count):
    return [dict(scheduler.schedule().num_scheduled_tokens) for _ in range(count)]


def test_off_keeps_first_come_first_served(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, prefill_yield="off")
    scheduler.add_request(_request(utils, "long", 4000))
    assert _steps(scheduler, 1) == [{"long": THRESHOLD}]
    scheduler.add_request(_request(utils, "short", 200))
    # The short request waits for the long prompt's whole prefill.
    assert _steps(scheduler, 3) == [
        {"long": THRESHOLD},
        {"long": THRESHOLD},
        {"long": 4000 - 3 * THRESHOLD},
    ]


def test_a_short_prompt_takes_the_step_and_the_chunks_keep_their_boundaries(
    monkeypatch, tmp_path
):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    scheduler.add_request(_request(utils, "long", 4000))
    assert _steps(scheduler, 1) == [{"long": THRESHOLD}]
    scheduler.add_request(_request(utils, "short", 200))
    steps = _steps(scheduler, 4)
    assert steps[0] == {"short": 200}
    # The long prompt resumes where it stopped: its chunks still end at multiples of
    # the threshold.
    assert steps[1:] == [
        {"long": THRESHOLD},
        {"long": THRESHOLD},
        {"long": 4000 - 3 * THRESHOLD},
    ]


def test_a_long_prompt_at_the_head_does_not_hold_up_a_short_one(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    scheduler.add_request(_request(utils, "long", 4000))
    scheduler.add_request(_request(utils, "short", 200))
    steps = _steps(scheduler, 2)
    assert steps == [{"short": 200}, {"long": THRESHOLD}]


def test_shorts_keep_their_order(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    scheduler.add_request(_request(utils, "long", 4000))
    for name in ("a", "b", "c"):
        scheduler.add_request(_request(utils, name, 300))
    (step,) = _steps(scheduler, 1)
    assert step == {"a": 300, "b": 300, "c": 300}


def test_nothing_short_waiting_changes_nothing(monkeypatch, tmp_path):
    utils, on = _scheduler(monkeypatch, tmp_path)
    for name in ("x", "y"):
        on.add_request(_request(utils, name, 3000))
    assert _steps(on, 4) == [
        {"x": THRESHOLD},
        {"x": THRESHOLD},
        {"x": 3000 - 2 * THRESHOLD},
        {"y": THRESHOLD},
    ]


def test_the_held_first_token_step_carries_no_long_chunk(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    scheduler.sink_rejection = True  # set by DeepSeek V4.1's sink detector
    scheduler.add_request(_request(utils, "long", 4000))
    assert _steps(scheduler, 1) == [{"long": THRESHOLD}]
    scheduler.add_request(_request(utils, "short", 200))
    assert _steps(scheduler, 1) == [{"short": 200}]
    assert "short" in scheduler._first_decode_due
    # The short prompt's first decode step is due: the long chunk waits.
    assert "long" not in scheduler.schedule().num_scheduled_tokens


def test_a_long_prompt_stops_yielding_at_its_limit(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, limit_s=1e-9)
    scheduler.sink_rejection = True
    scheduler.add_request(_request(utils, "long", 4000))
    assert _steps(scheduler, 1) == [{"long": THRESHOLD}]
    scheduler.add_request(_request(utils, "short", 200))
    assert _steps(scheduler, 1) == [{"short": 200}]
    # Its yield began in that step and is past the limit now.
    assert scheduler.schedule().num_scheduled_tokens.get("long") == THRESHOLD


def test_yield_excludes_lanes_and_cadence():
    from vllm.config import SchedulerConfig

    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=8192, is_encoder_decoder=False,
            prefill_yield="short", max_parallel_prefills=2,
        )
    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=8192, is_encoder_decoder=False,
            prefill_yield="short", prefill_schedule_interval=2,
        )
    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=8192, is_encoder_decoder=False,
            prefill_decode_interleave_steps=2, max_parallel_prefills=2,
        )
    with pytest.raises(ValueError):
        SchedulerConfig(
            max_model_len=8192, is_encoder_decoder=False,
            prefill_decode_interleave_steps=2, prefill_schedule_interval=2,
        )


def _step_and_advance_decodes(scheduler):
    output = scheduler.schedule()
    scheduled = dict(output.num_scheduled_tokens)
    for req_id in scheduled:
        req = scheduler.requests[req_id]
        if not req.is_prefill_chunk:
            req.append_output_token_ids(0)
    return scheduled


def test_interleave_k_graphed_decode_steps_between_full_prefill_chunks(
    monkeypatch, tmp_path
):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, prefill_yield="off", interleave_steps=2
    )
    dec = _request(utils, "dec", 20)
    dec.max_tokens = 100
    scheduler.add_request(dec)
    assert _step_and_advance_decodes(scheduler) == {"dec": 20}

    long_req = _request(utils, "long", 3000)
    long_req.max_tokens = 10
    scheduler.add_request(long_req)
    # Chunk 1 of long + dec, then 2 pure decode steps, Chunk 2 + dec, 2 pure
    # decode steps, Chunk 3 (final) + dec.
    steps = [_step_and_advance_decodes(scheduler) for _ in range(7)]
    assert steps == [
        {"dec": 1, "long": THRESHOLD},
        {"dec": 1},
        {"dec": 1},
        {"dec": 1, "long": THRESHOLD},
        {"dec": 1},
        {"dec": 1},
        {"dec": 1, "long": 3000 - 2 * THRESHOLD},
    ]


def test_interleave_works_at_full_slot_occupancy_stall_15(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, prefill_yield="off", interleave_steps=2
    )
    for idx in range(SEQS - 1):
        req = _request(utils, f"d{idx}", 10)
        req.max_tokens = 100
        scheduler.add_request(req)
    first_step = _step_and_advance_decodes(scheduler)
    assert len(first_step) == SEQS - 1

    long_req = _request(utils, "long", 2500)
    long_req.max_tokens = 10
    scheduler.add_request(long_req)
    # All 16 sequence slots are now full (15 decodes + 1 long prefill).
    step1 = _step_and_advance_decodes(scheduler)
    assert step1["long"] == THRESHOLD
    assert len(step1) == SEQS

    step2 = _step_and_advance_decodes(scheduler)
    step3 = _step_and_advance_decodes(scheduler)
    assert "long" not in step2 and len(step2) == SEQS - 1
    assert "long" not in step3 and len(step3) == SEQS - 1

    step4 = _step_and_advance_decodes(scheduler)
    assert step4["long"] == THRESHOLD
    assert len(step4) == SEQS


def test_interleave_admits_waiting_short_prefill_during_decode_step(
    monkeypatch, tmp_path
):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, prefill_yield="off", interleave_steps=2
    )
    dec = _request(utils, "dec", 20)
    dec.max_tokens = 100
    scheduler.add_request(dec)
    assert _step_and_advance_decodes(scheduler) == {"dec": 20}

    long_req = _request(utils, "long", 3000)
    long_req.max_tokens = 10
    scheduler.add_request(long_req)
    assert _step_and_advance_decodes(scheduler) == {"dec": 1, "long": THRESHOLD}

    # Queue both a second long request and a short request during chunk 1:
    # the interleaved step skips long2, admits short, and keeps long's chunk
    # boundaries intact.
    long2 = _request(utils, "long2", 3000)
    short = _request(utils, "short", 200)
    short.max_tokens = 50
    scheduler.add_request(long2)
    scheduler.add_request(short)

    assert _step_and_advance_decodes(scheduler) == {"dec": 1, "short": 200}
    assert _step_and_advance_decodes(scheduler) == {"dec": 1, "short": 1}
    assert _step_and_advance_decodes(scheduler) == {
        "dec": 1,
        "short": 1,
        "long": THRESHOLD,
    }


def test_interleave_does_not_pause_when_no_decodes_are_runnable(
    monkeypatch, tmp_path
):
    utils, scheduler = _scheduler(
        monkeypatch, tmp_path, prefill_yield="off", interleave_steps=2
    )
    scheduler.add_request(_request(utils, "long", 3000))
    assert _steps(scheduler, 3) == [
        {"long": THRESHOLD},
        {"long": THRESHOLD},
        {"long": 3000 - 2 * THRESHOLD},
    ]
