# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What aligned prefill chunks do to concurrent requests (spark-ds41f's concurrency-stall study).

Copy to tests/v1/core/ in a vLLM tree (next-candidate) and run on CPU:
    python -m pytest tests/v1/core/test_concurrency_stalls.py
It pins the scheduler behaviour the study's simulator assumes:
- a long prompt's whole aligned chunk takes the step's budget beside the decode rows, so a
  short request that arrives while it prefills waits for its last chunk;
- with two prefill lanes (--max-parallel-prefills 2), each lane's share is half the budget;
  two long prompts' whole chunks never fit, neither lane is released, and nothing prefills.
"""

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
THRESHOLD = BUDGET - SEQS  # one row a decoding request (no drafts), as the recipe derives it


def _scheduler(monkeypatch, tmp_path, lanes=1):
    utils = pytest.importorskip("tests.v1.core.utils")
    # DeepSeek V4.1's configuration sets the flag; this OPT model takes it here.
    monkeypatch.setattr(utils, "SchedulerConfig",
                        functools.partial(utils.SchedulerConfig, align_prefill_chunks=True))
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = utils.create_scheduler(
        model=str(tmp_path),
        max_num_seqs=SEQS,
        max_num_batched_tokens=BUDGET,
        long_prefill_token_threshold=THRESHOLD,
        max_parallel_prefills=lanes,
        max_model_len=8192,
        skip_tokenizer_init=True,
        device="cpu",
    )
    return utils, scheduler


def test_a_short_request_waits_for_the_long_prompts_last_chunk(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path)
    long, = utils.create_requests(num_requests=1, num_tokens=4000, req_ids=["long"])
    scheduler.add_request(long)
    assert scheduler.schedule().num_scheduled_tokens == {"long": THRESHOLD}
    short, = utils.create_requests(num_requests=1, num_tokens=40, req_ids=["short"])
    scheduler.add_request(short)
    steps = []
    for _ in range(3):
        steps.append(dict(scheduler.schedule().num_scheduled_tokens))
    # Two more whole chunks with no room beside them, then the last (partial) chunk and the
    # short request together.
    assert steps[0] == {"long": THRESHOLD}
    assert steps[1] == {"long": THRESHOLD}
    assert steps[2] == {"long": 4000 - 3 * THRESHOLD, "short": 40}  # 976 + 40 <= 1,024


def test_two_lanes_never_schedule_two_long_prompts(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, lanes=2)
    for request in utils.create_requests(num_requests=2, num_tokens=4000, req_ids=["a", "b"]):
        scheduler.add_request(request)
    for _ in range(3):
        assert scheduler.schedule().total_num_scheduled_tokens == 0


def test_one_lane_schedules_them_in_turn(monkeypatch, tmp_path):
    utils, scheduler = _scheduler(monkeypatch, tmp_path, lanes=1)
    for request in utils.create_requests(num_requests=2, num_tokens=4000, req_ids=["a", "b"]):
        scheduler.add_request(request)
    assert scheduler.schedule().num_scheduled_tokens == {"a": THRESHOLD}
    assert scheduler.schedule().num_scheduled_tokens == {"a": THRESHOLD}
