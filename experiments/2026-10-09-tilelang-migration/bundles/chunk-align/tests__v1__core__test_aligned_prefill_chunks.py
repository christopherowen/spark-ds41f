# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aligned prefill chunks: a chunk ends at an absolute multiple of the threshold."""

import json
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched import scheduler as scheduler_module


def _chunk(threshold, computed, remaining):
    owner = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold)
    )
    return scheduler_module.Scheduler._aligned_prefill_chunk(owner, computed, remaining)


def test_chunks_end_at_multiples_of_the_threshold():
    assert _chunk(400, 0, 800) == 400
    assert _chunk(400, 400, 400) == 400
    assert _chunk(400, 0, 150) == 150  # the prompt ends first
    # A prefix-cache hit of 1536 tokens: the first chunk ends at 8096, as in a
    # cold run, not 8096 tokens later.
    assert _chunk(8096, 1536, 9000) == 6560
    assert _chunk(0, 1536, 5000) == 5000  # no threshold: unchanged


def test_a_whole_chunk_must_fit_beside_the_decode_rows():
    def check(threshold, budget, seqs=16, spec=5):
        owner = SimpleNamespace(
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold),
            max_num_running_reqs=seqs,
            num_spec_tokens=spec,
            max_num_scheduled_tokens=budget,
        )
        scheduler_module.Scheduler._check_aligned_prefill_chunks(owner)

    check(8096, 8192)  # TP4: 16 streams of six rows beside a chunk
    check(4048, 4096, seqs=8)  # TP3
    for threshold, budget in ((8192, 8192), (0, 8192), (8097, 8192)):
        with pytest.raises(ValueError):
            check(threshold, budget)


# A minimal OPT-125m configuration, so the scheduler builds without the hub.
_OPT_CONFIG = {
    "architectures": ["OPTForCausalLM"],
    "model_type": "opt",
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "ffn_dim": 3072,
    "max_position_embeddings": 2048,
    "vocab_size": 50272,
    "word_embed_proj_dim": 768,
    "torch_dtype": "float16",
    "do_layer_norm_before": True,
    "pad_token_id": 1,
    "bos_token_id": 2,
    "eos_token_id": 2,
}


def _scheduled(monkeypatch, tmp_path, aligned):
    # The scheduler fixtures live in the source tree's tests, not in every image.
    utils = pytest.importorskip("tests.v1.core.utils")
    create_requests, create_scheduler = utils.create_requests, utils.create_scheduler
    monkeypatch.setattr(scheduler_module, "ALIGN_PREFILL_CHUNKS", aligned)
    (tmp_path / "config.json").write_text(json.dumps(_OPT_CONFIG))
    scheduler = create_scheduler(
        model=str(tmp_path),
        max_num_batched_tokens=1024,
        long_prefill_token_threshold=400,
        skip_tokenizer_init=True,
        device="cpu",
    )
    requests = create_requests(num_requests=3, num_tokens=800)
    for request in requests:
        scheduler.add_request(request)
    return requests, scheduler.schedule().num_scheduled_tokens


def test_no_partial_chunk_when_the_budget_is_short(monkeypatch, tmp_path):
    requests, scheduled = _scheduled(monkeypatch, tmp_path, aligned=True)
    assert scheduled[requests[0].request_id] == 400
    assert scheduled[requests[1].request_id] == 400
    assert requests[2].request_id not in scheduled


def test_off_the_third_request_takes_what_is_left(monkeypatch, tmp_path):
    requests, scheduled = _scheduled(monkeypatch, tmp_path, aligned=False)
    assert scheduled[requests[2].request_id] == 224
