# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder inputs that arrive without their data (the multimodal prefix-cache
fast path): accepted because the engine held them, they may be evicted before
the encoder needs them. The scheduler must then finish the request with a
retryable miss, never schedule the encoder without data (the worker's
"Encoder cache miss" stops the engine)."""

import pytest

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

# DeepSeek V4.1 aligns prefill chunks, which needs a chunk size: one chunk
# holds a whole 400-token prompt.
CHUNK = 400


def make_scheduler(**kwargs):
    kwargs.setdefault("long_prefill_token_threshold", CHUNK)
    return create_scheduler(enable_prefix_caching=True, **kwargs)


def image_requests(ids, identifiers, positions, num_tokens=400):
    return create_requests(
        num_requests=len(ids),
        num_tokens=num_tokens,
        mm_positions=[list(positions)] * len(ids),
        mm_hashes_list=[list(idents) for idents in identifiers],
        req_ids=list(ids),
    )


def prefill_step_output(scheduler_output):
    # Every scheduled request is still prefilling: no sampled tokens.
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def outputs_of(engine_outputs):
    return [o for client in engine_outputs.values() for o in client.outputs]


def test_stripped_item_held_nowhere_is_a_retryable_miss():
    scheduler = make_scheduler()
    [request] = image_requests(
        ["stripped"], [["image-1"]], [PlaceholderRange(offset=20, length=100)]
    )
    request.mm_features[0].data = None
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert "stripped" not in output.num_scheduled_tokens
    assert not output.scheduled_encoder_inputs

    [miss] = outputs_of(
        scheduler.update_from_output(output, prefill_step_output(output))
    )
    assert miss.request_id == "stripped"
    assert miss.finish_reason == FinishReason.ERROR
    assert miss.mm_cache_miss_hashes == ["image-1"]
    assert "stripped" not in scheduler.requests
    assert not scheduler.stripped_mm_misses

    # The frontend resends it with its data under the same id.
    [again] = image_requests(
        ["stripped"], [["image-1"]], [PlaceholderRange(offset=20, length=100)]
    )
    scheduler.add_request(again)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["stripped"] == 400
    assert output.scheduled_encoder_inputs["stripped"] == [0]
    assert not outputs_of(
        scheduler.update_from_output(output, prefill_step_output(output))
    )


def test_stripped_item_in_the_encoder_cache_is_scheduled():
    scheduler = make_scheduler()
    position = [PlaceholderRange(offset=20, length=100)]
    holder, stripped = image_requests(
        ["holder", "stripped"], [["image-1"], ["image-1"]], position
    )
    # Distinct prompts (create_requests), so no prefix-cache hit either.
    stripped.mm_features[0].data = None

    scheduler.add_request(holder)
    output = scheduler.schedule()
    assert output.scheduled_encoder_inputs["holder"] == [0]
    scheduler.update_from_output(output, prefill_step_output(output))

    scheduler.add_request(stripped)
    output = scheduler.schedule()
    # Its encoder output is cached: no encoder run, no miss.
    assert output.num_scheduled_tokens["stripped"] == 400
    assert "stripped" not in output.scheduled_encoder_inputs
    assert not scheduler.stripped_mm_misses


def test_chunked_prefill_reaching_an_evicted_stripped_item_is_a_miss():
    # The first chunk covers the first image (sent with data). The second needs
    # the stripped second image, which nothing holds any more.
    # Aligned chunks: the budget holds a 200-token chunk and 16 decode rows.
    scheduler = make_scheduler(
        max_num_batched_tokens=216, long_prefill_token_threshold=200
    )
    [request] = image_requests(
        ["chunked"],
        [["image-a", "image-b"]],
        [
            PlaceholderRange(offset=20, length=100),
            PlaceholderRange(offset=250, length=100),
        ],
    )
    request.mm_features[1].data = None
    scheduler.add_request(request)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens["chunked"] == 200
    assert output.scheduled_encoder_inputs["chunked"] == [0]
    assert not outputs_of(
        scheduler.update_from_output(output, prefill_step_output(output))
    )

    output = scheduler.schedule()
    assert "chunked" not in output.num_scheduled_tokens
    [miss] = outputs_of(
        scheduler.update_from_output(output, prefill_step_output(output))
    )
    assert miss.request_id == "chunked"
    assert miss.finish_reason == FinishReason.ERROR
    assert miss.mm_cache_miss_hashes == ["image-b"]
    assert "chunked" not in scheduler.requests
    assert not scheduler.running


def test_a_miss_does_not_hold_other_requests():
    scheduler = make_scheduler()
    position = [PlaceholderRange(offset=20, length=100)]
    stripped, other = image_requests(
        ["stripped", "other"], [["image-1"], ["image-2"]], position
    )
    stripped.mm_features[0].data = None
    scheduler.add_request(stripped)
    scheduler.add_request(other)

    output = scheduler.schedule()
    [miss] = outputs_of(
        scheduler.update_from_output(output, prefill_step_output(output))
    )
    assert miss.request_id == "stripped"
    # At most one step later, the other request runs.
    if "other" not in output.num_scheduled_tokens:
        output = scheduler.schedule()
    assert output.num_scheduled_tokens["other"] == 400
    assert output.scheduled_encoder_inputs["other"] == [0]
