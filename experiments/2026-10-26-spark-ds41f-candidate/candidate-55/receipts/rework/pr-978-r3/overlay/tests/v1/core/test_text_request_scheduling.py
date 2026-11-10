# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-only requests beside the multimodal prefix-cache fast path.

A request without multimodal items must enter the engine and be scheduled exactly
as before the fast path: added by the engine's request handler with nothing around
it, never looked up by the stripped-item check, and scheduled with the same
prefix-cache hits, chunks, computed tokens and block counts, re-runs on cached
blocks included. The pinned schedule below was recorded on the 1107 vLLM stack
before the fast path (spark-ds41f candidate-54); a change that means to alter how
text requests are scheduled re-records it.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from vllm.multimodal.cache import MultiModalCacheMissError
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore, EngineCoreProc, EngineShutdownState
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

BLOCK = 16


class HandlerCore(EngineCore):
    """EngineCore with the engine process's request handler, around a scheduler."""

    _handle_client_request = EngineCoreProc._handle_client_request
    _reject_add_in_shutdown = EngineCoreProc._reject_add_in_shutdown


def make_core(scheduler) -> HandlerCore:
    core = HandlerCore.__new__(HandlerCore)
    core.scheduler = scheduler
    core.vllm_config = scheduler.vllm_config
    core.mm_receiver_cache = None  # --mm-processor-cache-gb 0
    core._loop_watchdog = None
    core._add_received = {}
    core._stall_warning_s = 0
    core.shutdown_state = EngineShutdownState.RUNNING
    return core


def make_scheduler():
    # Prefix caching with chunked prefill (a 64-token long-prefill chunk) and
    # async scheduling, as served.
    return create_scheduler(
        enable_prefix_caching=True,
        block_size=BLOCK,
        max_num_batched_tokens=256,
        long_prefill_token_threshold=64,
        async_scheduling=True,
        num_blocks=4096,
    )


_hasher = None


def text_request(
    req_id: str, tokens: list[int], salt: str | None, max_tokens=8, mm_features=None
):
    global _hasher
    if _hasher is None:
        init_none_hash(sha256)
        _hasher = get_request_block_hasher(BLOCK, sha256)
    params = SamplingParams(ignore_eos=True, max_tokens=max_tokens, temperature=0.0)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(tokens),
        sampling_params=params,
        pooling_params=None,
        cache_salt=salt,
        mm_features=mm_features,
        block_hasher=_hasher,
    )


def prompt(seed: int, length: int) -> list[int]:
    return [1000 + (seed * 7919 + i * 104729) % 30000 for i in range(length)]


def _next_token(history: list[int]) -> int:
    digest = hashlib.blake2b(json.dumps(history[-32:]).encode(), digest_size=4)
    return 1000 + int.from_bytes(digest.digest(), "little") % 30000


class StandInRunner:
    """The worker's side of a step: a prefill chunk samples nothing; the step that
    finishes a prompt, and every decode step, samples the token that the
    request's own history decides (equal histories, equal tokens). With sink
    rejection on, draft zero (the token held back last step) is verified and
    committed first."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.state: dict[str, list[int]] = {}
        self.provisional: dict[str, int] = {}

    def __call__(self, so) -> ModelRunnerOutput:
        for rid in so.finished_req_ids | (so.preempted_req_ids or set()):
            self.state.pop(rid, None)
            self.provisional.pop(rid, None)
        draft_zero = so.sink_draft_zero_req_ids or set()
        req_ids = list(so.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            request = self.scheduler.requests[rid]
            if rid in draft_zero:
                history = self.state[rid]
                held = self.provisional[rid]
                history.append(held)
                self.provisional[rid] = _next_token(history)
                sampled.append([held, self.provisional[rid]])
            elif request.is_prefill_chunk:
                sampled.append([])
            else:
                history = list(request.prompt_token_ids) + list(request.output_token_ids)
                self.state[rid] = history
                token = _next_token(history)
                if getattr(self.scheduler, "sink_rejection", False):
                    self.provisional[rid] = token
                else:
                    history.append(token)
                sampled.append([token])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )


# Phases run one after another; within a phase, (arrival step, request id, prompt
# seed, prompt length, cache salt). The long prompt (300 tokens) prefills in
# 64-token chunks.
WORKLOAD = [
    [(0, "a", 1, 100, "salt-a")],
    [(0, "a-rerun", 1, 100, "salt-a")],  # the same salt: a prefix-cache re-run
    [(0, "a-fresh", 1, 100, "salt-fresh")],  # a fresh salt, as the gate's "again"
    [(0, f"short-{i}", 10 + i, 20, f"salt-short-{i}") for i in range(3)]
    + [(2, "a-rerun-mixed", 1, 100, "salt-a")],  # a re-run among others
    [(0, "long", 2, 300, "salt-long")],
    [(0, "long-rerun", 2, 300, "salt-long")],
    [(0, "long-extended", 2, 340, "salt-long")],  # a longer prompt on its prefix
    [(0, f"again-{i}", 10 + i, 20, f"salt-again-{i}") for i in range(3)],
]


def run_workload():
    scheduler = make_scheduler()
    core = make_core(scheduler)
    runner = StandInRunner(scheduler)
    steps: list[list] = []
    admitted: dict[str, list] = {}
    for phase in WORKLOAD:
        pending = sorted(phase)
        active = {job[1] for job in phase}
        step = 0
        while pending or active:
            while pending and pending[0][0] <= step:
                _, rid, seed, length, salt = pending.pop(0)
                tokens = prompt(seed, length)
                if rid == "long-extended":
                    tokens = prompt(seed, 300) + prompt(99, length - 300)
                core._handle_client_request(
                    EngineCoreRequestType.ADD, (text_request(rid, tokens, salt), 0)
                )
            so = scheduler.schedule()
            cached = so.scheduled_cached_reqs
            for new in so.scheduled_new_reqs:
                admitted[new.req_id] = [new.num_computed_tokens, len(new.block_ids[0])]
            steps.append(
                [
                    sorted(so.num_scheduled_tokens.items()),
                    [[r.req_id, r.num_computed_tokens, len(r.block_ids[0])]
                     for r in so.scheduled_new_reqs],
                    [[rid, cached.num_computed_tokens[i],
                      0 if cached.new_block_ids[i] is None else len(cached.new_block_ids[i][0])]
                     for i, rid in enumerate(cached.req_ids)],
                    sorted(so.finished_req_ids),
                ]
            )
            outputs = scheduler.update_from_output(so, runner(so))
            for client in outputs.values():
                for out in client.outputs:
                    if out.finish_reason is not None:
                        active.discard(out.request_id)
            step += 1
            assert step < 1000
    return steps, admitted


# Recorded on candidate-54 (the 1107 stack before the fast path; see the module
# docstring). Per request: computed tokens when first scheduled (its prefix-cache
# hit) and the blocks its first step holds.
EXPECTED_ADMISSION = {
    "a": [0, 4],
    "a-rerun": [96, 7],
    "a-fresh": [0, 4],
    "short-0": [0, 2],
    "short-1": [0, 2],
    "short-2": [0, 2],
    "a-rerun-mixed": [96, 7],
    "long": [0, 4],
    "long-rerun": [288, 19],
    "long-extended": [288, 20],
    "again-0": [0, 2],
    "again-1": [0, 2],
    "again-2": [0, 2],
}
# Every step (73): scheduled tokens per request, new requests' computed tokens and
# blocks, running requests' computed tokens and new blocks, finished requests.
EXPECTED_SCHEDULE_SHA256 = (
    "ac32653b4a720e9ea6c2b882b79825ccf195e1e73d295a7ef7dc4f70defb3cf1"
)


def test_text_request_enters_the_engine_as_before(monkeypatch):
    scheduler = make_scheduler()
    core = make_core(scheduler)
    lookups = []
    lookup = scheduler.kv_cache_manager.get_computed_blocks

    def counted(request):
        lookups.append(request.request_id)
        return lookup(request)

    monkeypatch.setattr(scheduler.kv_cache_manager, "get_computed_blocks", counted)
    request = text_request("text", prompt(3, 40), "salt")
    core._handle_client_request(EngineCoreRequestType.ADD, (request, 0))
    assert scheduler.requests["text"] is request
    # Nothing looked its prefix up before the scheduler does.
    assert lookups == []
    scheduler.schedule()
    assert lookups == ["text"]

    # A failure while adding it surfaces unchanged: no multimodal-miss handling
    # wraps a request without multimodal items.
    misses = []
    monkeypatch.setattr(
        core, "_handle_mm_cache_miss", lambda req, err: misses.append(req), raising=False
    )

    def failing_add(req, request_wave=0):
        raise MultiModalCacheMissError(["not-an-item-of-this-request"])

    monkeypatch.setattr(core, "add_request", failing_add)
    with pytest.raises(MultiModalCacheMissError):
        core._handle_client_request(
            EngineCoreRequestType.ADD, (text_request("other", prompt(4, 40), None), 0)
        )
    assert misses == []


def test_stripped_item_miss_at_add_is_still_a_retryable_response(monkeypatch):
    """The other side: a request with an item sent without its data that the
    engine no longer holds is answered with a retryable miss (round 2)."""
    scheduler = make_scheduler()
    core = make_core(scheduler)
    misses = []
    monkeypatch.setattr(
        core,
        "_handle_mm_cache_miss",
        lambda req, err: misses.append(err.mm_hashes),
        raising=False,
    )
    def image_request(data):
        feature = MultiModalFeatureSpec(
            data=data,
            mm_position=PlaceholderRange(offset=20, length=100),
            identifier="image-1",
            modality="image",
            mm_hash="image-1",
        )
        return text_request("image", prompt(5, 400), None, mm_features=[feature])

    core._handle_client_request(EngineCoreRequestType.ADD, (image_request(None), 0))
    assert misses == [["image-1"]]
    assert "image" not in scheduler.requests
    # With its data it is scheduled.
    core._handle_client_request(
        EngineCoreRequestType.ADD, (image_request(MultiModalKwargsItem.dummy()), 0)
    )
    assert "image" in scheduler.requests


def test_text_scheduling_on_prefix_cache_reruns_is_candidate_54s():
    steps, admitted = run_workload()
    # Prefix-cache hits as candidate-54 computes them: whole 16-token blocks of
    # the cached prefix, short of the prompt's last token; none with a fresh salt
    # or a prompt shorter than a block's worth of cached tokens.
    assert admitted == EXPECTED_ADMISSION
    digest = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    assert digest == EXPECTED_SCHEDULE_SHA256


def test_requests_without_multimodal_attributes_take_the_earlier_paths():
    """Request stand-ins without mm_features (as tests/v1/engine/
    test_stall_diagnostics.py and test_admission_control.py build them) go
    through the engine's handler and AsyncLLM._add_request as before, on an
    engine with the fast path on."""
    added = []
    engine = SimpleNamespace(
        _loop_watchdog=None,
        _stall_warning_s=0,
        _add_received={},
        _reject_add_in_shutdown=lambda request: False,
        add_request=lambda request, wave: added.append((request.request_id, wave)),
    )
    EngineCoreProc._handle_client_request(
        engine, EngineCoreRequestType.ADD, (SimpleNamespace(request_id="engine"), 0)
    )
    assert added == [("engine", 0)]

    sent = []

    async def add_request_async(request):
        sent.append(request.request_id)

    llm = AsyncLLM.__new__(AsyncLLM)
    llm._mm_prefix_fastpath = True
    llm.engine_core = SimpleNamespace(add_request_async=add_request_async)
    llm.output_processor = SimpleNamespace(
        has_request=lambda request_id: False, add_request=lambda *args: None
    )
    llm.check_admission = lambda **kwargs: None
    llm.log_requests = False
    request = SimpleNamespace(request_id="frontend", prompt_token_ids=[1, 2, 3])
    asyncio.run(llm._add_request(request, None, None, 0, None))
    assert sent == ["frontend"]
