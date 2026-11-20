# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the 1107 buffered admission patch."""

import asyncio
import os
import queue
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    ATOMIC_TEXT_BATCH_LIMIT,
    EngineCoreRequest,
    validate_atomic_text_batch,
)
from vllm.v1.engine import EngineCoreRequestType as Kind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def make(i):
    return EngineCoreRequest(
        str(i),
        [1, 2, 3],
        None,
        SamplingParams(temperature=0, max_tokens=16, seed=42),
        None,
        0.0,
        None,
        None,
        None,
    )


class Collector:
    def __init__(self, i):
        self.i = i
        self.closed = False

    def get_nowait(self):
        return RequestOutput(str(self.i), None, [1, 2, 3], None, [], True)

    def close(self):
        self.closed = True


async def proof():
    assert not torch.cuda.is_available()
    for name in (
        "vllm.v1.engine",
        "vllm.v1.engine.async_llm",
        "vllm.v1.engine.core",
        "vllm.v1.engine.core_client",
        "vllm.entrypoints.openai.chat_completion.batch_serving",
    ):
        assert (
            Path(sys.modules[name].__file__)
            .resolve()
            .is_relative_to(Path(os.environ["VLLM_ATOMIC_SOURCE"]).resolve() / "vllm")
        )
    assert (
        "atomic_results"
        in OpenAIServingChatBatch.chat_completion_full_generator_batch.__annotations__
    )
    sizes = tuple(sorted({1, 4, 8, ATOMIC_TEXT_BATCH_LIMIT}))
    for size in sizes:
        request = [make(i) for i in range(size)]
        frames = []

        async def send(kind, items):
            frames.append((kind, MsgpackEncoder().encode(items)))

        sender = SimpleNamespace(
            client_index=2, _send_input=send, _ensure_output_queue_task=lambda: None
        )
        await AsyncMPClient.add_atomic_text_batch_async(sender, request)
        assert len(frames) == 1 and frames[0][0] == Kind.ADD_ATOMIC_TEXT_BATCH
        decoded = MsgpackDecoder(list[EngineCoreRequest]).decode(frames[0][1])
        validate_atomic_text_batch(decoded)
        assert [r.request_id for r in decoded] == [str(i) for i in range(size)]
        assert all(
            r.client_index == 2 and r.sampling_params.seed == 42 for r in decoded
        )
        added = []
        receiver = SimpleNamespace(
            _loop_watchdog=None,
            _add_received={},
            _stall_warning_s=0,
            _reject_add_in_shutdown=lambda r: False,
            add_request=lambda r, wave: added.append(r.request_id),
        )
        receiver._handle_client_request = MethodType(
            EngineCoreProc._handle_client_request, receiver
        )
        receiver._handle_client_request(frames[0][0], [(r, 0) for r in decoded])
        assert added == [str(i) for i in range(size)]
    for invalid in (
        [],
        [make(i) for i in range(ATOMIC_TEXT_BATCH_LIMIT + 1)],
        [make(1), make(1)],
    ):
        try:
            validate_atomic_text_batch(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid batch accepted")

    for failure in ("none", "second_prompt", "disconnect"):
        queues = []
        sent = []
        aborted = []

        async def add(i, prompt, params, _deferred_atomic):
            if failure == "second_prompt" and i == 1:
                raise ValueError("fixture failure")
            _deferred_atomic.append(make(i))
            q = Collector(i)
            queues.append(q)
            return q

        async def submit(reqs):
            sent.append([r.request_id for r in reqs])

        async def abort(ids, internal):
            assert internal
            aborted.extend(ids)

        fixture = SimpleNamespace(
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=1)
            ),
            check_admission=lambda n: None,
            add_request=add,
            engine_core=SimpleNamespace(add_atomic_text_batch_async=submit),
            abort=abort,
        )
        g = AsyncLLM.generate_atomic_text_batch(
            fixture, [(None, None, i, {}) for i in range(4)]
        )
        if failure == "second_prompt":
            try:
                await anext(g)
            except ValueError:
                pass
            else:
                raise AssertionError("partial group returned")
            assert not sent and aborted == ["0"]
        elif failure == "disconnect":
            assert (await anext(g))[0] == 0
            await g.aclose()
            assert sent == [["0", "1", "2", "3"]] and aborted == ["0", "1", "2", "3"]
        else:
            assert [i async for i, _ in g] == [0, 1, 2, 3]
            assert sent == [["0", "1", "2", "3"]] and not aborted
        assert queues and all(q.closed for q in queues)

    # Exercise the real HTTP response assembly with no legacy generators. The
    # simulated atomic collector yields out of order; choices must still be
    # complete and ordered by their original conversation, with native counts.
    async def atomic_results():
        for i, text in ((1, "PASS"), (0, "READY")):
            c = CompletionOutput(
                index=0,
                text=text,
                token_ids=[i + 8],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
            yield i, RequestOutput(str(i), None, [1, 2, 3], None, [c], True)

    handler = SimpleNamespace(
        response_role="assistant",
        system_fingerprint=None,
        _raise_if_error=lambda *args: None,
    )
    request = SimpleNamespace(
        logprobs=False,
        include_reasoning=True,
        add_generation_prompt=True,
        echo=False,
        return_token_ids=True,
    )
    metadata = SimpleNamespace(final_usage_info=None)
    response = await OpenAIServingChatBatch.chat_completion_full_generator_batch(
        handler,
        request,
        [],
        "fixture",
        "fixture",
        [[], []],
        None,
        metadata,
        atomic_results=atomic_results(),
    )
    assert [(c.index, c.message.content, c.token_ids) for c in response.choices] == [
        (0, "READY", [8]),
        (1, "PASS", [9]),
    ]
    assert response.usage.prompt_tokens == 6 and response.usage.completion_tokens == 2

    if hasattr(EngineCoreProc, "_collect_tp8_background"):

        def item(s):
            return (Kind.ADD, (SimpleNamespace(request_id=s), 0))

        bg = item("chatcmpl-tp8bg-fixture")
        fg = item("chatcmpl-tp8fg-fixture")
        abort_item = (Kind.ABORT, ["chatcmpl-tp8bg-fixture"])
        check = EngineCoreProc._is_tp8_background
        assert check(bg) and not check(fg) and not check(abort_item)
        for arrived, expected in (
            ([bg] * 9, [bg] * 8),
            ([bg, fg, bg], [bg, bg, fg]),
            ([abort_item, bg], [bg, abort_item]),
            ([], [bg]),
        ):
            inbox = queue.Queue()
            for x in arrived:
                inbox.put(x)
            instance = SimpleNamespace(input_queue=inbox, _is_tp8_background=check)
            assert EngineCoreProc._collect_tp8_background(instance, bg) == expected


def test_wire_dispatch_and_buffered_cleanup():
    asyncio.run(proof())


def test_preprocessing_failure_never_exposes_partial_group():
    seen = []

    def preprocess(request):
        seen.append(request.request_id)
        if request.request_id == "1":
            raise ValueError("second member cannot be prepared")
        return request, 0

    fixture = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
        preprocess_add_request=preprocess,
    )
    import pytest

    with pytest.raises(ValueError, match="second member"):
        EngineCoreProc.preprocess_atomic_text_batch(
            fixture, [make(i) for i in range(4)]
        )
    assert seen == ["0", "1"]


def test_sender_rejects_invalid_group_before_any_wire_message():
    async def run():
        sent = []

        async def send(*args):
            sent.append(args)

        sender = SimpleNamespace(
            client_index=0,
            _send_input=send,
            _ensure_output_queue_task=lambda: None,
        )
        import pytest

        with pytest.raises(ValueError, match="distinct"):
            await AsyncMPClient.add_atomic_text_batch_async(sender, [make(0), make(0)])
        assert sent == []

    asyncio.run(run())


def test_deferred_registration_preserves_original_id_rollback(monkeypatch):
    import pytest
    import vllm.v1.engine.async_llm as module

    class Queue:
        def __init__(self, output_kind, request_id):
            self.request_id = request_id
            self.closed = False
            queues.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "RequestOutputCollector", Queue)

    async def one(failure, deferred):
        nonlocal queues
        queues = []
        registered = []
        sent = []
        aborted = []

        def assign(request):
            request.request_id = "internal-" + request.request_id

        def add_local(request, prompt, parent, index, queue):
            registered.append(request.request_id)
            if failure == "registration":
                raise ValueError("registration error")

        def admission(*args, **kwargs):
            if failure == "admission":
                raise ValueError("admission error")

        async def send(request):
            sent.append(request.request_id)
            if failure == "send":
                raise ValueError("send error")
            if failure == "cancel":
                raise asyncio.CancelledError

        async def abort(ids, internal):
            assert internal
            aborted.append(ids)
            if failure == "abort":
                raise ValueError("cleanup send error")

        fixture = SimpleNamespace(
            errored=False,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
            ),
            input_processor=SimpleNamespace(assign_request_id=assign),
            _run_output_handler=lambda: None,
            check_admission=admission,
            output_processor=SimpleNamespace(
                has_request=lambda request_id: False, add_request=add_local
            ),
            engine_core=SimpleNamespace(add_request_async=send),
            abort=abort,
            log_requests=False,
        )
        fixture._add_request = MethodType(AsyncLLM._add_request, fixture)
        request = make(5)
        if failure == "validator":
            request.sampling_params.max_tokens = 257
        if failure == "n":
            request.sampling_params.n = 2
        pending = [] if deferred else None
        kwargs = {"_deferred_atomic": pending} if deferred else {}
        if failure == "none":
            queue = await AsyncLLM.add_request(
                fixture, request.request_id, request, request.sampling_params, **kwargs
            )
            assert queue.request_id == "internal-5" and not queue.closed
            assert registered == ["internal-5"] and not aborted
            assert sent == ([] if deferred else ["internal-5"])
            assert pending == ([request] if deferred else None)
        else:
            error = (
                asyncio.CancelledError
                if failure == "cancel"
                else module.VLLMValidationError
                if failure in ("validator", "n")
                else ValueError
            )
            with pytest.raises(error):
                await AsyncLLM.add_request(
                    fixture,
                    request.request_id,
                    request,
                    request.sampling_params,
                    **kwargs,
                )
            assert queues and queues[0].closed
            assert aborted == ["internal-5"]
            if deferred:
                assert sent == []

    queues = []
    for failure in ("none", "validator", "n", "admission", "registration"):
        asyncio.run(one(failure, True))
    for failure in ("none", "admission", "registration", "send", "cancel"):
        asyncio.run(one(failure, False))


def test_trace_reports_actual_graph_mode_and_rows():
    from unittest.mock import patch

    from vllm.compilation.cuda_graph import CUDAGraphStat
    from vllm.v1.metrics.stats import SchedulerIterationDetails

    details = SchedulerIterationDetails(
        iteration_index=12,
        num_ctx_requests=0,
        num_ctx_tokens=0,
        num_generation_requests=8,
        num_generation_tokens=48,
        elapsed_ms=3,
    )
    graph = CUDAGraphStat(
        num_unpadded_tokens=48,
        num_padded_tokens=48,
        num_paddings=0,
        runtime_mode="FULL",
    )
    output = SimpleNamespace(
        scheduler_stats=SimpleNamespace(cudagraph_stats=graph, iteration_details=None)
    )
    with patch("vllm.v1.engine.core.logger.info") as log:
        EngineCoreProc._attach_iteration_details(
            SimpleNamespace(), {0: output}, details
        )
    args = log.call_args.args
    assert args[1:] == (12, 0, 0, 8, 48, "FULL", 48, 48)
    assert output.scheduler_stats.iteration_details is details
