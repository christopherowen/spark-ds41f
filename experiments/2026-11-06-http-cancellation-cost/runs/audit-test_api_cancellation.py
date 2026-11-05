# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import functools
from types import SimpleNamespace

import anyio
import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks

from vllm.entrypoints.serve.utils.api_utils import load_aware_call, with_cancellation


class Request:
    def __init__(self, state=None):
        self.app = SimpleNamespace(
            state=state
            or SimpleNamespace(enable_server_load_tracking=True, server_load_metrics=0)
        )
        self.state = SimpleNamespace()
        self.messages = asyncio.Queue()
        self.listening = asyncio.Event()
        self.listener_closed = asyncio.Event()

    async def receive(self):
        self.listening.set()
        try:
            return await self.messages.get()
        finally:
            # Cleanup may require an await (e.g. an engine abort).
            await asyncio.sleep(0)
            self.listener_closed.set()


def route(func):
    return with_cancellation(load_aware_call(func))


def run_async(test):
    """Run each case in its own loop, including a strict ownership check."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        async def run():
            before = asyncio.all_tasks()
            try:
                return await test(*args, **kwargs)
            finally:
                leaked = [
                    task for task in asyncio.all_tasks() - before if not task.done()
                ]
                for task in leaked:
                    task.cancel()
                await asyncio.gather(*leaked, return_exceptions=True)
                assert not leaked, [task.get_coro().__qualname__ for task in leaked]

        return asyncio.run(run())

    return wrapper


@run_async
@pytest.mark.parametrize("tracking", [False, True])
async def test_outer_cancellation_waits_for_both_children(tracking):
    request = Request()
    request.app.state.enable_server_load_tracking = tracking
    started, closed = asyncio.Event(), asyncio.Event()

    @route
    async def handle(_, raw_request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            closed.set()

    task = asyncio.create_task(handle(None, request))
    await started.wait()
    await request.listening.wait()
    task.cancel("caller timeout")
    with pytest.raises(asyncio.CancelledError, match="caller timeout"):
        await task
    assert closed.is_set()
    assert request.listener_closed.is_set()
    assert request.app.state.server_load_metrics == 0


@run_async
async def test_disconnect_waits_for_abort_before_returning():
    request = Request()
    started, abort_started, finish_abort, closed = (asyncio.Event() for _ in range(4))

    @route
    async def handle(_, raw_request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            abort_started.set()
            await finish_abort.wait()
            closed.set()

    task = asyncio.create_task(handle(None, request))
    await started.wait()
    await request.messages.put({"type": "http.disconnect"})
    await abort_started.wait()
    assert not task.done()
    finish_abort.set()
    assert await task is None
    assert closed.is_set()
    assert request.app.state.server_load_metrics == 0


@run_async
async def test_cancellation_scope_allows_async_cleanup():
    request = Request()
    started, closed = asyncio.Event(), asyncio.Event()

    @route
    async def handle(_, raw_request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            closed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(handle, None, request)
        await started.wait()
        await request.listening.wait()
        group.cancel_scope.cancel()
    assert closed.is_set()
    assert request.listener_closed.is_set()
    assert request.app.state.server_load_metrics == 0


@run_async
async def test_listener_error_propagates_and_cleans_handler():
    request = Request()
    closed = asyncio.Event()

    async def receive():
        await asyncio.sleep(0)
        raise OSError("receive failed")

    request.receive = receive

    @route
    async def handle(_, raw_request):
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            closed.set()

    with pytest.raises(OSError, match="receive failed"):
        await handle(None, request)
    assert closed.is_set()
    assert request.app.state.server_load_metrics == 0


@run_async
async def test_handler_error_preserved():
    request = Request()
    error = ValueError("invalid input")

    @route
    async def handle(_, raw_request):
        await raw_request.listening.wait()
        raise error

    with pytest.raises(ValueError) as caught:
        await handle(None, request)
    assert caught.value is error
    assert request.listener_closed.is_set()
    assert request.app.state.server_load_metrics == 0


@run_async
async def test_cancelled_handler_cleanup_error_is_retrieved():
    request = Request()
    started = asyncio.Event()
    errors = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: errors.append(context))

    @route
    async def handle(_, raw_request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            raise RuntimeError("abort failed")

    try:
        task = asyncio.create_task(handle(None, request))
        await started.wait()
        await request.messages.put({"type": "http.disconnect"})
        assert await task is None
        assert not errors
        assert request.app.state.server_load_metrics == 0
    finally:
        loop.set_exception_handler(previous)


@run_async
@pytest.mark.parametrize("background", ["none", "single", "multiple"])
@pytest.mark.parametrize("stream", [False, True])
async def test_response_handoff_keeps_background_and_bytes(background, stream):
    request = Request()
    calls = []

    async def record():
        calls.append("existing")

    bg = {
        "none": None,
        "single": BackgroundTask(record),
        "multiple": BackgroundTasks([BackgroundTask(record)]),
    }[background]

    async def chunks():
        yield 'data: {"text":"hello"}\n\n'
        yield "data: [DONE]\n\n"

    response = (
        StreamingResponse(chunks(), background=bg)
        if stream
        else JSONResponse({"text": "hello"}, background=bg)
    )

    @route
    async def handle(_, raw_request):
        await raw_request.listening.wait()
        # Both child tasks complete in the same event-loop turn.
        await raw_request.messages.put({"type": "http.disconnect"})
        return response

    assert await handle(None, request) is response
    assert request.listener_closed.is_set()
    assert request.app.state.server_load_metrics == 1
    if stream:
        assert [chunk async for chunk in response.body_iterator] == [
            'data: {"text":"hello"}\n\n',
            "data: [DONE]\n\n",
        ]
    else:
        assert response.body == b'{"text":"hello"}'
    await response.background()
    assert request.app.state.server_load_metrics == 0
    assert calls == ([] if background == "none" else ["existing"])


@run_async
async def test_four_long_requests_short_traffic_and_immediate_retry():
    state = SimpleNamespace(enable_server_load_tracking=True, server_load_metrics=0)
    active = set()
    starts = {i: asyncio.Event() for i in range(4)}
    requests = [Request(state) for _ in range(4)]
    # No token work is performed: these payloads exercise frontend ownership.
    payload = [[7] * (1048576 - 1024) for _ in range(4)]
    retained = {}

    @route
    async def handle(data, raw_request):
        key, prompt, block = data
        assert key not in active, "retry overlapped abandoned work"
        active.add(key)
        retained[key] = prompt
        try:
            if block:
                starts[key].set()
                await asyncio.Event().wait()
            return JSONResponse({"id": key, "tokens": len(prompt)})
        finally:
            await asyncio.sleep(0)
            active.remove(key)
            del retained[key]

    tasks = [
        asyncio.create_task(handle((i, payload[i], True), r))
        for i, r in enumerate(requests)
    ]
    await asyncio.gather(*(event.wait() for event in starts.values()))
    assert state.server_load_metrics == 4
    for i in range(16):
        response = await handle((100 + i, [1, 2], False), Request(state))
        assert response.status_code == 200
        await response.background()
    assert active == set(range(4))
    tasks[0].cancel()
    tasks[1].cancel()
    for i in (2, 3):
        await requests[i].messages.put({"type": "http.disconnect"})
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(x, asyncio.CancelledError) for x in results[:2])
    assert results[2:] == [None, None]
    assert not active and not retained
    assert state.server_load_metrics == 0
    for i in range(4):
        response = await handle((i, [7], False), Request(state))
        await response.background()
    assert not active and not retained
    assert state.server_load_metrics == 0


@run_async
async def test_cancel_during_response_handoff_releases_load():
    request = Request()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_closed = asyncio.Event()

    async def receive():
        request.listening.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()
            cleanup_closed.set()

    request.receive = receive

    @route
    async def handle(_, raw_request):
        await raw_request.listening.wait()
        return JSONResponse({"ok": True})

    task = asyncio.create_task(handle(None, request))
    await cleanup_started.wait()
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not task.done()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_closed.is_set()
        assert request.app.state.server_load_metrics == 0
    finally:
        finish_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
