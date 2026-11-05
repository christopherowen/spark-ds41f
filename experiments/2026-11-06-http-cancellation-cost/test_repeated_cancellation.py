import asyncio
import pytest
from test_api_cancellation import Request, route, run_async

@run_async
@pytest.mark.parametrize("repeats", [1, 3])
async def test_repeated_cancellation_waits_for_abort(repeats):
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

    parent = asyncio.create_task(handle(None, request))
    await started.wait()
    await request.listening.wait()
    parent.cancel("first timeout")
    await abort_started.wait()
    for i in range(repeats):
        parent.cancel(f"repeat timeout {i}")
        await asyncio.sleep(0)
    try:
        await asyncio.sleep(0.01)
        assert not parent.done(), "second cancellation interrupted owned abort cleanup"
        finish_abort.set()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert closed.is_set()
        assert request.listener_closed.is_set()
        assert request.app.state.server_load_metrics == 0
    finally:
        finish_abort.set()
        await asyncio.gather(parent, return_exceptions=True)


@run_async
async def test_repeated_cancellation_during_response_handoff():
    from fastapi.responses import JSONResponse
    request = Request()
    cleanup_started, finish_cleanup, closed = (asyncio.Event() for _ in range(3))

    async def receive():
        request.listening.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await finish_cleanup.wait()
            closed.set()

    request.receive = receive

    @route
    async def handle(_, raw_request):
        await raw_request.listening.wait()
        return JSONResponse({"ok": True})

    parent = asyncio.create_task(handle(None, request))
    await cleanup_started.wait()
    try:
        for i in range(3):
            parent.cancel(f"handoff timeout {i}")
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        assert not parent.done()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert closed.is_set()
        assert request.app.state.server_load_metrics == 0
    finally:
        finish_cleanup.set()
        await asyncio.gather(parent, return_exceptions=True)
