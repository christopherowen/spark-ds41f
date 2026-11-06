"""Fault injection for the isolated hardware experiment, never production policy."""

import asyncio
import json


class DeadlineMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        headers = dict(scope.get("headers", ()))
        value = headers.get(b"x-experiment-deadline-ms")
        if scope["type"] != "http" or value is None:
            return await self.app(scope, receive, send)
        if scope.get("path") not in ("/v1/completions", "/v1/chat/completions"):
            raise ValueError("deadline injection is restricted to inference routes")
        milliseconds = int(value)
        if not 100 <= milliseconds <= 5000:
            raise ValueError("experiment deadline must be 100-5000 ms")
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        deadline = asyncio.timeout(milliseconds / 1000)
        try:
            async with deadline:
                await self.app(scope, receive, tracked_send)
        except TimeoutError:
            if not deadline.expired():
                raise
            if started:
                raise RuntimeError("fault injection requires a non-streaming response")
            body = json.dumps({"error": "experiment request deadline"}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 504,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
