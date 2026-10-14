"""One streamed chat request, timed chunk by chunk.

The record keeps every streamed chunk's arrival time with the request's cumulative output
tokens at that moment, so a measurement window can count exactly the tokens produced
inside it. Servers that report usage on every chunk (vLLM's ``continuous_usage_stats``)
give exact counts; otherwise the final count is spread over the chunks in order and the
record says so (``token_times: "approximate"``).
"""

import json
import time
import urllib.request

TIMEOUT_S = 900.0


def stream_chat(base_url: str, payload: dict, *, continuous_usage: bool = False,
                timeout: float = TIMEOUT_S, clock=time.monotonic) -> dict:
    """Send ``payload`` to ``base_url``/v1/chat/completions as a stream and time it.

    Returns a record: ``sent``, ``first`` (first chunk carrying output), ``end`` (clock
    readings), ``events`` ([time, cumulative output tokens] per output chunk),
    ``prompt_tokens``, ``completion_tokens``, ``finish_reason``, ``token_times`` and
    ``error`` (None when the request completed)."""
    options = {"include_usage": True}
    if continuous_usage:
        options["continuous_usage_stats"] = True
    body = dict(payload, stream=True, stream_options=options)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    record = {"sent": clock(), "first": None, "end": None, "events": [],
              "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
              "token_times": "exact", "error": None}
    counted = True  # every output chunk so far carried a cumulative count
    usage: dict = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                event = json.loads(line[6:])
                now = clock()
                if event.get("error"):  # a server error sent inside the stream (HTTP 200)
                    detail = event["error"]
                    record["error"] = "server: " + (detail.get("message") if isinstance(detail, dict) else str(detail))[:300]
                    break
                usage = event.get("usage") or usage
                emitted = False
                for choice in event.get("choices") or []:
                    record["finish_reason"] = choice.get("finish_reason") or record["finish_reason"]
                    delta = choice.get("delta") or {}
                    for key in ("content", "reasoning_content", "reasoning"):
                        if isinstance(delta.get(key), str) and delta[key]:
                            emitted = True
                if not emitted:
                    continue
                if record["first"] is None:
                    record["first"] = now
                tokens = (event.get("usage") or {}).get("completion_tokens")
                if tokens is None:
                    counted = False
                record["events"].append([now, tokens])
    except Exception as exception:  # noqa: BLE001 - every failure belongs in the record
        record["error"] = f"{type(exception).__name__}: {exception}"
    record["end"] = clock()
    record["prompt_tokens"] = usage.get("prompt_tokens")
    record["completion_tokens"] = usage.get("completion_tokens")
    if record["error"] is None and record["first"] is None:
        record["error"] = "no output"
    events = record["events"]
    if events and not counted:
        record["token_times"] = "approximate"
        total = record["completion_tokens"] or len(events)
        for index, event in enumerate(events):
            event[1] = round(total * (index + 1) / len(events))
    return record


def tokens_at(record: dict, moment: float) -> int:
    """Output tokens the request had produced by ``moment`` (0 before its first chunk)."""
    produced = 0
    for at, cumulative in record["events"]:
        if at > moment:
            break
        produced = cumulative or produced
    return produced
