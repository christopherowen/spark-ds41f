"""Small fixed-shape workloads for attributing profiled engine costs."""

import json
import urllib.request
import uuid

from . import engine, prom


def one_row(base_url: str, *, requests: int = 32, prompt_token: int = 1) -> dict:
    """Profile cold single-token target forwards, without chat-template padding.

    A fresh cache salt per request ensures the token is computed, even with prefix
    caching. This is a cost probe, not a real-text throughput benchmark.
    """
    if requests < 24:
        raise ValueError("one-row profiling needs at least 24 engine iterations")
    base = base_url.rstrip("/")
    prom.require_idle(base, "one-row profiling")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]

    def forward():
        result = engine._post(base + "/v1/completions", {
            "model": model, "prompt": [prompt_token], "max_tokens": 1,
            "temperature": 0, "cache_salt": uuid.uuid4().hex,
        }, 120)
        if result["usage"]["prompt_tokens"] != 1:
            raise RuntimeError("one-row profile received a padded prompt")
        return result

    for _ in range(2):
        forward()
    engine._post(base + "/start_profile", None, 120)
    try:
        for _ in range(requests):
            forward()
    finally:
        engine._post(base + "/stop_profile", None, 120)
    return {"requests": requests, "prompt_tokens_per_request": 1}
