#!/usr/bin/env python3
"""Profile one single-stream decode request (a profiler-enabled arm must be serving).

usage: profile_decode.py BASE_URL [--tokens N]

Warms up with the same request, then wraps a second one in /start_profile and
/stop_profile. The request is the determinism probe's JSON prompt with
thinking off, so drafts are accepted at a steady rate.
"""
import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 128
PROMPT = "Return a JSON object describing three fictional planets with name, mass and moons."


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response) if response.headers.get("content-type", "").startswith("application/json") else None


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": TOKENS,
        "temperature": 0, "chat_template_kwargs": {"thinking": False}}
post("/v1/chat/completions", dict(body, cache_salt=uuid.uuid4().hex))
post("/start_profile")
result = post("/v1/chat/completions", dict(body, cache_salt=uuid.uuid4().hex))
post("/stop_profile")
print(json.dumps({"completion_tokens": result["usage"]["completion_tokens"]}))
