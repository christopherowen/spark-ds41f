#!/usr/bin/env python3
"""Profile one cold long prefill (a profiler-enabled arm must be serving).

usage: profile_prefill.py BASE_URL [--tokens N]   (N: prompt tokens, default 16384)

The prompt is servebench's prefill document (Python standard-library source from a fixed
seed, sized by the server's tokenizer), so every arm reads the same text the release
suite's prefill reads. A short request and one unprofiled prefill of the same document
(its own cache salt) warm the prefill kernels; the profiled request, with a fresh cache
salt and one generated token, is wrapped in /start_profile and /stop_profile. Prints
the prompt's token count and both requests' times to first token.
"""
import json
import os
import sys
import time
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from servebench import prefill  # noqa: E402

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 16384


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response) if response.headers.get("content-type", "").startswith("application/json") else None


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
text, counted = prefill.document(BASE, model, TOKENS, seed=7, vllm=True)


def timed(content):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex}
    began = time.monotonic()
    result = post("/v1/chat/completions", body)
    return result, time.monotonic() - began


timed("Say hello.")
_, warm = timed(f"{text}\n\n{prefill.QUESTION}")
post("/start_profile")
result, profiled = timed(f"{text}\n\n{prefill.QUESTION}")
post("/stop_profile")
print(json.dumps({"prompt_tokens": result["usage"]["prompt_tokens"], "warm_s": round(warm, 3),
                  "profiled_s": round(profiled, 3)}))
