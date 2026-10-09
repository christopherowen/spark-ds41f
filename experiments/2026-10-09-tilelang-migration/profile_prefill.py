#!/usr/bin/env python3
"""Profile one cold long prefill (a profiler-enabled arm must be serving).

usage: profile_prefill.py BASE_URL [--tokens N]   (N: approximate prompt tokens, default 16384)

The prompt is this experiment's own source text (real code and prose, the
same in every arm), cut to about N tokens; one generated token, a fresh cache
salt. A short warm-up request runs first; the profiled request is wrapped in
/start_profile and /stop_profile. Prints the prompt's token count.
"""
import glob
import json
import os
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 16384
HERE = os.path.dirname(os.path.abspath(__file__))


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response) if response.headers.get("content-type", "").startswith("application/json") else None


text = ""
for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
    text += open(path).read() + "\n"
    if len(text) > TOKENS * 4:
        break
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]


def body(content):
    return {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex}


post("/v1/chat/completions", body("Say hello."))
# Calibrate the cut to the tokenizer once, without profiling: about 3.3 characters per token for code.
probe = post("/v1/chat/completions", body(text[: TOKENS * 3]))["usage"]["prompt_tokens"]
cut = int(TOKENS * 3 * TOKENS / probe)
post("/start_profile")
result = post("/v1/chat/completions", body(text[:cut]))
post("/stop_profile")
print(json.dumps({"prompt_tokens": result["usage"]["prompt_tokens"]}))
