#!/usr/bin/env python3
"""Send one cold real-text prompt of about N tokens (this experiment's source text), one output token.

usage: send_prompt.py BASE_URL [--tokens N]   (prints the prompt's token count)
"""
import glob
import json
import os
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 3900
HERE = os.path.dirname(os.path.abspath(__file__))
text = ""
for path in sorted(glob.glob(os.path.join(HERE, "*.py"))):
    text += open(path).read() + "\n"
    if len(text) > TOKENS * 4:
        break
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": text[: int(TOKENS * 3.1)]}], "max_tokens": 1,
        "temperature": 0, "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex}
request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=900) as response:
    print(json.dumps({"prompt_tokens": json.load(response)["usage"]["prompt_tokens"]}))
