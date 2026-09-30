#!/usr/bin/env python3
"""Send the same temperature-0 request twice, one after the other (checksum debug arm).

usage: checksum_requests.py BASE_URL [--tokens N] [--once]

--once sends a single request and prints its text, for dumping between requests.
"""
import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 128
PROMPT = "Return a JSON object describing three fictional planets with name, mass and moons."
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
ONCE = "--once" in sys.argv
outputs = []
for _ in range(1 if ONCE else 2):
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": TOKENS, "temperature": 0, "chat_template_kwargs": {"thinking": False},
            "cache_salt": uuid.uuid4().hex}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        outputs.append(json.load(response)["choices"][0]["message"]["content"])
if ONCE:
    print(json.dumps({"text": outputs[0]}))
else:
    print(json.dumps({"identical": outputs[0] == outputs[1], "lengths": [len(o) for o in outputs]}))
