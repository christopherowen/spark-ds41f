#!/usr/bin/env python3
"""Profile each request type the deployment serves, one torch-profiler window apiece
(a profiler-enabled arm must be serving).

usage: profile_decode.py BASE_URL [--tokens N]

Every scenario is warmed with one request of its own kind, then profiled alone between
/start_profile and /stop_profile; the last one profiles eight mixed requests at once.
Each window leaves one trace a rank, in the order printed here (one JSON line a
scenario: its name, the HTTP status and the completion tokens). A request the server
rejects is recorded with its status and message and is not profiled.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 48
PROMPT = "Return a JSON object describing three fictional planets with name, mass and moons."
STORY = "Write a short story about a lighthouse keeper who finds a message in a bottle."
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "moons": {"type": "integer"}},
          "required": ["name", "moons"]}
# name: (prompt, request fields); thinking is off unless a scenario turns it on.
SCENARIOS = {
    "greedy": (PROMPT, {"temperature": 0}),
    "sampled": (STORY, {"temperature": 1.0, "seed": 7}),
    "top-p": (STORY, {"temperature": 0.7, "top_p": 0.9, "seed": 7}),
    "top-k": (STORY, {"temperature": 1.0, "top_k": 40, "seed": 7}),
    "top-p-top-k": (STORY, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7}),
    "structured": (PROMPT, {"temperature": 0, "response_format": {
        "type": "json_schema", "json_schema": {"name": "planet", "schema": SCHEMA}}}),
    "logprobs": (PROMPT, {"temperature": 0, "logprobs": True, "top_logprobs": 5}),
    "prompt-logprobs": (PROMPT, {"temperature": 0, "prompt_logprobs": 1}),
    "penalties": (STORY, {"temperature": 0, "repetition_penalty": 1.2, "frequency_penalty": 0.3}),
    "stop": (STORY, {"temperature": 0, "stop": [".", "\n"]}),
    "thinking": (PROMPT, {"temperature": 0, "chat_template_kwargs": {"thinking": True},
                          "thinking_token_budget": 32}),
    "n2": (STORY, {"temperature": 1.0, "n": 2, "seed": 7}),
}
MIXED = ["greedy", "sampled", "top-p", "top-k", "structured", "logprobs", "penalties", "thinking"]


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        text = response.read()
        return json.loads(text) if text.startswith(b"{") else None


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def chat(name):
    prompt, fields = SCENARIOS[name]
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex, **fields}
    return post("/v1/chat/completions", body)


def profiled(label, work):
    post("/start_profile")
    try:
        result = work()
    finally:
        post("/stop_profile")
        time.sleep(30)  # the ranks write their traces before the next window starts
    return result


for name in SCENARIOS:
    try:
        chat(name)
    except urllib.error.HTTPError as error:
        print(json.dumps({"scenario": name, "status": error.code,
                          "message": error.read().decode(errors="replace")[:300]}), flush=True)
        continue
    result = profiled(name, lambda: chat(name))
    print(json.dumps({"scenario": name, "status": 200,
                      "completion_tokens": result["usage"]["completion_tokens"]}), flush=True)

results = [None] * len(MIXED)


def run(index):
    results[index] = chat(MIXED[index])


def mixed():
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(MIXED))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


mixed()
profiled("mixed-c8", mixed)
print(json.dumps({"scenario": "mixed-c8", "status": 200,
                  "completion_tokens": sum(r["usage"]["completion_tokens"] for r in results if r)}), flush=True)
