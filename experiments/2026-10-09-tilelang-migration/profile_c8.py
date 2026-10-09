#!/usr/bin/env python3
"""Profile an eight-stream decode window with its verification counters.

usage: profile_c8.py BASE_URL [--case json|prose] [--streams N] [--tokens N]
(a profiler-enabled arm must be serving)

Runs one warm-up round, snapshots the spec-decode counters, wraps one round of
N concurrent fixed requests (thinking off for json) in /start_profile and
/stop_profile, snapshots the counters again, and prints one JSON line: wall
time, completion tokens, draft events, verified drafts and accepted drafts in
the window, verified and accepted drafts per draft event, and tokens/s. The
prompts are fixed, so both arms of a comparison decode the same requests.
"""
import json
import re
import sys
import threading
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
CASE = sys.argv[sys.argv.index("--case") + 1] if "--case" in sys.argv else "json"
STREAMS = int(sys.argv[sys.argv.index("--streams") + 1]) if "--streams" in sys.argv else 8
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 256
PROMPTS = {
    "json": [f"Return a JSON object describing {n} fictional {thing} with name and three attributes."
             for n, thing in zip(range(3, 11), ("planets", "cities", "ships", "rivers", "books",
                                                "birds", "machines", "islands"))],
    "prose": [f"Write a paragraph about {topic}." for topic in (
        "the history of the lighthouse", "a night market", "the first railway", "glaciers",
        "a violin maker", "tidal pools", "desert navigation", "an old printing press")],
}[CASE]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]
COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total")


def counters():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as response:
        text = response.read().decode()
    out = {}
    for name in COUNTERS:
        out[name] = sum(float(m) for m in re.findall(rf"^{re.escape(name)}{{[^}}]*}} ([0-9.e+]+)$", text, re.M))
    return out


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        data = response.read()
        return json.loads(data) if data[:1] == b"{" else None


def one(prompt, results):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "temperature": 0, "cache_salt": uuid.uuid4().hex}
    if CASE == "json":
        body["chat_template_kwargs"] = {"thinking": False}
    results.append(post("/v1/chat/completions", body)["usage"]["completion_tokens"])


def round_():
    results = []
    threads = [threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], results)) for i in range(STREAMS)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.monotonic() - start, sum(results)


round_()
before = counters()
post("/start_profile")
wall, tokens = round_()
post("/stop_profile")
after = counters()
d = {k.split(":")[1]: after[k] - before[k] for k in COUNTERS}
drafts = d["spec_decode_num_drafts_total"]
print(json.dumps({
    "case": CASE, "streams": STREAMS, "wall_s": round(wall, 3), "completion_tokens": tokens,
    "tokens_per_s": round(tokens / wall, 2), "draft_events": drafts,
    "verified_drafts": d["spec_decode_num_draft_tokens_total"],
    "accepted_drafts": d["spec_decode_num_accepted_tokens_total"],
    "verified_per_draft": round(d["spec_decode_num_draft_tokens_total"] / drafts, 4) if drafts else None,
    "accepted_per_draft": round(d["spec_decode_num_accepted_tokens_total"] / drafts, 4) if drafts else None,
}))
