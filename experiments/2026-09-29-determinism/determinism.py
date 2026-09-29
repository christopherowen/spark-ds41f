#!/usr/bin/env python3
"""Where do identical temperature-0 requests start to differ?

usage: determinism.py BASE_URL [--repeats N] [--tokens N] [--shared-cache] [--save PATH]

Each prompt is sent N times at one stream with the same parameters. By default
every request gets a fresh cache salt, so each run prefills the whole prompt the
same way; --shared-cache reuses one salt (later runs hit the prefix cache).
Per prompt it reports distinct outputs, the first token index where each run
departs from run 0, and the largest chosen-token logprob difference before that
point (0 means bit-identical logits up to the flip). --save writes run 0's
tokens and logprobs per prompt, for comparing arms with each other.
"""
import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")


def option(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


REPEATS = option("--repeats", 5)
TOKENS = option("--tokens", 256)
SHARED = "--shared-cache" in sys.argv
SAVE = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
PROMPTS = {
    "prose": "Explain in a few paragraphs why the sky is blue at noon and red at sunset.",
    "code": "Write a Python function that merges overlapping intervals, with a docstring and tests.",
    "json": "Return a JSON object describing three fictional planets with name, mass and moons.",
}


def request(model, prompt, salt):
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": TOKENS, "temperature": 0, "logprobs": True, "top_logprobs": 2,
        "chat_template_kwargs": {"thinking": False}, "cache_salt": salt,
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        choice = json.load(response)["choices"][0]
    items = choice["logprobs"]["content"]
    return [(item["token"], item["logprob"],
             item["top_logprobs"][1]["logprob"] if len(item["top_logprobs"]) > 1 else None)
            for item in items]


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
shared_salt = uuid.uuid4().hex
report = {}
saved = {}
for name, prompt in PROMPTS.items():
    runs = [request(model, prompt, shared_salt if SHARED else uuid.uuid4().hex)
            for _ in range(REPEATS)]
    base = runs[0]
    saved[name] = base
    rows = []
    for index, run in enumerate(runs[1:], start=1):
        first = next((i for i, (a, b) in enumerate(zip(base, run)) if a[0] != b[0]),
                     min(len(base), len(run)))
        drift = max((abs(a[1] - b[1]) for a, b in zip(base[:first], run[:first])), default=0.0)
        first_drift = next((i for i, (a, b) in enumerate(zip(base, run)) if a[1] != b[1]), None)
        margin = None
        if first < len(base):
            chosen, runner = base[first][1], base[first][2]
            margin = None if runner is None else round(chosen - runner, 5)
        rows.append({"run": index, "first_token_diff": first, "first_logprob_diff": first_drift,
                     "max_logprob_drift_before": round(drift, 6), "margin_at_flip": margin})
    distinct = len({tuple(t for t, _, _ in run) for run in runs})
    report[name] = {"distinct": f"{distinct}/{REPEATS}", "length": len(base), "runs": rows}
    print(json.dumps({name: report[name]}), flush=True)
if SAVE:
    with open(SAVE, "w") as handle:
        json.dump(saved, handle)
