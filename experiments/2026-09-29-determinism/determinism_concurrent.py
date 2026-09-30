#!/usr/bin/env python3
"""Does a temperature-0 request repeat across different batch compositions?

usage: determinism_concurrent.py BASE_URL [--tokens N] [--save FILE]

For each target prompt, five runs; in run i the target is sent together with a
different background mix (0, 2, 4, 7 and 7 other requests, different prompts
and lengths, the last mix reordered), so it is verified in batches of
different sizes and compositions. Reports, per target, how many distinct
outputs the five runs gave, the first differing token against run 1, and
the largest logprob difference before it. One JSON line per target.
"""
import json
import sys
import threading
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 256
SAVE = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
TARGETS = {
    "json": ("Return a JSON object describing three fictional planets with name, mass and moons.", False),
    "prose": ("Write a short paragraph about the history of the lighthouse.", False),
}
BACKGROUND = [
    ("List ten prime numbers and explain why each is prime.", 192),
    ("Write a haiku about rain, then explain its imagery.", 96),
    ("Summarize the plot of a heist film you invent.", 256),
    ("Explain how a hash map handles collisions.", 160),
    ("Describe a walk through a market in Marrakech.", 224),
    ("Give a recipe for lentil soup with timings.", 128),
    ("Compare TCP and UDP for game networking.", 256),
]
MIXES = [[], [0, 3], [1, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]]
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def complete(prompt, tokens, logprobs):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "temperature": 0, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    if logprobs:
        body.update(logprobs=True, top_logprobs=1)
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)["choices"][0]


def run(target, mix):
    threads = [threading.Thread(target=complete, args=(BACKGROUND[i][0], BACKGROUND[i][1], False))
               for i in mix]
    for thread in threads:
        thread.start()
    choice = complete(target, TOKENS, True)
    for thread in threads:
        thread.join()
    content = choice["logprobs"]["content"]
    return [c["token"] for c in content], [c["logprob"] for c in content]


saved = {}
for name, (prompt, _) in TARGETS.items():
    runs = [run(prompt, mix) for mix in MIXES]
    saved[name] = runs
    ref_tokens, ref_lp = runs[0]
    report = []
    for (tokens, lp), mix in zip(runs[1:], MIXES[1:]):
        diff = next((i for i, (a, b) in enumerate(zip(tokens, ref_tokens)) if a != b),
                    None if len(tokens) == len(ref_tokens) else min(len(tokens), len(ref_tokens)))
        upto = diff if diff is not None else min(len(tokens), len(ref_tokens))
        drift = max((abs(a - b) for a, b in zip(lp[:upto], ref_lp[:upto])), default=0.0)
        report.append({"background": len(mix), "first_token_diff": diff,
                       "max_logprob_drift_before": round(drift, 4)})
    distinct = len({tuple(t) for t, _ in runs})
    print(json.dumps({name: {"distinct": f"{distinct}/{len(runs)}", "length": len(ref_tokens),
                             "runs": report}}), flush=True)
if SAVE:
    json.dump(saved, open(SAVE, "w"))
