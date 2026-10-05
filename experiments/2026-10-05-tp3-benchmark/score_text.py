#!/usr/bin/env python3
"""Score fixed text through the prefill path, for comparing kernel families.

usage: score_text.py BASE_URL [TOKENS_PER_TEXT]

Each text is a file from this repository (Python source and English prose),
tokenized and scored with prompt_logprobs and a fresh cache salt. For every
position it records the logprob of the actual next token, and the top token
with its logprob. Two families' outputs for the same texts compare token for
token: the same tokens, so differences come only from numerics, not from
which text a generation happened to take. Prints one JSON object.
"""
import json
import sys
import urllib.request
import uuid
from pathlib import Path

base = sys.argv[1].rstrip("/")
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
root = Path(__file__).resolve().parents[2]
TEXTS = ["bin/spark", "scripts/topology.py", "TODO.md", "README.md"]
model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]


def post(path, body):
    request = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


results = {}
for name in TEXTS:
    tokens = post("/tokenize", {"model": model, "prompt": (root / name).read_text()})["tokens"][:limit]
    scored = post("/v1/completions", {
        "model": model, "prompt": tokens, "max_tokens": 1, "temperature": 0,
        "prompt_logprobs": 1, "cache_salt": uuid.uuid4().hex,
    })["choices"][0]["prompt_logprobs"]
    rows = []
    for token, entry in zip(tokens[1:], scored[1:]):
        best_id, best = max(entry.items(), key=lambda item: item[1]["logprob"])
        rows.append([token, entry[str(token)]["logprob"], int(best_id), best["logprob"]])
    results[name] = rows
print(json.dumps({"model": model, "tokens_per_text": limit, "texts": results}))
