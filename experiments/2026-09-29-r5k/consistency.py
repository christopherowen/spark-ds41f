#!/usr/bin/env python3
"""Decode-versus-prefill self-consistency of greedy generations.

usage: consistency.py BASE_URL [PROMPTS] [MAX_TOKENS]
For each prompt: generate MAX_TOKENS greedily (decode path, logprob of each
chosen token), then score prompt + those exact token ids as a prompt with a
fresh cache salt (prefill path recomputes everything, no prefix-cache hit).
Reports the mean and 95th-percentile absolute logprob gap over the generated
tokens, and how often prefill's argmax disagrees with the decoded token.
Decode builds compressed KV entries step by step; prefill builds them in one
pass, so a decode-only state bug shows up as a larger gap.
"""
import json
import statistics
import sys
import urllib.request
import uuid

base = sys.argv[1].rstrip("/")
count = int(sys.argv[2]) if len(sys.argv) > 2 else 8
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]
subjects = [
    "Explain how a write-ahead log makes a database crash-safe, with an example.",
    "Write a Python function that merges overlapping intervals and explain it.",
    "Describe the causes of the French Revolution in a few paragraphs.",
    "Walk through how TCP congestion control reacts to packet loss.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between processes and threads, with trade-offs.",
    "Derive the formula for the sum of a geometric series step by step.",
    "Review this idea: caching every HTTP response for an hour. What breaks?",
    "Explain how public-key cryptography lets strangers agree on a secret.",
    "Write a SQL query to find the second-highest salary per department.",
]


def post(path, body):
    request = urllib.request.Request(f"{base}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def ids(entries):
    return [int(token.split(":", 1)[1]) for token in entries]


gaps, flips, total = [], 0, 0
for subject in subjects[:count]:
    prompt = f"<｜begin▁of▁sentence｜><｜User｜>{subject}<｜Assistant｜></think>"
    generated = post("/v1/completions", {
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
        "logprobs": 1, "return_tokens_as_token_ids": True, "ignore_eos": True,
        "cache_salt": uuid.uuid4().hex,
    })["choices"][0]["logprobs"]
    decoded = ids(generated["tokens"])
    decode_lp = generated["token_logprobs"]
    prompt_ids = post("/tokenize", {"model": model, "prompt": prompt})["tokens"]
    scored = post("/v1/completions", {
        "model": model, "prompt": prompt_ids + decoded, "max_tokens": 1, "temperature": 0,
        "prompt_logprobs": 1, "cache_salt": uuid.uuid4().hex,
    })["choices"][0]["prompt_logprobs"]
    tail = scored[len(prompt_ids):]
    for token, lp_decode, entry in zip(decoded, decode_lp, tail):
        chosen = entry[str(token)]["logprob"]
        best = max(entry.values(), key=lambda item: item["logprob"])
        gaps.append(abs(chosen - lp_decode))
        flips += best["rank"] == 1 and entry[str(token)]["rank"] != 1
        total += 1
gaps.sort()
print(f"{total} tokens over {count} prompts: mean |logprob gap| {statistics.mean(gaps):.4f}, "
      f"p95 {gaps[int(0.95 * len(gaps))]:.4f}, max {gaps[-1]:.3f}, "
      f"prefill argmax differs on {flips} ({100 * flips / total:.2f}%)")
