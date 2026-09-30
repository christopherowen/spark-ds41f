#!/usr/bin/env python3
"""Do identical long-prompt requests repeat? (exercises the prefill kernels)

usage: determinism_long.py BASE_URL [--words N] [--repeats R] [--tokens T]

Builds one fixed document of about N words (numbered facts about fictional
towns, about 1.3 tokens a word) and asks a question about it, R times in
sequence with distinct cache salts so every run prefills from scratch
(4,096-token chunks: the mHC and BF16 prefill projections run). Temperature 0,
thinking off. Prints one JSON line: prompt tokens, distinct outputs, and per
run against run 1 the first differing token and the first-token logprob
difference (a prefill difference shows there first).
"""
import json
import sys
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")


def arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


WORDS, REPEATS, TOKENS = arg("--words", 4600), arg("--repeats", 5), arg("--tokens", 64)
COLOURS = ("red", "blue", "green", "amber", "violet", "grey", "white", "black")
TRADES = ("salt", "wool", "glass", "timber", "copper", "cider", "paper", "rope")
facts, words, i = [], 0, 0
while words < WORDS:
    fact = (f"Fact {i}: the town of Vell-{i:04d} paints its doors {COLOURS[i % 8]}, trades mainly in "
            f"{TRADES[(i * 3) % 8]}, holds its fair in month {1 + i % 12} and has {100 + (i * 37) % 900} "
            f"households along {2 + i % 5} streets.")
    facts.append(fact)
    words += len(fact.split())
    i += 1
QUESTION = ("Using only the facts above, list the towns whose doors are violet and whose fair is in "
            "month 7, with their main trade, as a JSON array.")
PROMPT = "\n".join(facts) + "\n\n" + QUESTION
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def run():
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": TOKENS,
            "temperature": 0, "logprobs": True, "top_logprobs": 1, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    content = data["choices"][0]["logprobs"]["content"]
    return data["usage"]["prompt_tokens"], [c["token"] for c in content], [c["logprob"] for c in content]


runs = [run() for _ in range(REPEATS)]
prompt_tokens, ref_tokens, ref_lp = runs[0]
report = []
for _, tokens, lp in runs[1:]:
    diff = next((k for k, (a, b) in enumerate(zip(tokens, ref_tokens)) if a != b),
                None if len(tokens) == len(ref_tokens) else min(len(tokens), len(ref_tokens)))
    report.append({"first_token_diff": diff, "first_token_logprob_diff": round(abs(lp[0] - ref_lp[0]), 6)})
print(json.dumps({"prompt_tokens": prompt_tokens,
                  "distinct": f"{len({tuple(t) for _, t, _ in runs})}/{REPEATS}",
                  "length": len(ref_tokens), "runs": report}))
