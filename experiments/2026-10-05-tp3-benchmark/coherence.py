#!/usr/bin/env python3
"""Greedy answers to a few varied prompts, saved in full, for reading.

usage: coherence.py BASE_URL OUTPUT_JSON [MAX_TOKENS]

Each prompt goes through /v1/chat/completions at temperature 0 with
reasoning on. The script saves the reasoning and the answer, and prints for
each answer its length, the share of repeated 8-grams (a loop shows as a high
share) and the opening of the answer.
"""
import json
import sys
import urllib.request

base, output = sys.argv[1].rstrip("/"), sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]
PROMPTS = {
    "explain": "Explain how a hash map handles collisions, comparing chaining and open addressing.",
    "code": "Write a Python function that returns the longest palindromic substring of a string, with a short explanation.",
    "math": "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the working.",
    "story": "Write a five-sentence story about a cat who learns to open doors.",
    "json": "Return a JSON object with keys name, year and reason for three programming languages created before 1980.",
    "reason": "If all bloops are razzies and some razzies are lazzies, must some bloops be lazzies? Explain.",
}


def repeated_share(text: str, n: int = 8) -> float:
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


results = {}
for name, prompt in PROMPTS.items():
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
    }).encode(), headers={"Content-Type": "application/json"})
    choice = json.load(urllib.request.urlopen(request, timeout=900))["choices"][0]
    message = choice["message"]
    answer = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    results[name] = {"prompt": prompt, "finish_reason": choice.get("finish_reason"),
                     "reasoning": reasoning, "answer": answer}
    print(f"== {name}: finish {choice.get('finish_reason')}, reasoning {len(reasoning)} chars, "
          f"answer {len(answer)} chars, repeated 8-grams {100 * repeated_share(reasoning + ' ' + answer):.1f}%")
    print("   " + answer[:400].replace("\n", "\n   "))
json.dump(results, open(output, "w"), indent=1)
print(f"{len(results)} passed")
