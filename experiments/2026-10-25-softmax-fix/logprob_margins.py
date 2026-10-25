#!/usr/bin/env python3
"""Served log probabilities near 0, as a build reports them, for comparing builds.

usage: logprob_margins.py BASE_URL [--tokens T] [--top K] [--seed S]

One request at a time, thinking off, fresh cache salts:

- greedy: twelve prompts at temperature 0 (token_determinism's eight, and four code
  and data prompts, where confident tokens are common), each with its chosen tokens'
  log probabilities and the top K;
- prompt: a fixed, repetitive Python passage as a completion with prompt logprobs
  (K) and one token, its prompt token ids returned;
- sampled: the eight prompts at temperature 1, each with its own seed, chosen
  tokens' log probabilities (block verification decides the sampled tokens).

One JSON line per request with its tokens and values (tables_arms.py compares builds
position by position), then a summary: of the chosen tokens' (and the prompt tokens')
log probabilities, how many are exactly 0, how many are in (-1e-5, 0), and how many
of those sit on the release's 2^-23 grid; and SHA-256 digests of the texts and of
the values. Exits 0 either way; the run records.
"""
import argparse
import hashlib
import json
import urllib.request
import uuid

PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
]
CODE = [
    "Write a Python LRU cache class with get and put in O(1), with type hints and "
    "docstrings, and nothing else.",
    "Write the JSON Schema for a user record with id, email, name, roles (an array of "
    "strings) and created_at, and nothing else.",
    "Write a Rust function that parses a comma-separated line of integers into a "
    "Vec<i64>, returning an error on bad input, with unit tests.",
    "List the first forty prime numbers as a JSON array, and nothing else.",
]
GRID = 2.0**-23


def passage() -> str:
    """Python source whose later lines repeat its earlier ones: confident prompt
    tokens."""
    lines = ["class Inventory:", '    """A store\'s stock, by item."""', "",
             "    def __init__(self):", "        self.items = {}", ""]
    for k in range(24):
        lines += [f"    def add_{k}(self, name, count={k}):",
                  "        self.items[name] = self.items.get(name, 0) + count",
                  "        return self.items[name]", ""]
    return "\n".join(lines)


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def chat(self, prompt: str, tokens: int, top: int, temperature: float,
             seed: int | None) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "temperature": temperature, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True, "top_logprobs": top,
                "chat_template_kwargs": {"thinking": False}}
        if seed is not None:
            body["seed"] = seed
        content = self.post("/v1/chat/completions", body)["choices"][0]["logprobs"]["content"]
        return {"tokens": [c["token"] for c in content],
                "logprobs": [c["logprob"] for c in content],
                "top": [[[t["token"], t["logprob"]] for t in c["top_logprobs"]]
                        for c in content]}

    def prompt(self, text: str, top: int) -> dict:
        body = {"model": self.model, "prompt": text, "max_tokens": 1, "temperature": 0,
                "prompt_logprobs": top, "return_token_ids": True,
                "cache_salt": uuid.uuid4().hex}
        choice = self.post("/v1/completions", body)["choices"][0]
        ids = choice["prompt_token_ids"]
        positions = choice["prompt_logprobs"]
        # Position 0 has none; each other maps token ids to their scores.
        logprobs, top_ids = [], []
        for token, entry in zip(ids[1:], positions[1:]):
            logprobs.append(entry[str(token)]["logprob"])
            top_ids.append(sorted([int(k), v["logprob"]] for k, v in entry.items()))
        return {"tokens": ids[1:], "logprobs": logprobs, "top": top_ids}


def counts(values: list[float]) -> dict:
    tiny = [v for v in values if -1e-5 < v < 0]
    return {"n": len(values), "zero": sum(v == 0.0 for v in values), "tiny": len(tiny),
            "on_grid": sum(abs(v / GRID - round(v / GRID)) < 1e-3 for v in tiny),
            "smallest_nonzero": max(tiny) if tiny else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    records = []
    for index, prompt in enumerate(PROMPTS + CODE):
        record = {"mode": "greedy", "index": index,
                  **client.chat(prompt, args.tokens, args.top, 0.0, None)}
        records.append(record)
        print(json.dumps(record), flush=True)
    record = {"mode": "prompt", "index": 0, **client.prompt(passage(), args.top)}
    records.append(record)
    print(json.dumps(record), flush=True)
    for index, prompt in enumerate(PROMPTS):
        record = {"mode": "sampled", "index": index,
                  **client.chat(prompt, args.tokens, 0, 1.0, args.seed + index)}
        record.pop("top")
        records.append(record)
        print(json.dumps(record), flush=True)

    def digest(payload) -> str:
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    summary = {"summary": "logprobs near 0", "tokens": args.tokens, "top": args.top}
    for mode in ("greedy", "prompt", "sampled"):
        mine = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            "chosen": counts([v for r in mine for v in r["logprobs"]]),
            "text_sha256": digest([r["tokens"] for r in mine]),
            "values_sha256": digest([[r["logprobs"], r.get("top")] for r in mine]),
        }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
