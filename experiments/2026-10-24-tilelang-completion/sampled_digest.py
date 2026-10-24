#!/usr/bin/env python3
"""Seeded temperature-1 outputs, as digests two builds can compare bit for bit.

usage: sampled_digest.py BASE_URL [--tokens T] [--seed S]

token_determinism.py's eight prompts, thinking off, temperature 1, each request with its
own fixed seed (``--seed`` plus its index), a fixed number of tokens, fresh cache salts:

- alone: one at a time (the reference);
- c8: all eight at once (the draws are keyed by request seed and position, so the
  company must not change them);
- again: one at a time once more.

Speculative decoding samples its drafts, verifies them and resamples rejected positions
on this path, so a build whose sampling kernels changed is compared on all of it. Each
request asks for the chosen tokens' logprobs. One JSON line per scenario, then a summary
with "identical" (every scenario equal to the reference, tokens and logprobs) and the
reference's SHA-256 digest, which must be the same for two builds whose sampling is the
same bit for bit. Exits 0 either way; the run records.
"""
import argparse
import hashlib
import json
import threading
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


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def complete(self, prompt: str, tokens: int, seed: int) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "temperature": 1.0, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True,
                "chat_template_kwargs": {"thinking": False}}
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.load(response)["choices"][0]
        content = choice["logprobs"]["content"]
        return {"tokens": [c["token"] for c in content],
                "logprobs": [c["logprob"] for c in content]}


def together(client: Client, prompts: list[str], tokens: int, seeds: list[int]) -> list[dict]:
    results: list[dict] = [{} for _ in prompts]

    def run(index):
        results[index].update(client.complete(prompts[index], tokens, seeds[index]))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(not r for r in results):
        raise SystemExit("a request returned nothing")
    return results


def digest(results: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[r["tokens"], r["logprobs"]] for r in results])
                          .encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    reference = [client.complete(p, args.tokens, s) for p, s in zip(PROMPTS, seeds)]
    scenarios = {
        "c8": together(client, PROMPTS, args.tokens, seeds),
        "again": [client.complete(p, args.tokens, s) for p, s in zip(PROMPTS, seeds)],
    }
    identical, report = True, {}
    for name, results in scenarios.items():
        differing = [i for i, (r, ref) in enumerate(zip(results, reference)) if r != ref]
        identical &= not differing
        report[name] = {"differing": differing}
        print(json.dumps({"scenario": name, "differing": differing,
                          "sha256": digest(results)}), flush=True)
    print(json.dumps({"summary": "seeded temperature-1 digest", "identical": identical,
                      "tokens": args.tokens, "seed": args.seed, "scenarios": report,
                      "reference_sha256": digest(reference)}), flush=True)


if __name__ == "__main__":
    main()
