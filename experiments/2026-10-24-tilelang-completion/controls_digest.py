#!/usr/bin/env python3
"""The generation controls' outputs, as digests two builds can compare bit for bit.

usage: controls_digest.py BASE_URL [--tokens T] [--seed S] [--scenarios NAME,...]

token_determinism.py's eight prompts, thinking off unless a scenario turns it on, fresh
cache salts, a fixed number of tokens. Each scenario sets one control the default
requests never use, so the kernels that apply it run (penalties with their token
histogram, min-p, logit bias, bad words, the thinking budget, custom logprob token
ids):

- alone: the eight one at a time;
- c8: all eight at once (the controls' kernels then see a batch of rows, and with
  speculative decoding each request's drafts).

Each request asks for the chosen tokens' logprobs. One JSON line per scenario and mode
with the SHA-256 digest of the choices (message, logprobs, finish reason) and any prompt
logprobs, then a summary of every digest. Two builds whose controls are the same bit for
bit print the same digests. A scenario vLLM rejects by design (HTTP 400 for min_p and
logit_bias under speculative decoding, REJECTABLE) is recorded with the server's message,
and its digest is the message's. Any other HTTP error is a failure: it is recorded, the
remaining scenarios still run, and the script exits 1; so is a refused or dropped
connection.
"""
import argparse
import hashlib
import http.client
import json
import threading
import urllib.error
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

# Ordinary vocabulary ids well inside the tokenizer's range, the same in every build.
BIAS = {"1000": -100.0, "2000": -100.0, "3000": 4.0, "4000": 2.5}
LOGPROB_IDS = [100, 1000, 3000, 4000, 5000]

SCENARIOS = {
    "penalties": {"temperature": 0.0, "repetition_penalty": 1.3,
                  "frequency_penalty": 0.4, "presence_penalty": 0.3},
    "penalties-sampled": {"temperature": 1.0, "repetition_penalty": 1.15,
                          "frequency_penalty": 0.2, "presence_penalty": -0.1},
    "min-p": {"temperature": 1.0, "min_p": 0.1},
    "logit-bias": {"temperature": 0.0, "logit_bias": BIAS},
    "bad-words": {"temperature": 0.0, "bad_words": [" the", " and", "ing"]},
    "thinking-budget": {"temperature": 0.0, "thinking_token_budget": 48,
                        "chat_template_kwargs": {"thinking": True}},
    "logprob-ids": {"temperature": 0.0, "logprob_token_ids": LOGPROB_IDS},
}
# Scenarios vLLM refuses with HTTP 400 under speculative decoding.
REJECTABLE = {"min-p", "logit-bias"}
# Prompt logprobs kill candidate 4's engine (scheduler.py's "no partial prefill outputs"
# assertion, 2026-10-24 run tilelang-ports-controls; fixed by 0266): named only with
# --scenarios, for a build that carries the fix.
OPT_IN = {
    "prompt-logprobs": {"temperature": 0.0, "prompt_logprobs": 5},
}


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def complete(self, prompt: str, tokens: int, seed: int, controls: dict) -> list:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True,
                "chat_template_kwargs": {"thinking": False}, **controls}
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        choice = result["choices"][0]
        return [{key: choice.get(key) for key in ("message", "logprobs", "finish_reason")},
                result.get("prompt_logprobs")]


def together(client: Client, tokens: int, seeds: list[int], controls: dict) -> list:
    """The eight at once; a request's error is raised here, as a request alone raises."""
    results: list = [None] * len(PROMPTS)
    errors: list = []

    def run(index):
        try:
            results[index] = client.complete(PROMPTS[index], tokens, seeds[index], controls)
        except Exception as error:  # noqa: BLE001 - re-raised in the caller's thread
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def digest(results: list) -> str:
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--scenarios", help="comma-separated scenario names "
                        f"(default: {', '.join(SCENARIOS)}; also {', '.join(OPT_IN)})")
    args = parser.parse_args()
    known = {**SCENARIOS, **OPT_IN}
    names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")
    client = Client(args.base_url.rstrip("/"))
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    digests = {}
    failed = []
    for name in names:
        controls = known[name]
        try:
            client.complete(PROMPTS[0], 1, seeds[0], controls)
            alone = [client.complete(p, args.tokens, s, controls)
                     for p, s in zip(PROMPTS, seeds)]
            c8 = together(client, args.tokens, seeds, controls)
        except (OSError, http.client.HTTPException) as error:
            # HTTPError is an OSError; a refused or dropped connection is a failure too.
            code = getattr(error, "code", None)
            message = (error.read().decode(errors="replace")
                       if isinstance(error, urllib.error.HTTPError) else repr(error))
            if code == 400 and name in REJECTABLE:
                print(json.dumps({"scenario": name, "rejected": code, "message": message}),
                      flush=True)
                digests[f"{name}/rejected"] = hashlib.sha256(message.encode()).hexdigest()
            else:
                print(json.dumps({"scenario": name, "failed": code, "message": message}),
                      flush=True)
                failed.append(name)
            continue
        for mode, results in (("alone", alone), ("c8", c8)):
            digests[f"{name}/{mode}"] = digest(results)
            print(json.dumps({"scenario": name, "mode": mode, "sha256": digests[f"{name}/{mode}"],
                              "finish": [r[0]["finish_reason"] for r in results]}), flush=True)
    print(json.dumps({"summary": "controls digests", "tokens": args.tokens, "seed": args.seed,
                      "digests": digests, "failed": failed}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
