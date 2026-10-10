#!/usr/bin/env python3
"""Temperature-0 determinism at the output: the same prompts in different company.

usage: token_determinism.py BASE_URL [--tokens T] [--long-lines L]

Eight distinct prompts (scripts/distinct_streams.py's first eight), thinking off,
temperature 0, a fixed number of tokens, fresh cache salts, served:

- alone: one at a time (the reference);
- c8: all eight at once (decode steps of eight streams);
- staggered: arriving 100 ms apart (steps of every size from one to eight);
- mixed: all eight at once with a long prompt arriving 0.3 s later, so decode rows
  share steps with its prefill chunks; the long prompt's own output is compared with
  it served alone;
- cached: each prompt twice with one salt; the second reuses the first's KV and
  computes only its last token;
- again: one at a time once more (run to run, after the traffic above).

Each request asks for the chosen tokens' logprobs. Tokens must match the reference
exactly; logprobs are compared bit for bit as well (equal tokens with unequal
logprobs is a numeric difference that has not flipped a token yet). One JSON line
per scenario, then the summary that lab.py's tables read: "identical" is true only
when every token (and, where the server returns them, every logprob) matched. Exits
0 either way; the run records.
"""
import argparse
import json
import random
import threading
import time
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


def long_prompt(lines: int) -> str:
    """A fixed, roughly 15-token-per-line field log, then a question about it."""
    rng = random.Random(1210)
    colours = ["amber", "slate", "crimson", "olive", "ivory", "teal", "umber", "violet"]
    things = ["heron", "lantern", "ferry", "orchard", "glacier", "market", "bridge", "kiln"]
    places = ["north ridge", "harbour", "old mill", "east field", "lower lock", "quarry"]
    body = "\n".join(f"Entry {i}: a {rng.choice(colours)} {rng.choice(things)} near the "
                     f"{rng.choice(places)}, count {rng.randint(1, 99)}, wind {rng.randint(0, 40)} knots."
                     for i in range(lines))
    return f"Field log:\n{body}\n\nWhich entries mention a glacier near the harbour? List their numbers."


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]
        self.logprobs = True

    def complete(self, prompt: str, tokens: int, salt: str | None = None) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
                "temperature": 0, "ignore_eos": True, "cache_salt": salt or uuid.uuid4().hex,
                "chat_template_kwargs": {"thinking": False}}
        if self.logprobs:
            body["logprobs"] = True
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                choice = json.load(response)["choices"][0]
        except urllib.error.HTTPError as error:
            if self.logprobs and error.code == 400:  # the server does not return logprobs: tokens only
                self.logprobs = False
                return self.complete(prompt, tokens, salt)
            raise
        content = (choice.get("logprobs") or {}).get("content")
        if content:
            return {"tokens": [c["token"] for c in content], "logprobs": [c["logprob"] for c in content]}
        return {"tokens": None, "text": choice["message"]["content"], "logprobs": None}


def together(client: Client, jobs: list[tuple[float, str, int, str | None]]) -> list[dict]:
    """Run (delay seconds, prompt, tokens, salt) jobs concurrently; results in job order."""
    results: list[dict] = [{} for _ in jobs]
    start = time.monotonic()

    def run(index, delay, prompt, tokens, salt):
        time.sleep(max(0.0, start + delay - time.monotonic()))
        results[index].update(client.complete(prompt, tokens, salt))

    threads = [threading.Thread(target=run, args=(i, *job)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(not r for r in results):
        raise SystemExit("a request returned nothing")
    return results


def compare(result: dict, reference: dict) -> dict:
    """First differing token (None: all equal) and whether every logprob matched."""
    if result["tokens"] is None or reference["tokens"] is None:
        a, b = result.get("text") or "", reference.get("text") or ""
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
        return {"first_token_diff": None, "first_char_diff": first, "logprobs_equal": None}
    a, b = result["tokens"], reference["tokens"]
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
    shared = len(a) if first is None else first
    lp_equal = result["logprobs"][:shared] == reference["logprobs"][:shared]
    lp_first = next((i for i in range(shared) if result["logprobs"][i] != reference["logprobs"][i]), None)
    return {"first_token_diff": first, "logprobs_equal": lp_equal, "first_logprob_diff": lp_first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--long-lines", type=int, default=400)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    tokens, long_text = args.tokens, long_prompt(args.long_lines)

    reference = [client.complete(p, tokens) for p in PROMPTS]
    long_reference = client.complete(long_text, 64)
    scenarios = {
        "c8": together(client, [(0.0, p, tokens, None) for p in PROMPTS]),
        "staggered": together(client, [(0.1 * i, p, tokens, None) for i, p in enumerate(PROMPTS)]),
    }
    mixed = together(client, [(0.0, p, tokens, None) for p in PROMPTS] + [(0.3, long_text, 64, None)])
    scenarios["mixed"], long_mixed = mixed[:-1], mixed[-1]
    cached = []
    for prompt in PROMPTS:
        salt = uuid.uuid4().hex
        client.complete(prompt, tokens, salt)
        cached.append(client.complete(prompt, tokens, salt))
    scenarios["cached"] = cached
    scenarios["again"] = [client.complete(p, tokens) for p in PROMPTS]

    identical, report = True, {}
    for name, results in scenarios.items():
        rows = [dict(compare(r, ref), prompt=i) for i, (r, ref) in enumerate(zip(results, reference))]
        if name == "mixed":
            rows.append(dict(compare(long_mixed, long_reference), prompt="long"))
        token_diffs = sum(r.get("first_token_diff") is not None or r.get("first_char_diff") is not None for r in rows)
        logprob_diffs = sum(r["logprobs_equal"] is False for r in rows)
        identical &= token_diffs == 0 and logprob_diffs == 0
        report[name] = {"token_diffs": token_diffs, "logprob_diffs": logprob_diffs}
        print(json.dumps({"scenario": name, **report[name], "requests": rows}), flush=True)
    print(json.dumps({"summary": "temperature-0 determinism", "identical": identical,
                      "logprobs": client.logprobs, "tokens": tokens, "scenarios": report}), flush=True)


if __name__ == "__main__":
    main()
