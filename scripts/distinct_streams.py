#!/usr/bin/env python3
"""N concurrent streams with N different prompts: decode throughput without shared routing.

usage: scripts/distinct_streams.py BASE_URL [--streams N] [--samples S] [--tokens T]

The bench's concurrent decode points send copies of one prompt; with batch invariance
the copies write the same text, route to the same experts and decode faster for that
reason alone. Here each stream has its own prompt (fresh cache salts), thinking off,
temperature 0, the same prompts in every sample. Per sample (one JSON line): generated
tokens over the window from the first stream's first token to the last stream's last
token, the per-stream decode rate, time to first token, and that window in wall-clock
seconds ("wall"), so external power or clock logs can be aligned with it. The last line
is the summary that lab.py's tables read.
"""
import argparse
import json
import statistics
import threading
import time
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
    "Explain how public-key cryptography lets two strangers agree on a secret.",
    "Describe the life cycle of a star from nebula to remnant.",
    "Write a letter from a beekeeper to a neighbour about a swarm in their garden.",
    "Explain how a compiler turns source code into machine instructions.",
    "Give practical advice for learning to play the cello as an adult.",
    "Describe how tides work and why there are two each day.",
    "Write a dialogue between a chess grandmaster and a curious child.",
    "Explain the water cycle and how climate change affects it.",
]


def model_id(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def stream(base: str, model: str, prompt: str, tokens: int, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "ignore_eos": True, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    out["offset"] = time.time() - start
    first = last = None
    count = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                count = chunk["usage"]["completion_tokens"]
            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                now = time.monotonic()
                first = first or now
                last = now
    out.update(start=start, first=first, last=last, tokens=count)


def sample(base: str, model: str, prompts: list[str], tokens: int) -> dict:
    results = [{} for _ in prompts]
    threads = [threading.Thread(target=stream, args=(base, model, p, tokens, r)) for p, r in zip(prompts, results)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(r.get("first") is None for r in results):
        raise SystemExit("a stream returned no content")
    window = max(r["last"] for r in results) - min(r["first"] for r in results)
    total = sum(r["tokens"] for r in results)
    per_stream = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in results if r["last"] > r["first"]]
    wall = (min(r["first"] + r["offset"] for r in results), max(r["last"] + r["offset"] for r in results))
    return {"tps": round(total / window, 2), "tokens": total, "wall": [round(w, 3) for w in wall],
            "per_stream_tps": round(statistics.mean(per_stream), 2) if per_stream else None,
            "ttft_s": round(statistics.mean(r["first"] - r["start"] for r in results), 3)}


def summary(streams: int, rates: list[float]) -> dict:
    mean = statistics.mean(rates)
    half = 1.96 * statistics.stdev(rates) / len(rates) ** 0.5 if len(rates) > 1 else 0.0
    return {"summary": f"c{streams} distinct prompts", "streams": streams, "tps_mean": round(mean, 2),
            "ci95_pct": round(100 * half / mean, 2), "samples": len(rates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.streams <= len(PROMPTS):
        parser.error(f"--streams must be between 1 and {len(PROMPTS)}")
    base = args.base_url.rstrip("/")
    model = model_id(base)
    rates = []
    for index in range(args.samples):
        result = sample(base, model, PROMPTS[: args.streams], args.tokens)
        rates.append(result["tps"])
        print(json.dumps({"sample": index, "streams": args.streams, **result}), flush=True)
    print(json.dumps(summary(args.streams, rates)), flush=True)


if __name__ == "__main__":
    main()
