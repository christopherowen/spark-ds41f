#!/usr/bin/env python3
"""DSpark acceptance per turn of fixed coding conversations, then on held-out prompts.

usage: online_adaptation_workload.py BASE_URL [--turn-tokens N] [--heldout-tokens N]

One request at a time (concurrency 1), temperature 0, thinking off:

1. The coding set: four fixed conversations of six user turns each, in order. Each
   turn sends the conversation so far, the assistant turns being the model's own
   earlier replies, with the conversation's cache salt (fresh per run). A reply ends
   at its end of text or at --turn-tokens.
2. The held-out set: distinct_streams.py's sixteen general prompts, each alone with a
   fresh cache salt and --heldout-tokens tokens (ignore_eos), as acceptance_sweep.py.

For every request the server's speculative-decoding counters are read before and
after it: accepted tokens per draft step, and the acceptance length (1 + that: the
tokens a verification step emits), verified drafts per step, and the decode rate
(tokens after the first over the time from the first to the last token). Every
reply's text is hashed, so arms that must write the same text can be compared.
One JSON line per request, then one summary line per set and a last summary line.
Any request finished by someone else during the run makes the script exit 3.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from distinct_streams import PROMPTS, finished_requests, model_id  # noqa: E402

COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total")

CODING = [
    [
        "Write a Python class LRUCache with get(key) and put(key, value) in O(1), using "
        "collections.OrderedDict, and a capacity given to __init__.",
        "Add an optional ttl_seconds to LRUCache: entries older than their ttl are treated "
        "as missing by get and are evicted lazily. Show the whole class.",
        "Make LRUCache safe to use from several threads with threading.Lock, without "
        "holding the lock while computing anything expensive. Show the changed methods.",
        "Write pytest tests for LRUCache covering eviction order, ttl expiry (use a fake "
        "clock injected through __init__) and capacity 1.",
        "Add full type hints with typing.Generic[K, V] and docstrings to LRUCache, and "
        "show the final class.",
        "Add a stats() method returning a dataclass CacheStats with hits, misses and "
        "evictions, and update get/put to count them.",
    ],
    [
        "Write an Express.js REST API for todos with GET /todos, POST /todos, PATCH "
        "/todos/:id and DELETE /todos/:id, storing todos in memory.",
        "Validate the POST and PATCH bodies with zod: title is a non-empty string up to "
        "200 characters, done is a boolean. Return 400 with the zod issues on failure.",
        "Add pagination to GET /todos with limit and offset query parameters (defaults 20 "
        "and 0, limit at most 100) and a total count in the response.",
        "Add an error-handling middleware that logs the error and returns JSON "
        "{ error: message } with status 500, and use it from the routes.",
        "Write supertest tests for the todo API: create, list with pagination, patch, "
        "delete, and the 400 validation response.",
        "Convert the todo API to TypeScript with a Todo interface and typed request "
        "handlers, and show the main file.",
    ],
    [
        "Write a Rust command-line program that counts word frequencies in text read from "
        "stdin and prints the counts, using a HashMap<String, usize>.",
        "Change it to read the files named on the command line instead of stdin, with "
        "proper error handling using std::io::Result and a message for unreadable files.",
        "Print only the top N words (N from a --top flag, default 10) using a "
        "BinaryHeap, ties broken alphabetically.",
        "Lowercase words and strip punctuation with char::is_alphanumeric, and write unit "
        "tests for the tokenizer function.",
        "Count words from several files in parallel with rayon, merging the per-file "
        "HashMaps. Show the changed main function.",
        "Refactor the program into a lib.rs with count_words and top_n functions and a "
        "thin main.rs, and show both files.",
    ],
    [
        "Write a PostgreSQL schema for an online shop: customers, products, orders and "
        "order_items, with primary keys, foreign keys and sensible types.",
        "Write a SQL query for monthly revenue per product category over the last 12 "
        "months, using the schema above.",
        "Write Python code that runs that query with psycopg and loads the result into a "
        "pandas DataFrame pivoted by category.",
        "This line raises a KeyError when a month has no orders: "
        "df.pivot(index='month', columns='category', values='revenue')['books']. "
        "Fix it so missing categories and months appear as zero.",
        "Suggest indexes for the schema to speed up the monthly revenue query, with the "
        "CREATE INDEX statements and why each helps.",
        "Write an Alembic migration that adds a discount_cents column to order_items "
        "and backfills it with 0.",
    ],
]


def counters(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        text = response.read().decode()
    totals = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in totals:
            totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


def chat(base: str, model: str, messages: list[dict], tokens: int, salt: str,
         ignore_eos: bool) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": tokens, "temperature": 0,
            "stream": True, "ignore_eos": ignore_eos, "cache_salt": salt,
            "chat_template_kwargs": {"thinking": False},
            "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.monotonic()
    first = last = None
    count = 0
    pieces = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                count = chunk["usage"]["completion_tokens"]
            delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
            if delta.get("content"):
                now = time.monotonic()
                first = first or now
                last = now
                pieces.append(delta["content"])
    return {"start": start, "first": first, "last": last, "tokens": count,
            "text": "".join(pieces)}


def measured(base: str, model: str, messages: list[dict], tokens: int, salt: str,
             ignore_eos: bool) -> tuple[dict, str]:
    before = counters(base)
    result = chat(base, model, messages, tokens, salt, ignore_eos)
    after = counters(base)
    delta = {name: after[name] - before[name] for name in COUNTERS}
    drafts = delta[COUNTERS[0]]
    accepted = delta[COUNTERS[2]] / drafts if drafts else None
    window = (result["last"] - result["first"]) if result["first"] is not None else 0.0
    row = {
        "tokens": result["tokens"],
        "draft_steps": int(drafts),
        "accepted_per_draft": round(accepted, 4) if accepted is not None else None,
        "acceptance_length": round(1 + accepted, 4) if accepted is not None else None,
        "verified_per_draft": round(delta[COUNTERS[1]] / drafts, 4) if drafts else None,
        "tps": round((result["tokens"] - 1) / window, 2) if window > 0 else None,
        "ttft_s": round(result["first"] - result["start"], 3) if result["first"] else None,
        "text_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
    }
    return row, result["text"]


def mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def pooled(rows: list[dict]) -> float | None:
    """Acceptance length over every draft step of the rows (weights by steps)."""
    steps = sum(r["draft_steps"] for r in rows)
    accepted = sum(r["accepted_per_draft"] * r["draft_steps"] for r in rows
                   if r["accepted_per_draft"] is not None)
    return round(1 + accepted / steps, 4) if steps else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--turn-tokens", type=int, default=384)
    parser.add_argument("--heldout-tokens", type=int, default=256)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    model = model_id(base)
    run = uuid.uuid4().hex
    finished_before = finished_requests(base)
    texts = hashlib.sha256()
    coding: list[dict] = []
    for c, turns in enumerate(CODING):
        messages: list[dict] = []
        salt = f"{run}-coding-{c}"
        for t, user in enumerate(turns):
            messages.append({"role": "user", "content": user})
            row, text = measured(base, model, messages, args.turn_tokens, salt, False)
            messages.append({"role": "assistant", "content": text})
            texts.update(text.encode())
            row = {"set": "coding", "conversation": c, "turn": t, **row}
            coding.append(row)
            print(json.dumps(row), flush=True)
    heldout: list[dict] = []
    for p, prompt in enumerate(PROMPTS):
        row, text = measured(base, model, [{"role": "user", "content": prompt}],
                             args.heldout_tokens, f"{run}-heldout-{p}", True)
        texts.update(text.encode())
        row = {"set": "heldout", "prompt": p, **row}
        heldout.append(row)
        print(json.dumps(row), flush=True)
    finished_after = finished_requests(base)
    own = len(coding) + len(heldout)
    foreign = (None if finished_before is None or finished_after is None
               else int(finished_after - finished_before) - own)

    per_turn = {t: [r for r in coding if r["turn"] == t] for t in range(len(CODING[0]))}
    coding_summary = {
        "summary": "coding conversations", "requests": len(coding),
        "acceptance_length": pooled(coding),
        "acceptance_length_per_turn": {t: pooled(rows) for t, rows in per_turn.items()},
        "acceptance_length_per_conversation": {
            c: pooled([r for r in coding if r["conversation"] == c])
            for c in range(len(CODING))},
        "tps_mean": mean([r["tps"] for r in coding]),
        "tokens": sum(r["tokens"] for r in coding),
    }
    heldout_summary = {
        "summary": "held-out prompts", "requests": len(heldout),
        "acceptance_length": pooled(heldout),
        "acceptance_length_mean": mean([r["acceptance_length"] for r in heldout]),
        "tps_mean": mean([r["tps"] for r in heldout]),
        "tokens": sum(r["tokens"] for r in heldout),
    }
    print(json.dumps(coding_summary), flush=True)
    print(json.dumps(heldout_summary), flush=True)
    print(json.dumps({
        "summary": "online adaptation workload",
        "coding_acceptance_length": coding_summary["acceptance_length"],
        "coding_tps_mean": coding_summary["tps_mean"],
        "heldout_acceptance_length": heldout_summary["acceptance_length"],
        "heldout_tps_mean": heldout_summary["tps_mean"],
        "text_sha256": texts.hexdigest(),
        "foreign_requests": foreign,
    }), flush=True)
    if foreign:
        sys.exit(3)


if __name__ == "__main__":
    main()
