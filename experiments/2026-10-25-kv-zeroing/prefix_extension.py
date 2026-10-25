#!/usr/bin/env python3
"""A prefix-cache hit whose first chunk is shorter than the 128-token window, at
temperature 0: the hit's tokens equal the same prompt computed from scratch.

usage: prefix_extension.py URL --threshold T [--extension N] [--max-tokens M]

T is the arm's chunk threshold (``--long-prefill-token-threshold``: 16,288 on the
ring at 16,384 batched tokens, 8,144 on the triangle at 8,192). Aligned chunking cuts
a prompt resumed at token L into a first chunk of T - L % T tokens. The script picks
the smallest block-aligned L (a multiple of 256, at least T) whose first chunk is
under 128 tokens (32,512 and 64 tokens on the ring), so the next steps' windows reach back across the hit:

- seed: the first L + 100 tokens of a real-text document (as token ids, through
  /v1/completions), which caches the prefix's blocks up to L;
- hit: the first L + 100 + N tokens of the same document: the prefix cache serves L
  tokens, the first chunk is short and the later chunks are full (CED's nonfinal
  one-row chunks among them);
- fresh: the hit's prompt again with a fresh cache salt, computed from scratch.

"identical" when the hit's tokens equal the fresh tokens; "hit_tokens" is what the
server's prefix-cache counter added during the hit (L when the whole prefix was
served; a shorter hit tests less). One JSON line a request, then the summary. Exits 0
either way. Stdlib and the repository's servebench only.
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from servebench import prom, prompts  # noqa: E402

BLOCK = 256  # the prefix cache's granularity (the scheduler's block size)
WINDOW = 128


def post(base: str, path: str, body: dict, timeout: float = 3600.0) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def hit_point(threshold: int) -> int:
    length = -(-threshold // BLOCK) * BLOCK
    while not 0 < threshold - length % threshold < WINDOW:
        length += BLOCK
    return length


def complete(base: str, model: str, ids: list[int], max_tokens: int, salt: str | None) -> dict:
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
            "return_token_ids": True}
    if salt is not None:
        body["cache_salt"] = salt
    before, started = prom.scrape(base, prom.VLLM) or {}, time.monotonic()
    record = {"prompt_tokens": len(ids)}
    try:
        choice = post(base, "/v1/completions", body)["choices"][0]
        record.update(output_ids=choice.get("token_ids"), finish_reason=choice.get("finish_reason"))
    except Exception as error:  # noqa: BLE001 - a failed request is recorded, not raised
        record["error"] = repr(error)[:300]
    after = prom.scrape(base, prom.VLLM) or {}
    record["seconds"] = round(time.monotonic() - started, 3)
    record["hit_tokens"] = after.get("prefix_cache_hits", 0) - before.get("prefix_cache_hits", 0)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--extension", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    hit = hit_point(args.threshold)
    total = hit + 100 + args.extension
    ids: list[int] = []
    seed = 3000
    while len(ids) < total:
        text = prompts.source_text(total, seed)
        ids += post(base, "/tokenize", {"model": model, "prompt": text,
                                        "add_special_tokens": not ids})["tokens"]
        seed += 1
    ids = ids[:total]
    salt = uuid.uuid4().hex  # this run's own cache: the seed's blocks are what the hit finds
    seeded = complete(base, model, ids[: hit + 100], args.max_tokens, salt)
    hitting = complete(base, model, ids, args.max_tokens, salt)
    fresh = complete(base, model, ids, args.max_tokens, uuid.uuid4().hex)
    for name, record in (("seed", seeded), ("hit", hitting), ("fresh", fresh)):
        print(json.dumps({"request": name, **record}), flush=True)
    first_chunk = args.threshold - hitting["hit_tokens"] % args.threshold if hitting["hit_tokens"] else None
    print(json.dumps({
        "summary": "prefix hit with a short first chunk",
        "identical": "error" not in hitting and "error" not in fresh
        and hitting.get("output_ids") == fresh.get("output_ids"),
        "hit_point": hit,
        "hit_tokens": hitting["hit_tokens"],
        "first_chunk": first_chunk,
        "short_first_chunk": first_chunk is not None and first_chunk < WINDOW,
        "failed": sum("error" in r for r in (seeded, hitting, fresh)),
        "prompt_tokens": total,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
