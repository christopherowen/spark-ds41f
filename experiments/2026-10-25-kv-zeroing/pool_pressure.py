#!/usr/bin/env python3
"""Pool pressure at temperature 0: every request's tokens with the pool exhausted equal
its tokens served alone.

usage: pool_pressure.py URL [--streams N] [--documents D] [--tokens T] [--max-tokens M]

Run on an arm with a small KV pool (this experiment's *-pressure arms). D real-text
documents of about T tokens (servebench's standard-library source at fixed seeds) are
each asked N/D different questions, so requests on one document share its prefix:

- alone: the N requests one at a time. A document's later questions hit its cached
  prefix; the window, compressed and ring blocks they allocate come from blocks other
  groups freed.
- together: all N at once. Their blocks exceed the pool, so the scheduler preempts and
  recomputes running requests, evicts cached prefixes and hands their blocks to other
  cache groups.

Before and after each pass it reads the server's counters: preemptions, prefix-cache
queries and hits, requests with NaN logits (``vllm:corrupted_requests``, counted under
VLLM_COMPUTE_NANS_IN_LOGITS), drafted and accepted tokens. One JSON line a pass, then
the summary the experiment's table reads: "identical" when every request's tokens
together equal its tokens alone, "pressure" when the together pass preempted, the
digests of both passes' outputs (equal across arms for builds that agree), the NaN
count, and failed requests. Exits 0 either way; the run records. Stdlib and the
repository's servebench only.
"""
import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from servebench import load as loads  # noqa: E402
from servebench import prom, prompts  # noqa: E402

QUESTIONS = [
    "List the three most important functions defined above, one line each.",
    "What does the first class above do? Answer in two sentences.",
    "Name every module the code above imports.",
    "Which function above is the longest, and what does it return?",
    "Summarize the document above in one sentence.",
    "Write a one-line docstring for the last function above.",
    "How many classes does the code above define? Name them.",
    "Which exceptions can the code above raise?",
]
COUNTERS = {
    **{key: prom.VLLM[key] for key in ("preemptions", "prefix_cache_queries",
                                       "prefix_cache_hits", "draft_tokens",
                                       "accepted_tokens", "requests_finished")},
    "nan_requests": "vllm:corrupted_requests_total",
    "nan_requests_untyped": "vllm:corrupted_requests",
}


def model_id(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def requests_for(base: str, model: str, streams: int, documents: int, tokens: int) -> list[dict]:
    texts = []
    for d in range(documents):
        seed = 2000 + d
        text = prompts.source_text(tokens, seed)
        counted = loads.token_count(base, model, text)
        if counted:
            text = prompts.source_text(tokens, seed, chars_per_token=len(text) / counted)
        texts.append(text)
    return [{"index": i, "document": i % documents,
             "text": f"{texts[i % documents]}\n\n{QUESTIONS[(i // documents) % len(QUESTIONS)]}"}
            for i in range(streams)]


def generate(base: str, model: str, request: dict, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": request["text"]}],
            "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True,
            "chat_template_kwargs": {"thinking": False}}
    post = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    record = {"index": request["index"], "document": request["document"]}
    try:
        with urllib.request.urlopen(post, timeout=timeout) as response:
            answer = json.load(response)
    except Exception as error:  # noqa: BLE001 - a failed request is recorded, not raised
        record["error"] = repr(error)[:300]
        return record
    choice = answer["choices"][0]
    record.update(prompt_tokens=answer.get("usage", {}).get("prompt_tokens"),
                  output_ids=choice.get("token_ids"), finish_reason=choice.get("finish_reason"),
                  seconds=round(time.monotonic() - started, 3))
    if record["output_ids"] is None:
        record["error"] = "no token ids in the reply (the server needs return_token_ids)"
    return record


def counters(base: str) -> dict:
    return prom.scrape(base, COUNTERS, timeout=60) or {}


def delta(before: dict, after: dict) -> dict:
    out = {key: after[key] - before.get(key, 0.0) for key in after}
    out["nan_requests"] = out.pop("nan_requests", 0.0) + out.pop("nan_requests_untyped", 0.0)
    return out


def digest(records: list[dict]) -> str:
    rows = [[r["index"], r.get("output_ids")] for r in sorted(records, key=lambda r: r["index"])]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def run_pass(name, base, model, requests, max_tokens, timeout, concurrency):
    before, started = counters(base), time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(pool.map(lambda r: generate(base, model, r, max_tokens, timeout), requests))
    after = counters(base)
    line = {"pass": name, "seconds": round(time.monotonic() - started, 1),
            "counters": delta(before, after), "outputs_sha256": digest(records),
            "failed": sum(1 for r in records if "error" in r), "requests": records}
    print(json.dumps(line), flush=True)
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--documents", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=110_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    model = model_id(base)
    requests = requests_for(base, model, args.streams, args.documents, args.tokens)
    alone = run_pass("alone", base, model, requests, args.max_tokens, args.timeout, 1)
    together = run_pass("together", base, model, requests, args.max_tokens, args.timeout,
                        len(requests))
    by_index = {r["index"]: r for r in alone["requests"]}
    differ = [r["index"] for r in together["requests"]
              if "error" in r or r.get("output_ids") != by_index[r["index"]].get("output_ids")]
    print(json.dumps({
        "summary": "pool pressure at temperature 0",
        "identical": not differ and not alone["failed"],
        "differ": differ,
        "pressure": together["counters"].get("preemptions", 0) > 0,
        "preemptions": together["counters"].get("preemptions", 0),
        "prefix_cache_hits": alone["counters"].get("prefix_cache_hits", 0)
        + together["counters"].get("prefix_cache_hits", 0),
        "nan_requests": alone["counters"]["nan_requests"] + together["counters"]["nan_requests"],
        "failed": alone["failed"] + together["failed"],
        "alone_sha256": alone["outputs_sha256"],
        "together_sha256": together["outputs_sha256"],
        "streams": args.streams, "documents": args.documents, "tokens": args.tokens,
        "max_tokens": args.max_tokens,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
