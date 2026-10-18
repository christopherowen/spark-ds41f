"""Load generation.

- ``closed``: N workers each keep one request in flight, starting the next as soon as
  theirs ends, so the server sees a steady N with new prompts' prefills mixed into decode
  steps, as it does under real traffic.
- ``poisson``: requests arrive as a Poisson process at a fixed rate whatever the server
  does (open loop); queueing shows up as latency.
- ``burst``: N requests at once and nothing else, the old distinct-stream shape, kept for
  comparison with earlier numbers.

Closed and Poisson runs warm up first, then measure for a fixed duration split into
equal sub-windows (batch means: each sub-window is one sample for the intervals), then
stop issuing and let the in-flight requests drain outside the window. Server counters
are read at every sub-window boundary, and once more just before the run's first request
and after its last has drained: between those two fences the server finished every one of
the run's requests, so anything else it finished there was someone else's.

With ``context_tokens`` (closed mode) each stream owns a long document and asks every
question after it, under one cache salt per stream. Each document is prefilled once
before the clock starts, so the window measures decode at that depth plus each
question's short prefill, as with a long conversation or a coding agent's repository
context. The window's prefix-cache hit rate shows whether the documents stayed cached.
"""

import itertools
import random
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass

import json
import urllib.request

from . import client, prom
from . import prompts as prompt_sets


@dataclass
class Load:
    mode: str = "closed"          # closed | poisson | burst
    concurrency: int = 8          # closed and burst
    rate: float = 1.0             # poisson: requests per second
    warmup_s: float = 60.0
    duration_s: float = 300.0
    batches: int = 10
    max_tokens: int = 256
    fixed_length: bool = True     # every request writes exactly max_tokens (vLLM adapter)
    seed: int = 0
    max_inflight: int = 512       # poisson safety cap; arrivals past it are counted, not sent
    context_tokens: int = 0       # closed: each stream's cached document before every question

    def describe(self) -> dict:
        return asdict(self)


def payload(model: str, prompt: dict, load: Load, *, vllm: bool, extra_body: dict,
            document: str | None = None, salt: str | None = None) -> dict:
    text = prompt["text"] if document is None else f"{document}\n\n{prompt['text']}"
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": load.max_tokens, "temperature": 0}
    if vllm:
        # A fresh salt per request: a prompt seen earlier in the run must not hit the
        # prefix cache, or repeats would skip their prefill. A stream's own document
        # keeps one salt so that it does.
        body["cache_salt"] = salt or uuid.uuid4().hex
        if load.fixed_length:
            body["ignore_eos"] = True
            body["min_tokens"] = load.max_tokens
    body.update(extra_body)
    return body


def _sleep_until(moment: float, clock) -> None:
    while True:
        remaining = moment - clock()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.5))


def token_count(base_url: str, model: str, text: str) -> int | None:
    """Tokens in ``text`` by the server's tokenizer (vLLM /tokenize), or None."""
    request = urllib.request.Request(base_url.rstrip("/") + "/tokenize",
                                     data=json.dumps({"model": model, "prompt": text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response).get("count")
    except Exception:  # noqa: BLE001 - sizing falls back to the characters-per-token estimate
        return None


def documents(base_url: str, model: str, load: Load, vllm: bool) -> list[dict]:
    """One document per stream, sized to ``load.context_tokens`` by the server's tokenizer
    when it has one (one rescaling step)."""
    out = []
    for slot in range(load.concurrency):
        seed = load.seed * 1000 + slot
        text = prompt_sets.source_text(load.context_tokens, seed)
        counted = token_count(base_url, model, text) if vllm else None
        if counted:
            ratio = len(text) / counted
            text = prompt_sets.source_text(load.context_tokens, seed, chars_per_token=ratio)
            counted = token_count(base_url, model, text)
        out.append({"slot": slot, "text": text, "tokens": counted, "salt": uuid.uuid4().hex})
    return out


def run(base_url: str, model: str, prompts: list[dict], load: Load, *, vllm: bool = True,
        extra_body: dict | None = None, metrics: dict | None = None, clock=time.monotonic,
        log=lambda message: print(message, file=sys.stderr, flush=True)) -> dict:
    """Run ``load`` and return the raw run: records, window bounds, boundary scrapes."""
    extra_body = extra_body or {}
    metrics = prom.VLLM if metrics is None else metrics
    order = list(prompts)
    random.Random(load.seed).shuffle(order)
    counter = itertools.count()
    lock = threading.Lock()
    records: list[dict] = []
    rejected = [0]
    docs: list[dict] = []
    timeout = 900.0
    if load.context_tokens:
        if load.mode != "closed":
            raise ValueError("context_tokens needs closed mode")
        docs = documents(base_url, model, load, vllm)
        timeout = max(timeout, load.context_tokens / 500)
        for doc in docs:  # one at a time: each prefill is bounded, and none is evicted mid-run
            began = clock()
            body = payload(model, order[doc["slot"] % len(order)], load, vllm=vllm,
                           extra_body=extra_body | {"max_tokens": 1, "min_tokens": 1},
                           document=doc["text"], salt=doc["salt"])
            primed = client.stream_chat(base_url, body, continuous_usage=vllm, timeout=timeout, clock=clock)
            doc["prime_s"] = round(clock() - began, 2)
            doc["prime_error"] = primed["error"]
            log(f"servebench: document {doc['slot'] + 1}/{len(docs)} "
                f"({doc['tokens'] or load.context_tokens} tokens) primed in {doc['prime_s']:.0f} s")

    def send(slot: int) -> None:
        index = next(counter)
        prompt = order[index % len(order)]
        doc = docs[slot] if docs else None
        body = payload(model, prompt, load, vllm=vllm, extra_body=extra_body,
                       document=doc["text"] if doc else None, salt=doc["salt"] if doc else None)
        record = client.stream_chat(base_url, body, continuous_usage=vllm, timeout=timeout, clock=clock)
        record.update(prompt_id=prompt["id"], slot=slot, index=index)
        with lock:
            records.append(record)

    documents_meta = [{key: doc[key] for key in ("slot", "tokens", "prime_s", "prime_error")} for doc in docs]
    fences = {"times": [clock()], "scrapes": [prom.scrape(base_url, metrics)]}

    def close_fence() -> None:
        fences["times"].append(clock())
        fences["scrapes"].append(prom.scrape(base_url, metrics))

    start = clock()
    if load.mode == "burst":
        threads = [threading.Thread(target=send, args=(slot,)) for slot in range(load.concurrency)]
        before = prom.scrape(base_url, metrics)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        after = prom.scrape(base_url, metrics)
        close_fence()
        return {"records": records, "start": start, "window": None,
                "boundaries": [], "scrapes": [before, after], "scrape_times": [start, clock()],
                "rejected": 0, "fences": fences}

    w0 = start + load.warmup_s
    w1 = w0 + load.duration_s
    boundaries = [w0 + k * load.duration_s / load.batches for k in range(load.batches + 1)]
    scrapes: list = [None] * len(boundaries)
    scrape_times: list = [None] * len(boundaries)

    def scraper() -> None:
        for k, boundary in enumerate(boundaries):
            _sleep_until(boundary, clock)
            scrape_times[k] = clock()
            scrapes[k] = prom.scrape(base_url, metrics)
            if k:
                log(f"servebench: window {k}/{load.batches} done")

    scraping = threading.Thread(target=scraper, daemon=True)
    scraping.start()
    log(f"servebench: {load.mode} warm-up {load.warmup_s:.0f} s, then {load.duration_s:.0f} s "
        f"in {load.batches} windows")
    workers: list[threading.Thread] = []
    if load.mode == "closed":
        def worker(slot: int) -> None:
            while clock() < w1:
                send(slot)
        workers = [threading.Thread(target=worker, args=(slot,), daemon=True)
                   for slot in range(load.concurrency)]
        for thread in workers:
            thread.start()
    elif load.mode == "poisson":
        arrivals = random.Random(load.seed + 1)
        moment = start
        while True:
            moment += arrivals.expovariate(load.rate)
            if moment >= w1:
                break
            _sleep_until(moment, clock)
            with lock:
                inflight = sum(1 for thread in workers if thread.is_alive())
            if inflight >= load.max_inflight:
                rejected[0] += 1
                continue
            thread = threading.Thread(target=send, args=(len(workers),), daemon=True)
            thread.start()
            workers.append(thread)
    else:
        raise ValueError(f"unknown mode {load.mode!r}")
    for thread in workers:
        thread.join()
    scraping.join()
    close_fence()
    return {"records": records, "start": start, "window": [w0, w1], "boundaries": boundaries,
            "scrapes": scrapes, "scrape_times": scrape_times, "rejected": rejected[0],
            "documents": documents_meta, "fences": fences}
