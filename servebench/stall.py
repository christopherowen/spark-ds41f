"""Decode stalls while a long cold prompt prefills.

Every other stream on the server waits while a long prompt prefills: each engine step
that carries one of its chunks is longer than a decode step, and the long-prefill budget
sets how much longer. ``streams`` requests decode (fixed length, fresh cache salts, the
default prompt set); after a warm-up, the prefill step's cold real-text document of
``size`` tokens is sent for one output token. A stream's gaps between output chunks are
its step times: those that ended before the document was sent are the decode baseline,
and those that overlap the document's prefill are the stall. The prefill starts after the
server has tokenized and admitted the document, and until then the streams decode at their
baseline: a stream's overlapping gaps before its first one longer than ``LONG_GAP_FACTOR``
baseline steps are the admission (``admission_s``, sent to that gap's start, the streams'
median), and the stall's gaps are those from it to the first token. The longest is the worst
wait any stream saw; their mean, p90 and p99 show how slowly decoding ran meanwhile, and the
long share is the fraction of them a prefill chunk held up. The median is not compared: with
the admission's gaps counted, about as many short gaps as chunk steps put it on either side
from one repeat to the next. Each repeat keeps every overlapping gap (stream, start after
the document was sent, length, whether it was the admission's) and, with vLLM, the engine
steps between the document's send and its first token, from the step-tokens histogram (vLLM
counts a prompt's tokens at its first token, so the histogram does not tell chunk steps from
decode steps; the count does: about one a stream's gap). The streams stop ``tail_s`` after
the document's first token, so each records the chunk that ends its wait. A repeat fails unless every stream
decoded across the whole prefill.

``probes`` sends short cold requests (``probe_tokens`` of real text, fresh cache salts)
that many seconds after the document: their time to first token is what a new
conversation waits while a long prompt prefills. Only probes sent before the document's
first token count. Leave them free slots: streams + 1 + the probes in flight must not
exceed the server's ``--max-num-seqs``, or the probes measure slot queueing instead.
"""
import threading
import time
import uuid

from . import client, prefill, prom, prompts, stats

SCHEMA = "servebench-stall/1"
# A gap this many times the baseline step's median is a step a prefill chunk held up.
LONG_GAP_FACTOR = 2.0
STEP_HISTOGRAM = "vllm:iteration_tokens_total"


def split_gaps(events: list, sent: float, first: float) -> tuple[list[float], list[float]]:
    """A stream's gaps between consecutive output chunks: those ending by ``sent`` (the
    baseline) and those overlapping the prefill, from ``sent`` to ``first``."""
    before: list[float] = []
    during = [length for _, length in overlapping(events, sent, first, before)]
    return before, during


def overlapping(events: list, sent: float, first: float, before: list | None = None) -> list[tuple[float, float]]:
    """The (start, length) of a stream's gaps that overlap the prefill; the gaps ending by
    ``sent`` go to ``before``."""
    gaps = []
    for (start, _), (end, _) in zip(events, events[1:]):
        if end <= sent:
            if before is not None:
                before.append(end - start)
        elif start < first:
            gaps.append((start, end - start))
    return gaps


def admission(gaps: list[tuple[float, float]], baseline: float | None) -> int:
    """How many of a stream's overlapping gaps came before the prefill's first chunk step:
    those before the first longer than ``LONG_GAP_FACTOR`` baseline steps (none if no gap is)."""
    if not baseline:
        return 0
    for index, (_, length) in enumerate(gaps):
        if length > LONG_GAP_FACTOR * baseline:
            return index
    return 0


def engine_window(start: str | None, end: str | None) -> dict | None:
    """The engine steps between two /metrics scrapes, and their step-tokens histogram."""
    if start is None or end is None:
        return None
    before = prom.by_label(start, STEP_HISTOGRAM + "_bucket", "le")
    after = prom.by_label(end, STEP_HISTOGRAM + "_bucket", "le")
    buckets = {bound: after[bound] - before.get(bound, 0.0) for bound in sorted(after, key=float)}
    if not buckets:
        return None
    return {"steps": buckets[max(buckets, key=float)],
            "buckets": [[bound, count] for bound, count in buckets.items()]}


def engine_note(sample: dict) -> str:
    """The engine's steps during the prefill against each stream's gaps, for the log."""
    engine = sample.get("engine")
    if not engine or not sample["covering"]:
        return ""
    return (f", engine steps {engine['steps']:.0f}, "
            f"overlapping gaps a stream {len(sample['gaps']) / sample['covering']:.1f}")


def long_share(during: list[float], baseline: float | None) -> float | None:
    """The fraction of the overlapping gaps longer than ``LONG_GAP_FACTOR`` baseline steps."""
    if not during or not baseline:
        return None
    return sum(gap > LONG_GAP_FACTOR * baseline for gap in during) / len(during)


def stream_body(model: str, text: str, max_tokens: int, vllm: bool, extra_body: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "temperature": 0}
    if vllm:
        body.update(cache_salt=uuid.uuid4().hex, ignore_eos=True, min_tokens=max_tokens)
    body.update(extra_body)
    return body


def probe_body(model: str, text: str, max_tokens: int, vllm: bool, extra_body: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": f"{text}\n\n{prefill.QUESTION}"}],
            "max_tokens": max_tokens, "temperature": 0}
    if vllm:
        body["cache_salt"] = uuid.uuid4().hex
    body.update(extra_body)
    return body


def measure(base_url: str, model: str, size: int, seed: int, texts: list[str], *, max_tokens: int,
            warmup_s: float, tail_s: float, vllm: bool, extra_body: dict, probes: tuple = (),
            probe_tokens: int = 500, probe_max_tokens: int = 64, sleep=time.sleep) -> dict:
    """One prefill of ``size`` tokens under ``len(texts)`` decoding streams, with short
    probes sent ``probes`` seconds after it."""
    document, counted = prefill.document(base_url, model, size, seed, vllm)
    probe_texts = [prefill.document(base_url, model, probe_tokens, seed * 100 + 1 + i, vllm)[0]
                   for i in range(len(probes))]
    stop = threading.Event()
    records: list[dict | None] = [None] * len(texts)
    probe_records: list[dict | None] = [None] * len(probes)

    def decode(i: int) -> None:
        records[i] = client.stream_chat(base_url, stream_body(model, texts[i], max_tokens, vllm, extra_body),
                                        continuous_usage=vllm, stop=stop)

    def probe(i: int) -> None:
        sleep(probes[i])
        probe_records[i] = client.stream_chat(
            base_url, probe_body(model, probe_texts[i], probe_max_tokens, vllm, extra_body), continuous_usage=vllm)

    threads = [threading.Thread(target=decode, args=(i,), daemon=True) for i in range(len(texts))]
    for thread in threads:
        thread.start()
    sleep(warmup_s)
    probers = [threading.Thread(target=probe, args=(i,), daemon=True) for i in range(len(probes))]
    for thread in probers:
        thread.start()
    scraped = prom.fetch(base_url) if vllm else None
    record = client.stream_chat(base_url, probe_body(model, document, 1, vllm, extra_body),
                                continuous_usage=vllm, timeout=max(900.0, size / 500))
    engine = engine_window(scraped, prom.fetch(base_url) if scraped is not None else None)
    sleep(tail_s)
    stop.set()
    for thread in threads:
        thread.join()
    for thread in probers:
        thread.join()
    sent, first = record["sent"], record["first"]
    before, by_stream, covering = [], {}, 0
    for index, stream in enumerate(records):
        events = (stream or {}).get("events") or []
        if first is None or not events or events[0][0] >= sent or events[-1][0] < first:
            continue
        covering += 1
        by_stream[index] = overlapping(events, sent, first, before)
    before_s = stats.percentiles(before)
    during, gaps, admitted = [], [], []
    for index, stream_gaps in by_stream.items():
        waited = admission(stream_gaps, before_s.get("p50"))
        if stream_gaps:
            admitted.append(stream_gaps[waited][0] - sent if waited else 0.0)
        for k, (start, length) in enumerate(stream_gaps):
            gaps.append([index, round(start - sent, 4), round(length, 4), k < waited])
            if k >= waited:
                during.append(length)
    tokens = record["prompt_tokens"] or counted
    ttft = first - sent if first is not None else None
    error = record["error"]
    if error is None and covering < len(texts):
        error = f"{len(texts) - covering} of {len(texts)} streams did not decode across the prefill"
    probe_rows = []
    for offset, probe_record in zip(probes, probe_records):
        probe_record = probe_record or {}
        p_sent, p_first = probe_record.get("sent"), probe_record.get("first")
        probe_rows.append({"offset_s": offset,
                           "ttft_s": p_first - p_sent if p_sent is not None and p_first is not None else None,
                           "during": first is not None and p_sent is not None and p_sent < first,
                           "error": probe_record.get("error")})
    if error is None and any(row["error"] or row["ttft_s"] is None for row in probe_rows):
        error = "a probe failed"
    probe_ttfts = [row["ttft_s"] for row in probe_rows if row["during"] and row["ttft_s"] is not None]
    return {"size": size, "prompt_tokens": tokens, "ttft_s": ttft,
            "tps": tokens / ttft if tokens and ttft else None, "streams": len(texts), "covering": covering,
            "before_s": before_s, "during_s": stats.percentiles(during),
            "long_share": long_share(during, before_s.get("p50")), "gaps": gaps, "engine": engine,
            "admission_s": stats.percentiles(admitted).get("p50"),
            "stall_s": max(during) if during else None, "probes": probe_rows,
            "probe_ttft_s": stats.percentiles(probe_ttfts),
            "probe_ttft_max_s": max(probe_ttfts) if probe_ttfts else None, "error": error}


def run(base_url: str, model: str, sizes: list[int], *, streams: int = 8, max_tokens: int = 16384,
        warmup_s: float = 20.0, tail_s: float = 2.0, repeats: int = 3, vllm: bool = True,
        extra_body: dict | None = None, probes: tuple = (), probe_tokens: int = 500,
        probe_max_tokens: int = 64, log=lambda message: None, sleep=time.sleep) -> list[dict]:
    pool = [p["text"] for p in prompts.default_prompts()]
    samples = []
    for repeat in range(repeats):
        for index, size in enumerate(sizes):
            turn = repeat * len(sizes) + index
            texts = [pool[(turn * streams + i) % len(pool)] for i in range(streams)]
            sample = measure(base_url, model, size, 1000 * repeat + index, texts, max_tokens=max_tokens,
                             warmup_s=warmup_s, tail_s=tail_s, vllm=vllm, extra_body=extra_body or {},
                             probes=tuple(probes), probe_tokens=probe_tokens,
                             probe_max_tokens=probe_max_tokens, sleep=sleep)
            sample["repeat"] = repeat
            samples.append(sample)
            log(f"servebench: stall {size} repeat {repeat + 1}/{repeats}: "
                + (f"worst {sample['stall_s']:.2f} s, decode mean {sample['during_s']['mean']:.3f} s, "
                   f"p90 {sample['during_s']['p90']:.3f} s, p99 {sample['during_s']['p99']:.3f} s, "
                   f"long {100 * (sample['long_share'] or 0):.0f}%, "
                   f"admission {sample['admission_s'] or 0:.2f} s "
                   f"(baseline {sample['before_s'].get('p50', float('nan')):.3f} s)"
                   + engine_note(sample)
                   + (f", probe TTFT p50 {sample['probe_ttft_s']['p50']:.2f} s" if sample["probe_ttft_s"]["n"] else "")
                   if sample["error"] is None else f"failed ({sample['error']})"))
    return samples


def summarize(samples: list[dict]) -> list[dict]:
    rows = []
    for size in dict.fromkeys(s["size"] for s in samples):
        ok = [s for s in samples if s["size"] == size and s["error"] is None]
        series = {"stall_s": [s["stall_s"] for s in ok],
                  "during_mean_s": [s["during_s"]["mean"] for s in ok],
                  "admission_s": [s["admission_s"] for s in ok if s.get("admission_s") is not None],
                  "during_p90_s": [s["during_s"]["p90"] for s in ok],
                  "during_p99_s": [s["during_s"]["p99"] for s in ok],
                  "long_share": [s["long_share"] for s in ok if s.get("long_share") is not None],
                  "before_p50_s": [s["before_s"]["p50"] for s in ok if s["before_s"]["n"]],
                  "tps": [s["tps"] for s in ok],
                  "probe_ttft_p50_s": [s["probe_ttft_s"]["p50"] for s in ok if s.get("probe_ttft_s", {}).get("n")],
                  "probe_ttft_max_s": [s["probe_ttft_max_s"] for s in ok if s.get("probe_ttft_max_s") is not None]}
        rows.append({"size": size, "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
                     **{name: stats.describe(values) for name, values in series.items()},
                     "samples": {name: [round(v, 5) for v in values] for name, values in series.items()},
                     "failed": sum(1 for s in samples if s["size"] == size and s["error"] is not None)})
    return rows


METRICS = (("stall_s", "worst decode wait during {size:,} prefill (s)", False),
           ("during_p99_s", "decode step p99 during {size:,} prefill (s)", False),
           ("during_mean_s", "decode step mean during {size:,} prefill (s)", False),
           ("admission_s", "decode before {size:,} prefill's first chunk (s)", False),
           ("during_p90_s", "decode step p90 during {size:,} prefill (s)", False),
           ("long_share", "share of decode steps a chunk held during {size:,} prefill", False),
           ("tps", "prefill {size:,} tok/s under decode", True),
           ("probe_ttft_p50_s", "short request TTFT p50 during {size:,} prefill (s)", False),
           ("probe_ttft_max_s", "short request TTFT max during {size:,} prefill (s)", False))


def compare(base: list[dict], candidate: list[dict], *, tolerance_pct: float = 3.0) -> dict:
    """Size by size, each metric's change with its Welch interval over repeats."""
    by_size = {row["size"]: row for row in base}
    rows, warnings = [], []
    for row in candidate:
        other = by_size.get(row["size"])
        if other is None:
            warnings.append(f"stall {row['size']}: not in the base run")
            continue
        if other["prompt_tokens"] != row["prompt_tokens"]:
            warnings.append(f"stall {row['size']}: {other['prompt_tokens']} vs {row['prompt_tokens']} prompt tokens")
        for key, label, higher_is_better in METRICS:
            if key not in other["samples"] or key not in row["samples"]:
                continue
            test = stats.welch(other["samples"][key], row["samples"][key])
            if test is None:
                continue
            rows.append({"metric": label.format(size=row["size"]), "base": other[key]["mean"],
                         "candidate": row[key]["mean"], "change_pct": test["change_pct"],
                         "ci95_pct": test["ci95_pct"],
                         "verdict": stats.verdict(test["change_pct"], test["ci95_pct"], higher_is_better,
                                                  tolerance_pct)})
    return {"warnings": warnings, "rows": rows}
