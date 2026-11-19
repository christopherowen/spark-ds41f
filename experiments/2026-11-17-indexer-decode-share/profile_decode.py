#!/usr/bin/env python3
"""One stream's decode steps at increasing context depth, timed and then profiled.

usage: profile_decode.py BASE_URL [--depths 4096,262144,1000000] [--tokens 192] [--replays 30] [--settle 30]
                         [--no-profile] [--repeats 2] [--real-rows]

A profiler-enabled arm with VLLM_SERVER_DEV_MODE=1 must be serving. For each depth, in order:

1. fill: a cold source-text document of about that many tokens (servebench's prefill
   document, seed = the depth's index) with one output token, so its prefix is cached;
2. real: the same request again for LEAD + tokens output tokens (temperature 0, thinking
   off, ignore_eos), not profiled. The decode span runs from LEAD output chunks in to
   the end; its engine steps come from the server's step counter;
3. real-profiled: the same request once more, with the torch profiler started LEAD chunks
   in and stopped when the stream ends, so the window holds decode steps only;
4. grid: the engine grid's fixed decode step for one request at 3 and 6 rows at that
   context length (rounded down to a 4,096 multiple, at most the server's limit less
   8,192), not profiled;
5. grid-profiled: the same grid under the profiler.

One JSON line per phase. Every profiler window writes one trace per rank; the windows'
order is the traces' order.

With --no-profile (a server without --profiler-config) the profiled phases are left out
and phase 2 runs --repeats times, so the result is timing alone.

With --real-rows (a server whose engine grid has real_rows) two more phases follow:

6. grid-real: the same fixed steps with real rows, which route to experts and read a
   context the server fills first, not profiled;
7. grid-real-profiled: the same under the profiler, on the context phase 6 filled.
"""
import json
import sys
import time
import uuid
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spark_ds41f import sonar  # noqa: E402

engine = sonar.module("engine")
prefill = sonar.module("prefill")
prom = sonar.module("prom")

LEAD = 12  # output chunks before the decode span starts: the uncached tail's prefill is over


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BASE = sys.argv[1].rstrip("/")
DEPTHS = [int(value) for value in option("--depths", "4096,262144,1000000").split(",")]
TOKENS = int(option("--tokens", "192"))
REPLAYS = int(option("--replays", "30"))
SETTLE = float(option("--settle", "30"))
PROFILE = "--no-profile" not in sys.argv
REAL_ROWS = "--real-rows" in sys.argv
WINDOWS = 3 if REAL_ROWS else 2  # profiler windows a depth
REPEATS = int(option("--repeats", "2")) if not PROFILE else 1
QUESTION = "Explain the document above in detail, file by file and function by function."


def get(path, timeout=60):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.load(response)


def post(path, body=None, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read()
        return json.loads(text) if text.strip().startswith(b"{") else None


def emit(**record):
    print(json.dumps(record), flush=True)


def counters():
    return prom.scrape(BASE, prom.VLLM) or {}


def stream(body, at_lead=None, timeout=3600):
    """Stream one chat request. ``at_lead`` runs once, LEAD output chunks in. Returns the
    time sent, of the first chunk, at the lead mark (after ``at_lead``) and at the end,
    the chunk counts and the usage."""
    payload = dict(body, stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True})
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    record = {"sent": time.time(), "first": None, "lead": None, "end": None, "chunks": 0,
              "lead_completion_tokens": None, "usage": {}}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(f"server error in the stream: {str(event['error'])[:300]}")
            record["usage"] = event.get("usage") or record["usage"]
            if not any((choice.get("delta") or {}).get(key) for choice in event.get("choices") or []
                       for key in ("content", "reasoning_content", "reasoning")):
                continue
            record["chunks"] += 1
            if record["first"] is None:
                record["first"] = time.time()
            if record["chunks"] == LEAD and record["lead"] is None:
                record["lead_completion_tokens"] = record["usage"].get("completion_tokens")
                if at_lead:
                    at_lead()
                record["lead"] = time.time()
    record["end"] = time.time()
    return record


def span(record, before, after):
    """The decode span from the lead mark to the stream's end: seconds, engine steps and
    output tokens in it (the server's counters), and milliseconds per step."""
    seconds = record["end"] - record["lead"]
    steps = after.get("steps", 0) - before.get("steps", 0)
    tokens = after.get("generation_tokens", 0) - before.get("generation_tokens", 0)
    return {"span_s": round(seconds, 4), "steps": steps, "output_tokens": tokens,
            "step_ms": round(1000 * seconds / steps, 4) if steps else None,
            "tokens_per_step": round(tokens / steps, 4) if steps else None,
            "drafts": after.get("drafts", 0) - before.get("drafts", 0),
            "verified_rows": after.get("draft_tokens", 0) - before.get("draft_tokens", 0)}


def settle():
    time.sleep(SETTLE)  # the ranks write their traces before the next window starts


model = get("/v1/models")["data"][0]
MODEL, LIMIT = model["id"], model.get("max_model_len")
emit(phase="start", model=MODEL, max_model_len=LIMIT, depths=DEPTHS, tokens=TOKENS, replays=REPLAYS, lead=LEAD)

for index, depth in enumerate(DEPTHS):
    post("/reset_prefix_cache", None, 120)
    text, counted = prefill.document(BASE, MODEL, depth, seed=index, vllm=True)
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"{text}\n\n{QUESTION}"}],
            "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"thinking": False},
            "cache_salt": uuid.uuid4().hex}

    filled = stream(dict(body, max_tokens=1))
    emit(phase="fill", depth=depth, document_tokens=counted, usage=filled["usage"],
         ttft_s=round(filled["first"] - filled["sent"], 3) if filled["first"] else None)

    marks = {}
    for repeat in range(REPEATS):
        timed = stream(dict(body, max_tokens=LEAD + TOKENS), at_lead=lambda: marks.update(before=counters()))
        emit(phase="real", depth=depth, repeat=repeat, usage=timed["usage"], chunks=timed["chunks"],
             ttft_s=round(timed["first"] - timed["sent"], 3),
             lead_completion_tokens=timed["lead_completion_tokens"], **span(timed, marks["before"], counters()))

    def start_profile():
        post("/start_profile")
        marks["profiled"] = counters()

    if PROFILE:
        began = time.time()
        try:
            profiled = stream(dict(body, max_tokens=LEAD + TOKENS), at_lead=start_profile)
            after = counters()
        finally:
            post("/stop_profile")
        emit(phase="real-profiled", depth=depth, window=WINDOWS * index, usage=profiled["usage"], chunks=profiled["chunks"],
             request_sent=round(began, 3), profile_started=round(profiled["lead"], 3),
             stream_ended=round(profiled["end"], 3), lead_completion_tokens=profiled["lead_completion_tokens"],
             **span(profiled, marks["profiled"], after))
        settle()

    context = min(depth, LIMIT - 8192) if LIMIT else depth
    context = context // 4096 * 4096 or context  # whole 4K multiples, as the release grid's depths are
    shapes = [{"kind": "decode", "num_reqs": 1, "num_tokens": rows, "context_len": context} for rows in (3, 6)]
    emit(phase="grid", depth=depth, context_len=context,
         grid=engine.summarize(engine.run(BASE, shapes, replays=REPLAYS, warmup=2)))
    if PROFILE:
        post("/start_profile")
        began = time.time()
        try:
            result = engine.summarize(engine.run(BASE, shapes, replays=REPLAYS, warmup=2))
        finally:
            ended = time.time()
            post("/stop_profile")
        emit(phase="grid-profiled", depth=depth, context_len=context, window=WINDOWS * index + 1, replays=REPLAYS,
             warmup=2, start=round(began, 3), end=round(ended, 3), grid=result)
        settle()

    if REAL_ROWS:
        real = [dict(shape, real_rows=True) for shape in shapes]
        began = time.time()
        result = engine.summarize(engine.run(BASE, real, replays=REPLAYS, warmup=2))
        emit(phase="grid-real", depth=depth, context_len=context, seconds=round(time.time() - began, 1), grid=result)
        if PROFILE:
            post("/start_profile")
            began = time.time()
            try:
                result = engine.summarize(engine.run(BASE, real, replays=REPLAYS, warmup=2, fill_real_context=False))
            finally:
                ended = time.time()
                post("/stop_profile")
            emit(phase="grid-real-profiled", depth=depth, context_len=context, window=WINDOWS * index + 2,
                 replays=REPLAYS, warmup=2, start=round(began, 3), end=round(ended, 3), grid=result)
            settle()

emit(phase="done", windows=WINDOWS * len(DEPTHS) if PROFILE else 0)
