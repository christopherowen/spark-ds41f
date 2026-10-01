#!/usr/bin/env python3
"""Eight identical requests at once, traced; each stream becomes a run for analyze_trace3.py.

usage: c8_trace.py BASE_URL OUT_DIR [--rounds R] [--tokens T]   (on dgx1, a trace arm serving)

The bench's eight concurrent copies of one prompt split with every fix: the
stream the scheduler prefilled alone (a small step) wrote one text, the seven
prefilled together (one sequence-parallel step) another. Per round: reset the
debug logs, send eight copies of a ~70-token prompt at once (distinct cache
salts, so each prefills), dump when all are done, copy every rank's logs once
to OUT_DIR/logs-m<round>/<node>/, and make OUT_DIR/c8-m<round>-r<stream>/
(target.json, node links to the shared logs), so analyze_trace3.py OUT_DIR c8
compares every stream with stream 0 by row position and token.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import trace_io

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ROUNDS, TOKENS, STREAMS = int(arg("--rounds", "2")), int(arg("--tokens", "16")), 8
PROMPT = ("Here are three notes about lighthouses. Note 0: the red lighthouse at the northern cape was built "
          "in 1700 and first lit with whale oil. Note 1: the red lighthouse at a harbour mouth was built in 1707 "
          "and first lit with whale oil. Note 2: the red lighthouse at an offshore reef was built in 1714. "
          "Summarize the notes in one sentence.")
NODES = ("dgx1", "dgx2", "dgx3")
LOGDIR = "/cache/kkref/moe-checksums"
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]




def complete(request_id, keep):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": TOKENS,
            "temperature": 0, "cache_salt": uuid.uuid4().hex, "chat_template_kwargs": {"thinking": False},
            "logprobs": True, "top_logprobs": 1, "return_tokens_as_token_ids": True}
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Request-Id": request_id})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    content = data["choices"][0]["logprobs"]["content"]
    keep.update(tokens=[c["token"] for c in content], logprobs=[c["logprob"] for c in content],
                prompt_tokens=data["usage"]["prompt_tokens"], seconds=round(time.monotonic() - started, 3),
                tokens_ids=[int(c["token"].split(":", 1)[1]) for c in content if c["token"].startswith("token_id:")])


for m in range(ROUNDS):
    trace_io.reset_logs()
    tags = [f"trace-c8-m{m}-r{r}" for r in range(STREAMS)]
    targets = [{} for _ in tags]
    threads = [threading.Thread(target=complete, args=(tag, keep)) for tag, keep in zip(tags, targets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    trace_io.dump_and_wait()
    logs = os.path.join(OUT, f"logs-m{m}")
    trace_io.fetch_logs({node: os.path.join(logs, node) for node in NODES})
    for r, (tag, target) in enumerate(zip(tags, targets)):
        run_dir = os.path.join(OUT, f"c8-m{m}-r{r}")
        os.makedirs(run_dir, exist_ok=True)
        for node in NODES:
            link = os.path.join(run_dir, node)
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", f"logs-m{m}", node), link)
        json.dump(dict(target, tag=tag, mix=m), open(os.path.join(run_dir, "target.json"), "w"))
        print(json.dumps({"run": tag, "tokens": len(target["tokens"]), "seconds": target["seconds"],
                          "first_logprob": target["logprobs"][0], "text_head": "".join(target["tokens"][:8])}),
              flush=True)
