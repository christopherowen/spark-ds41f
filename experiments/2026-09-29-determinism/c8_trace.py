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


def on_nodes(command):
    for node in NODES:
        subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c '{command}'"],
                       check=True, capture_output=True)


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
    on_nodes(f"mkdir -p {LOGDIR}; rm -f {LOGDIR}/rank*; touch {LOGDIR}/reset")
    time.sleep(2.5)  # the watcher polls once a second
    tags = [f"trace-c8-m{m}-r{r}" for r in range(STREAMS)]
    targets = [{} for _ in tags]
    threads = [threading.Thread(target=complete, args=(tag, keep)) for tag, keep in zip(tags, targets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    on_nodes(f"touch {LOGDIR}/dump")
    time.sleep(8)
    logs = os.path.join(OUT, f"logs-m{m}")
    for node in NODES:
        node_dir = os.path.join(logs, node)
        os.makedirs(node_dir, exist_ok=True)
        files = subprocess.run(
            ["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c 'cd {LOGDIR} && ls rank*'"],
            check=True, capture_output=True, text=True).stdout.split()
        for f in files:
            with open(os.path.join(node_dir, f), "wb") as handle:
                subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken cat {LOGDIR}/{f}"],
                               check=True, stdout=handle)
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
