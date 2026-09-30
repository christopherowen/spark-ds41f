#!/usr/bin/env python3
"""Repeat a tagged target beside background mixes, with per-step traces on every rank.

usage: trace_mixes.py BASE_URL OUT_DIR [--repeats R] [--tokens T] [--prompts json,prose]
(on dgx1, a batch-trace arm serving: SPARK3_MOE_CHECKSUM_DIR=/cache/kkref/moe-checksums)

For each target prompt, mix and repeat: reset the debug logs on all three
nodes, start the background requests, send the target 0.3 s later with
X-Request-Id "trace-<prompt>-m<mix>-r<rep>" (so the model runner's step log
names its rows), dump the logs as soon as the target is done, and copy every rank's
runner, tag and schedule logs to OUT_DIR/<prompt>-m<mix>-r<rep>/<node>/, with
the target's tokens (as token ids) and logprobs in target.json. Thinking is off.
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


REPEATS, TOKENS = int(arg("--repeats", "3")), int(arg("--tokens", "128"))
PROMPTS = {
    "json": "Return a JSON object describing three fictional planets with name, mass and moons.",
    "prose": "Write a short paragraph about the history of the lighthouse.",
}
PROMPTS = {k: v for k, v in PROMPTS.items() if k in arg("--prompts", "json,prose").split(",")}
BACKGROUND = [
    ("List ten prime numbers and explain why each is prime.", 192),
    ("Write a haiku about rain, then explain its imagery.", 96),
    ("Summarize the plot of a heist film you invent.", 256),
    ("Explain how a hash map handles collisions.", 160),
    ("Describe a walk through a market in Marrakech.", 224),
    ("Give a recipe for lentil soup with timings.", 128),
    ("Compare TCP and UDP for game networking.", 256),
]
MIXES = [[], [0, 3], [1, 2, 4, 6], [0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]]
NODES = ("dgx1", "dgx2", "dgx3")
LOGDIR = "/cache/kkref/moe-checksums"
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]


def on_nodes(command):
    for node in NODES:
        subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c '{command}'"],
                       check=True, capture_output=True)


def complete(prompt, tokens, request_id=None, keep=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}}
    headers = {"Content-Type": "application/json"}
    if request_id:
        body.update(logprobs=True, top_logprobs=1, return_tokens_as_token_ids=True)
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    if keep is not None:
        content = data["choices"][0]["logprobs"]["content"]
        keep.update(tokens=[c["token"] for c in content], logprobs=[c["logprob"] for c in content],
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    tokens_ids=[int(c["token"].split(":", 1)[1]) for c in content
                                if c["token"].startswith("token_id:")])


for name, prompt in PROMPTS.items():
    for m, mix in enumerate(MIXES):
        for r in range(REPEATS):
            tag = f"trace-{name}-m{m}-r{r}"
            run_dir = os.path.join(OUT, f"{name}-m{m}-r{r}")
            on_nodes(f"mkdir -p {LOGDIR}; rm -f {LOGDIR}/rank*; touch {LOGDIR}/reset")
            time.sleep(2.5)  # the watcher polls once a second
            threads = [threading.Thread(target=complete, args=(BACKGROUND[i][0], BACKGROUND[i][1]))
                       for i in mix]
            for t in threads:
                t.start()
            if threads:
                time.sleep(0.3)
            target = {}
            complete(prompt, TOKENS, request_id=tag, keep=target)
            # Dump as soon as the target is done: the background may run on,
            # and its later steps would wrap the logs over the target's.
            on_nodes(f"touch {LOGDIR}/dump")
            time.sleep(8)
            for t in threads:
                t.join()
            for node in NODES:
                node_dir = os.path.join(run_dir, node)
                os.makedirs(node_dir, exist_ok=True)
                files = subprocess.run(
                    ["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c 'cd {LOGDIR} && ls rank*'"],
                    check=True, capture_output=True, text=True).stdout.split()
                for f in files:
                    with open(os.path.join(node_dir, f), "wb") as handle:
                        subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken cat {LOGDIR}/{f}"],
                                       check=True, stdout=handle)
            json.dump(dict(target, tag=tag, mix=mix), open(os.path.join(run_dir, "target.json"), "w"))
            print(json.dumps({"run": tag, "background": len(mix), "tokens": len(target["tokens"]),
                              "text_head": "".join(target["tokens"][:12])}), flush=True)
