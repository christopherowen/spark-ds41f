#!/usr/bin/env python3
"""Batch-invariance probe.

Sends the 40 greedy_probe prompts (raw completions, T=0, seed 1, 128 tokens) and records
text, token ids and log-probabilities, optionally while background streams keep the
server busy, or with the probe prompts themselves sent concurrently.

usage: bi_probe.py ROOT OUT.json [--bg N] [--bg-temp T] [--bg-tokens K] [--bg-top-p P]
                   [--logprobs K] [--conc C] [--tokens N] [--prompts N] [--settle S]
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash-spark-vllm")
TOPICS = ["the history of the Roman aqueducts", "how a transformer attention head works", "a recipe for sourdough bread", "why the sky is blue", "the rules of chess for a beginner", "a haiku sequence about winter", "the plot of a heist movie", "how to change a bicycle tire", "the economics of solar power", "a letter of complaint to an airline", "Python code to parse a CSV file", "the difference between TCP and UDP", "a product description for a hiking boot", "the causes of the French Revolution", "a bedtime story about a fox", "how vaccines train the immune system", "the geology of the Grand Canyon", "a SQL query explanation for a join", "the life cycle of a star", "tips for learning Spanish"]
PROMPTS = []
for t in TOPICS:
    PROMPTS.append(f"Write a short, detailed explanation of {t}.\n\n")
    PROMPTS.append(f"Here are five surprising facts about {t}:\n1.")


def post(root, body, timeout=900):
    req = urllib.request.Request(root + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--bg", type=int, default=0, help="background streams")
    ap.add_argument("--bg-temp", type=float, default=0.8)
    ap.add_argument("--bg-tokens", type=int, default=256)
    ap.add_argument("--bg-top-p", type=float, default=None)
    ap.add_argument("--bg-seed", type=int, default=None, help="fixed seed for background streams")
    ap.add_argument("--logprobs", type=int, default=0)
    ap.add_argument("--conc", type=int, default=1, help="probe prompts in flight at once")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--settle", type=float, default=5.0)
    a = ap.parse_args()

    stop = threading.Event()
    bg_done = [0] * a.bg
    bg_err = [0] * a.bg

    def bg(i):
        while not stop.is_set():
            body = {"model": MODEL, "prompt": f"Story {i}: once upon a time", "max_tokens": a.bg_tokens, "temperature": a.bg_temp}
            if a.bg_top_p is not None:
                body["top_p"] = a.bg_top_p
            if a.bg_seed is not None:
                body["seed"] = a.bg_seed + i
            try:
                post(a.root, body, 600)
                bg_done[i] += 1
            except Exception:
                bg_err[i] += 1
                time.sleep(0.5)

    threads = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(a.bg)]
    for t in threads:
        t.start()
    if a.bg:
        time.sleep(a.settle)

    def one(p):
        body = {"model": MODEL, "prompt": p, "max_tokens": a.tokens, "temperature": 0, "seed": 1}
        if a.logprobs:
            body["logprobs"] = a.logprobs
            body["return_tokens_as_token_ids"] = True
        t0 = time.time()
        d = post(a.root, body)
        c = d["choices"][0]
        rec = {"text": c["text"], "finish": c.get("finish_reason"), "secs": round(time.time() - t0, 3),
               "completion_tokens": d.get("usage", {}).get("completion_tokens")}
        lp = c.get("logprobs")
        if lp:
            rec["tokens"] = lp.get("tokens")
            rec["token_logprobs"] = lp.get("token_logprobs")
            rec["top_logprobs"] = lp.get("top_logprobs")
        return rec

    prompts = PROMPTS[: a.prompts]
    t0 = time.time()
    if a.conc > 1:
        with ThreadPoolExecutor(a.conc) as ex:
            recs = list(ex.map(one, prompts))
    else:
        recs = [one(p) for p in prompts]
    wall = time.time() - t0
    stop.set()
    json.dump({"root": a.root, "args": vars(a), "started": t0, "wall_s": round(wall, 2), "bg_requests_done": sum(bg_done),
               "bg_errors": sum(bg_err), "prompts": prompts, "outputs": [r["text"] for r in recs], "records": recs},
              open(a.out, "w"))
    print(f"wrote {a.out} n={len(recs)} wall={wall:.1f}s bg_done={sum(bg_done)} bg_err={sum(bg_err)}")


if __name__ == "__main__":
    main()
