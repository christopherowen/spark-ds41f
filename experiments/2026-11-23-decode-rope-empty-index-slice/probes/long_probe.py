#!/usr/bin/env python3
"""Long-context batch-invariance probe.

N distinct prompts of about TOKENS tokens each (token-id prompts, sized with /tokenize).
Greedy, seed 1, GEN new tokens, top-5 logprobs. Phases:
  cold   each prompt alone, first time (full prefill)
  solo   each prompt alone again (prefix-cached prefill)  <- reference
  conc1  all N in flight at once
  conc2  all N in flight at once
  solo2  each prompt alone again
usage: long_probe.py ROOT OUT.json [--tokens 20000] [--n 3] [--gen 128]
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_probe import MODEL, TOPICS  # noqa: E402

NOUNS = ["valve", "ledger", "beam", "relay", "pump", "crate", "sensor", "gasket", "rotor", "panel", "spool", "bracket"]
ADJ = ["amber", "spare", "cracked", "sealed", "north", "heavy", "idle", "bent", "clean", "late", "damp", "third"]


def post(root, path, body, timeout=900):
    req = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def text(j, sentences):
    x = 1234567 + 7919 * j
    out = [f"Maintenance log {j + 1} for the plant, kept in order. The subject index is {TOPICS[j % len(TOPICS)]}."]
    for i in range(sentences):
        x = (x * 1103515245 + 12345) % (1 << 31)
        a, n, b = ADJ[x % 12], NOUNS[(x >> 5) % 12], NOUNS[(x >> 11) % 12]
        out.append(f"Entry {i + 1}: the {a} {n} in bay {(x >> 7) % 40 + 1} was checked against the {b} and logged at {(x >> 13) % 24:02d}:{(x >> 17) % 60:02d}.")
    return " ".join(out)


def ask(root, ids, gen):
    t0 = time.time()
    d = post(root, "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": gen, "temperature": 0, "seed": 1, "logprobs": 5,
                                       "return_tokens_as_token_ids": True})
    c = d["choices"][0]
    return {"text": c["text"], "tokens": c["logprobs"]["tokens"], "lp": c["logprobs"]["token_logprobs"], "top": c["logprobs"]["top_logprobs"],
            "prompt_tokens": d["usage"]["prompt_tokens"], "secs": round(time.time() - t0, 2)}


def together(root, prompts, gen):
    out = [None] * len(prompts)
    gate = threading.Barrier(len(prompts))

    def run(i):
        gate.wait()
        out[i] = ask(root, prompts[i], gen)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def diff(a, b):
    m = min(len(a["tokens"]), len(b["tokens"]))
    fork = next((k for k in range(m) if a["tokens"][k] != b["tokens"][k]), None)
    upto = m if fork is None else fork
    lp = next((k for k in range(upto) if a["lp"][k] != b["lp"][k] or a["top"][k] != b["top"][k]), None)
    return fork, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--gen", type=int, default=128)
    a = ap.parse_args()
    tail = post(a.root, "/tokenize", {"model": MODEL, "prompt": "\n\nIn one paragraph, the log above shows that", "add_special_tokens": False})["tokens"]
    prompts = []
    for j in range(a.n):
        ids = post(a.root, "/tokenize", {"model": MODEL, "prompt": text(j, a.tokens // 14), "add_special_tokens": False})["tokens"]
        assert len(ids) >= a.tokens, (j, len(ids))
        prompts.append(ids[: a.tokens - len(tail)] + tail)
    ph = {}
    ph["cold"] = [ask(a.root, p, a.gen) for p in prompts]
    ph["solo"] = [ask(a.root, p, a.gen) for p in prompts]
    ph["conc1"] = together(a.root, prompts, a.gen)
    ph["conc2"] = together(a.root, prompts, a.gen)
    ph["solo2"] = [ask(a.root, p, a.gen) for p in prompts]
    assert all(r["prompt_tokens"] == a.tokens for r in ph["solo"])
    summary = {}
    for k in ("cold", "conc1", "conc2", "solo2"):
        ds = [diff(x, y) for x, y in zip(ph["solo"], ph[k])]
        summary[k] = {"tokens_identical": sum(f is None for f, _ in ds), "logprobs_identical": sum(f is None and l is None for f, l in ds),
                      "first_token_fork": [f for f, _ in ds], "first_logprob_diff": [l for _, l in ds]}
        print(f"{k:6s} vs solo: tokens identical {summary[k]['tokens_identical']}/{a.n}, logprobs identical {summary[k]['logprobs_identical']}/{a.n}, "
              f"fork at {summary[k]['first_token_fork']}, first lp diff at {summary[k]['first_logprob_diff']}")
    print("secs cold/solo/conc:", [r["secs"] for r in ph["cold"]], [r["secs"] for r in ph["solo"]], [r["secs"] for r in ph["conc1"]])
    json.dump({"root": a.root, "args": vars(a), "summary": summary, "phases": ph}, open(a.out, "w"))


if __name__ == "__main__":
    main()
