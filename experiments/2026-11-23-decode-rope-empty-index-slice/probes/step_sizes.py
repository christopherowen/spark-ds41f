#!/usr/bin/env python3
"""Step-size sweep: which prefill step sizes change a request's first token?

For each total N, ONE API request carries K short token-id prompts (each <= --maxlen tokens)
whose lengths sum to N, so they enter the engine together and share a prefill step.
max_tokens=1. Every prompt's top-5 first-token logprobs are compared with the same prompt
sent alone. Prompts are distinct slices of a long text, so nothing is prefix-cached.

usage: step_sizes.py ROOT OUT.json [--totals 18-208] [--maxlen 16] [--reps 2]
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_probe import MODEL, TOPICS  # noqa: E402


def post(root, path, body):
    req = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def ask(root, prompts):
    d = post(root, "/v1/completions", {"model": MODEL, "prompt": prompts, "max_tokens": 1, "temperature": 0, "seed": 1,
                                       "logprobs": 5, "return_tokens_as_token_ids": True})
    out = [None] * len(prompts)
    for c in d["choices"]:
        out[c["index"]] = (c["logprobs"]["tokens"][0], c["logprobs"]["top_logprobs"][0])
    return out, d["usage"]["prompt_tokens"]


def parse_totals(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--totals", default="18-208")
    ap.add_argument("--maxlen", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    # 32 distinct base texts -> token ids; prompt j of length L is base[j][:L]
    base = []
    for j in range(32):
        t = f"Note {j + 1}. " + " ".join(f"{TOPICS[(j + i) % len(TOPICS)]}," for i in range(6))
        ids = post(a.root, "/tokenize", {"model": MODEL, "prompt": t, "add_special_tokens": False})["tokens"]
        assert len(ids) >= a.maxlen, (j, len(ids))
        base.append(ids)
    solo = {}

    def solo_of(j, L):
        if (j, L) not in solo:
            got, n = ask(a.root, [base[j][:L]])
            assert n == L
            solo[(j, L)] = got[0]
        return solo[(j, L)]

    rows = []
    for N in parse_totals(a.totals):
        K = -(-N // a.maxlen)
        if K > 32 or N < 2 * K:
            continue
        lens = [N // K + (1 if i < N % K else 0) for i in range(K)]
        refs = [solo_of(j, L) for j, L in enumerate(lens)]
        res = []
        for _ in range(a.reps):
            got, n = ask(a.root, [base[j][:L] for j, L in enumerate(lens)])
            assert n == N, (n, N)
            res.append(sum(g != r for g, r in zip(got, refs)))
        rows.append({"total_rows": N, "prompts": K, "lens": lens, "differing_prompts": res})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"root": a.root, "args": vars(a), "rows": rows}, open(a.out, "w"))
    bad = [r["total_rows"] for r in rows if any(r["differing_prompts"])]
    runs, start, prev = [], None, None
    for n in bad:
        if prev is None or n != prev + 1:
            if start is not None:
                runs.append((start, prev))
            start = n
        prev = n
    if start is not None:
        runs.append((start, prev))
    print("totals with any differing prompt:", " ".join(f"{a_}..{b_}" if a_ != b_ else str(a_) for a_, b_ in runs) or "none")


if __name__ == "__main__":
    main()
