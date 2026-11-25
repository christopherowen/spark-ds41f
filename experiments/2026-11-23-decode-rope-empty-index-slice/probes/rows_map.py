#!/usr/bin/env python3
"""Row map: which prefill step sizes change a request's first token?

One request carries two token-id prompts, so both enter the engine together: the probe P
(p tokens) and a filler of f tokens. max_tokens=1, so only the shared prefill step computes
P's logits. P's top-5 first-token logprobs are classed against P sent alone.

usage: rows_map.py ROOT OUT.json [--p 8] [--totals 8-80,96,97,128,129,160,192,200,204,208] [--reps 2]
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bi_probe import MODEL  # noqa: E402

TEXT = ("The aqueducts of ancient Rome carried water across valleys and through hills for hundreds of miles, "
        "and the engineers who built them understood gradients, siphons and settling tanks well enough that "
        "several still stand today. ") * 12
FILL = ("Sourdough bread starts with a culture of wild yeast and bacteria that a baker feeds with flour and "
        "water every day until it rises and falls on a steady schedule in a warm kitchen. ") * 12


def post(root, path, body):
    req = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def tokenize(root, text):
    return post(root, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]


def ask(root, prompts):
    d = post(root, "/v1/completions", {"model": MODEL, "prompt": prompts, "max_tokens": 1, "temperature": 0, "seed": 1,
                                       "logprobs": 5, "return_tokens_as_token_ids": True})
    out = {}
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
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--totals", default="8-80,96,97,128,129,160,192,200,204,208")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--order", default="pf", help="pf: probe first; fp: filler first")
    a = ap.parse_args()
    P = tokenize(a.root, TEXT)[: a.p]
    F = tokenize(a.root, FILL)
    solo, n = ask(a.root, [P])
    solo2, _ = ask(a.root, [P])
    assert n == a.p, (n, a.p)
    classes = [solo[0]]
    print(f"p={a.p} solo prompt_tokens={n} solo repeat identical={solo[0] == solo2[0]} order={a.order}")
    rows = []
    for total in parse_totals(a.totals):
        f = total - a.p
        if f < 0:
            continue
        cls = []
        for _ in range(a.reps):
            if f == 0:
                got, used = ask(a.root, [P])
                mine = got[0]
            elif a.order == "pf":
                got, used = ask(a.root, [P, F[:f]])
                mine = got[0]
            else:
                got, used = ask(a.root, [F[:f], P])
                mine = got[1]
            assert used == total, (used, total)
            if mine not in classes:
                classes.append(mine)
            cls.append(classes.index(mine))
        rows.append({"total_rows": total, "filler": f, "class": cls})
    json.dump({"root": a.root, "args": vars(a), "probe_ids": P, "classes": [{"token": t, "top": top} for t, top in classes], "rows": rows}, open(a.out, "w"))
    # compact run-length print: class per total
    line, prev, start = [], None, None
    for r in rows + [{"total_rows": None, "class": None}]:
        key = None if r["class"] is None else tuple(r["class"])
        if key != prev:
            if prev is not None:
                line.append(f"{start}..{last}:{'/'.join('ABCDEFGHIJ'[c] if c < 10 else '?' for c in prev)}")
            prev, start = key, r["total_rows"]
        last = r["total_rows"]
    print("rows->class (A = same as alone):", "  ".join(line))
    print("classes:", len(classes), [(t, round(max(abs(v - classes[0][1].get(k, -99)) for k, v in top.items()), 4)) for t, top in classes])


if __name__ == "__main__":
    main()
