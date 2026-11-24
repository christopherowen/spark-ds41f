#!/usr/bin/env python3
"""Small throughput bracket: C1, C5, C16, C32 greedy streams, 256 tokens each, 3 passes.
usage: quick_bench.py ROOT OUT.json"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from bi_probe import MODEL, PROMPTS
root, out = sys.argv[1], sys.argv[2]
def one(p):
    body = {"model": MODEL, "prompt": p, "max_tokens": 256, "temperature": 0, "seed": 1}
    req = urllib.request.Request(root + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)["usage"]["completion_tokens"]
res = {}
one(PROMPTS[0])
for c in (1, 5, 16, 32):
    runs = []
    for rep in range(3):
        ps = [PROMPTS[(rep * 7 + i) % len(PROMPTS)] for i in range(c)]
        t = time.time()
        with ThreadPoolExecutor(c) as ex: toks = sum(ex.map(one, ps))
        runs.append(round(toks / (time.time() - t), 1))
    res[f"c{c}"] = runs; print(f"C{c}: {runs} tok/s aggregate", flush=True)
json.dump(res, open(out, "w"))
