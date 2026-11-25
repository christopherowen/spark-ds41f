#!/usr/bin/env python3
"""From a rank-0 step trace: for all-prefill steps, do later rows of each request match the same
token prefix computed alone? Grouped by the CUDA graph's row count. Prints the predicted set of
row counts where the last TP rank's index slice is empty.
usage: graph_verdict.py TRACE.jsonl TP"""
import collections, json, sys
S = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]; tp = int(sys.argv[2])
ref = {}
for s in S:
    if s["num_reqs"] == 1 and s["prefilling"][0] and len(s["chk"]) == s["n"]:
        ids = tuple(s["input_ids"])
        for r in range(s["n"]): ref.setdefault(ids[: r + 1], (s["chk"][r], s["chk2"][r]))
by = collections.defaultdict(lambda: [0, 0, set()])
for s in S:
    if s["num_reqs"] < 2 or not all(s["prefilling"]) or len(s["chk"]) != s["n"]: continue
    q = s["qsl"]; ok = bad = 0
    for j in range(s["num_reqs"]):
        ids = tuple(s["input_ids"][q[j]: q[j + 1]])
        for r in range(1, len(ids)):
            k = ids[: r + 1]
            if k in ref:
                if ref[k] == (s["chk"][q[j] + r], s["chk2"][q[j] + r]): ok += 1
                else: bad += 1
    key = (s["cg_mode"].split(".")[-1], s["bd_tokens"]); by[key][0] += ok; by[key][1] += bad; by[key][2].add(s["n"])
pred = [t for t in range(24, 209) if (tp - 1) * (-(-t // tp)) >= t]
badg, cleang, tot_ok, tot_bad, mixed = [], [], 0, 0, []
for k in sorted(by, key=lambda k: (k[0], k[1])):
    ok, bad, ns = by[k]
    if ok + bad == 0 or k[0] != "FULL": continue
    tot_ok += ok; tot_bad += bad
    (badg if bad else cleang).append(k[1])
    if bad and ok: mixed.append((k[1], ok, bad))
    if bad or k[1] in pred: print(f"  graph {k[1]:4d}: ok {ok:5d} bad {bad:5d}  actual rows {sorted(ns)}")
print(f"TP{tp}: rows checked on captured graphs {tot_ok + tot_bad} (ok {tot_ok}, bad {tot_bad})")
print("captured graph sizes with bad later rows:", badg, "| mixed ok+bad:", mixed)
print("captured graph sizes all clean:", cleang)
print(f"predicted row counts with an empty last-rank slice (rows >= 24): {pred}")
print("bad set is a subset of predicted:", set(badg) <= set(pred), "| predicted sizes that were captured and came out clean:", sorted(set(pred) & set(cleang)))
