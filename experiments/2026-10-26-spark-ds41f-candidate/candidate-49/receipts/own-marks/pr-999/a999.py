#!/usr/bin/env python3
"""Summarise PR #999's bench_2boot.py JSONs: candidate-49 boots 1/2 (base = 1107
indexer.py with the _e2m1_code shim, cand = image) and, when present, the 1107 boots."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
out = Path(sys.argv[2])


def load(name):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


c = [load("bench-candidate49-boot1.json"), load("bench-candidate49-boot2.json")]
r = [load("bench-1107-boot1.json"), load("bench-1107-boot2.json")]
PAIRS = [
    ("dense_524k_k512", "dense_524k_k512_base_us", "dense_524k_k512_cand_us"),
    ("src_1m_select_blocks_k2048_g8", "src_1m_blk_k2048_g8_base_us", "src_1m_blk_k2048_g8_cand_us"),
    ("src_1m_select_k512", "src_1m_sel_k512_base_us", "src_1m_sel_k512_cand_us"),
    ("src_1m_total", "src_1m_total_base_us", "src_1m_total_cand_us"),
    ("raw_1m_k512_g1", "raw_1m_k512_g1_base_us", "raw_1m_k512_g1_cand_us"),
    ("raw_1m_k2048_g1", "raw_1m_k2048_g1_base_us", "raw_1m_k2048_g1_cand_us"),
]


def cell(b, cnd):
    lo_b, hi_b = min(b), max(b)
    lo_c, hi_c = min(cnd), max(cnd)
    mb, mc = sum(b) / len(b), sum(cnd) / len(cnd)
    return {
        "base_boots": b,
        "cand_boots": cnd,
        "speedup_of_means": round(mb / mc, 3),
        "gain_beyond_interval": hi_c < lo_b,
        "worse_beyond_interval": lo_c > hi_b,
    }


res = {"exactness": [x.get("exactness") if x else None for x in c], "reindex_16k": {}, "grid": {}}
for M in c[0]["reindex_16k"]:
    b = [x["reindex_16k"][M]["base_us"] for x in c]
    k = [x["reindex_16k"][M]["cand_us"] for x in c]
    res["reindex_16k"][M] = cell(b, k)
n_cells = n_gain = n_worse = 0
speedups = {}
for N, rows in c[0]["grid"].items():
    for M in rows:
        key = f"N={N},M={M}"
        res["grid"][key] = {}
        for name, bk, ck in PAIRS:
            if bk not in rows[M]:
                continue
            b = [x["grid"][N][M][bk] for x in c]
            k = [x["grid"][N][M][ck] for x in c]
            cl = cell(b, k)
            res["grid"][key][name] = cl
            n_cells += 1
            n_gain += cl["gain_beyond_interval"]
            n_worse += cl["worse_beyond_interval"]
            speedups.setdefault(name, []).append(cl["speedup_of_means"])
        # score kernels: cand-only on each image; candidate image vs 1107 image
        for sk in ("score_p128_us", "score_p256_us"):
            if sk in rows[M] and r[0] and r[1]:
                cs = [x["grid"][N][M][sk] for x in c]
                rs = [x["grid"][N][M][sk] for x in r]
                res["grid"][key][sk + "_1107_vs_candidate49"] = cell(rs, cs)
for M, cl in res["reindex_16k"].items():
    n_cells += 1
    n_gain += cl["gain_beyond_interval"]
    n_worse += cl["worse_beyond_interval"]
    speedups.setdefault("reindex_16k", []).append(cl["speedup_of_means"])
res["summary"] = {
    "selection_cells": n_cells,
    "gain_beyond_interval": n_gain,
    "worse_beyond_interval": n_worse,
    "speedup_range_by_selector": {k: [min(v), max(v)] for k, v in speedups.items()},
}
if r[0] and r[1]:
    aa = []
    for N, rows in r[0]["grid"].items():
        for M in rows:
            for name, bk, ck in PAIRS:
                if bk in rows[M]:
                    for x in r:
                        aa.append(x["grid"][N][M][bk] / x["grid"][N][M][ck])
    res["summary"]["aa_1107_base_over_cand_range"] = [round(min(aa), 3), round(max(aa), 3)]
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res["summary"], indent=1))
