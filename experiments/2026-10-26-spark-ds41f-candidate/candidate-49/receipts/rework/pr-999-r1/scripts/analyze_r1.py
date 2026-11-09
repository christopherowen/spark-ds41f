#!/usr/bin/env python3
"""Score PR #999 rework r1 (corrected 0762 on candidate-49) against 1107.

Interval rule (own-marks): two boots per arm; a gain is beyond the interval when the
slower candidate boot is faster than the faster baseline boot, a loss when the faster
candidate boot is slower than the slower baseline boot.

usage: analyze_r1.py <receipts own-marks json dir> <r1 json dir> <out.json>

Sections
- pr_bench_unwarmed: the PR's bench_2boot.py as is. Boots are scored when (a) no other
  GPU container ran during them (quiet.sh / quiet2.sh) and (b) their in-process 1107
  arm shows no gross perturbation: no 1107 cell above 2x its receipt minimum (the
  GB10 SM clock idles near 200 MHz, and a section timed right after kernel
  construction starts at a low clock). Rule (b) reads only the 1107 arm.
- pr_bench_warmed / served_gaps / short_sweep: the same benches through warm_wrap.py
  (>= 100 ms of matmul before every timed arm), two boots each.
- corrected vs the receipts' 1107 boots (not re-measured), and vs the in-process
  1107 arm of the same boots; the gap bench and sweep have no receipts, so only the
  in-process 1107 arm.
"""
import json
import sys
from pathlib import Path

own, r1, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
load = lambda p: json.loads(p.read_text())  # noqa: E731

PAIRS = [
    ("dense_524k_k512", "dense_524k_k512_base_us", "dense_524k_k512_cand_us"),
    ("src_1m_select_blocks_k2048_g8", "src_1m_blk_k2048_g8_base_us", "src_1m_blk_k2048_g8_cand_us"),
    ("src_1m_select_k512", "src_1m_sel_k512_base_us", "src_1m_sel_k512_cand_us"),
    ("src_1m_total", "src_1m_total_base_us", "src_1m_total_cand_us"),
    ("raw_1m_k512_g1", "raw_1m_k512_g1_base_us", "raw_1m_k512_g1_cand_us"),
    ("raw_1m_k2048_g1", "raw_1m_k2048_g1_base_us", "raw_1m_k2048_g1_cand_us"),
]


def cell(b, c):
    mb, mc = sum(b) / len(b), sum(c) / len(c)
    return {
        "base_boots": b,
        "cand_boots": c,
        "speedup_of_means": round(mb / mc, 3),
        "gain_beyond_interval": max(c) < min(b),
        "worse_beyond_interval": min(c) > max(b),
    }


def tally(cells):
    return {
        "cells": len(cells),
        "gain_beyond_interval": sum(c["gain_beyond_interval"] for c in cells.values()),
        "worse_beyond_interval": sum(c["worse_beyond_interval"] for c in cells.values()),
        "worse_cells": sorted(k for k, c in cells.items() if c["worse_beyond_interval"]),
        "neither_cells": sorted(
            k for k, c in cells.items() if not c["gain_beyond_interval"] and not c["worse_beyond_interval"]
        ),
    }


def pr_keys(x):
    for M in x["reindex_16k"]:
        yield f"reindex_16k M={M}", ("reindex_16k", M, "base_us"), ("reindex_16k", M, "cand_us")
    for N, rows in x["grid"].items():
        for M in rows:
            for name, bk, ck in PAIRS:
                if bk in rows[M]:
                    yield f"{name} N={N} M={M}", ("grid", N, M, bk), ("grid", N, M, ck)


def get(x, path):
    for p in path:
        x = x[p]
    return x


rec = [load(own / f"bench-candidate49-boot{i}.json") for i in (1, 2)]


def pr_section(boot_files):
    cor = [load(r1 / f) for f in boot_files]
    vs_rec, vs_inrun, vs_c49 = {}, {}, {}
    for k, bp, cp in pr_keys(rec[0]):
        rb = [get(x, bp) for x in rec]
        rc = [get(x, cp) for x in rec]
        ib = [get(x, bp) for x in cor]
        cc = [get(x, cp) for x in cor]
        vs_rec[k], vs_inrun[k], vs_c49[k] = cell(rb, cc), cell(ib, cc), cell(rc, cc)
    return {
        "boots": boot_files,
        "exactness": [x.get("exactness") for x in cor],
        "vs_receipts_1107": {"summary": tally(vs_rec), "cells": vs_rec},
        "vs_inrun_1107": {"summary": tally(vs_inrun), "cells": vs_inrun},
        "corrected_vs_candidate49_receipts": {
            "summary": {k: v for k, v in tally(vs_c49).items() if k != "neither_cells"},
            "speedup_of_means_range": [min(c["speedup_of_means"] for c in vs_c49.values()),
                                       max(c["speedup_of_means"] for c in vs_c49.values())],
        },
        "slow_cell_raw_1m_k512_g1_N=524288_M=1": vs_rec["raw_1m_k512_g1 N=524288 M=1"],
        "headline": {k: vs_rec[k] for k in (
            "reindex_16k M=1", "reindex_16k M=1024", "src_1m_total N=1048576 M=16",
            "src_1m_total N=1048576 M=256", "src_1m_total N=1048576 M=1024",
            "dense_524k_k512 N=524288 M=256", "dense_524k_k512 N=524288 M=1024",
            "raw_1m_k2048_g1 N=1048576 M=256")},
    }


def control(boot_file):
    """Rule (b): worst ratio of an in-process 1107 cell to its receipt minimum."""
    x = load(r1 / boot_file)
    worst = max((get(x, bp) / min(get(r, bp) for r in rec), k) for k, bp, _ in pr_keys(rec[0]))
    return {"worst_1107_over_receipt_min": round(worst[0], 3), "cell": worst[1], "ok": worst[0] <= 2.0}


def generic(files, arms):
    d = [load(r1 / f) for f in files]
    res = {a: {} for a in arms}
    exact = True
    for sel, cells in d[0]["cells"].items():
        for key, c0 in cells.items():
            exact &= all(x["cells"][sel][key]["exact"] for x in d)
            if "blk_base_us" in c0:
                parts = [(".blk", "blk_base_us", "blk_cand_us"), (".sel", "sel_base_us", "sel_cand_us"),
                         (".total", "total_base_us", "total_cand_us")]
                for tag, bk, ck in parts:
                    res["cand"][f"{sel}{tag} {key}"] = cell([x["cells"][sel][key][bk] for x in d],
                                                           [x["cells"][sel][key][ck] for x in d])
                continue
            for a in arms:
                res[a][f"{sel} {key}"] = cell([x["cells"][sel][key]["base_us"] for x in d],
                                              [x["cells"][sel][key][f"{a}_us"] for x in d])
    return {"boots": files, "all_exact": bool(exact),
            **{f"{a}_vs_1107": {"summary": tally(res[a]), "cells": res[a]} for a in arms}}


res = {"interval_rule": "gain: max(cand boots) < min(1107 boots); worse: min(cand boots) > max(1107 boots)"}
unwarmed = [f"r1-bench2boot-boot{i}.json" for i in (1, 2, 3, 4, 5) if (r1 / f"r1-bench2boot-boot{i}.json").exists()]
res["pr_bench_boot_controls"] = {f: control(f) for f in unwarmed}
contaminated = {"r1-bench2boot-boot1.json"}  # a foreign spark-ds41f:1107 container ran during it
clean = [f for f in unwarmed if f not in contaminated]
# Every container-clean boot at once: a gain needs every corrected boot below the
# faster receipt boot, a loss every corrected boot above the slower one.
res["pr_bench_unwarmed"] = pr_section(clean)
res["pr_bench_warmed"] = pr_section(["r1w-bench2boot-boot1.json", "r1w-bench2boot-boot2.json"])
res["served_gaps"] = generic(["r1w-served-gaps-boot1.json", "r1w-served-gaps-boot2.json"], ["cand"])
res["short_sweep"] = generic(["r1w-short-sweep-boot1.json", "r1w-short-sweep-boot2.json"], ["cand", "c49"])
res["pair_c49"] = generic(["r1w-pair-c49-boot1.json", "r1w-pair-c49-boot2.json"], ["cand", "c49"])
_pd = [load(r1 / f) for f in res["pair_c49"]["boots"]]
_pc = {}
for sel, cells in _pd[0]["cells"].items():
    for key in cells:
        _pc[f"{sel} {key}"] = cell([x["cells"][sel][key]["c49_us"] for x in _pd],
                                   [x["cells"][sel][key]["cand_us"] for x in _pd])
res["pair_c49"]["corrected_vs_candidate49_same_process"] = {"summary": tally(_pc), "cells": _pc}
res["supplementary_unwarmed"] = {
    "served_gaps": generic(["r1-served-gaps-boot2.json", "r1-served-gaps-boot3.json"], ["cand"]),
    "short_sweep": generic(["r1-short-sweep-boot1.json", "r1-short-sweep-boot2.json"], ["cand", "c49"]),
}
out.write_text(json.dumps(res, indent=1) + "\n")


def brief(sec):
    return {k: v["summary"] for k, v in sec.items() if isinstance(v, dict) and "summary" in v}


print(json.dumps({
    "controls": res["pr_bench_boot_controls"],
    "pr_bench_unwarmed": {"boots": res["pr_bench_unwarmed"].get("boots"),
                          "exactness": res["pr_bench_unwarmed"].get("exactness"),
                          **brief(res["pr_bench_unwarmed"])},
    "pr_bench_warmed": {"exactness": res["pr_bench_warmed"]["exactness"], **brief(res["pr_bench_warmed"])},
    "served_gaps": {"exact": res["served_gaps"]["all_exact"], **brief(res["served_gaps"])},
    "short_sweep": {"exact": res["short_sweep"]["all_exact"], **brief(res["short_sweep"])},
    "pair_c49": {"exact": res["pair_c49"]["all_exact"], **brief(res["pair_c49"])},
}, indent=1))
