#!/usr/bin/env python3
"""analyze.py RAW: the #995 guard's identity, bench cells and tests from raw/ (outputs as run).
Cell rule (candidate-53 own marks, analyze/common.py): two boots an arm, ABBA in one session;
delta = mean(cand) - mean(base); spread = the larger boot-to-boot change within an arm;
'worse' only when every candidate boot is above every base boot AND |delta| > spread
('better' mirrored), else 'same'. Writes RAW/analysis.json."""
import json, os, re, statistics, sys

d = sys.argv[1]
BOOTS = ("base-boot1", "cand-boot1", "cand-boot2", "base-boot2")


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]


def cell(b, c):
    mb, mc = statistics.mean(b), statistics.mean(c)
    dd = mc - mb
    spread = max(abs(b[0] - b[1]), abs(c[0] - c[1]))
    better = max(c) < min(b) and abs(dd) > spread
    worse = min(c) > max(b) and abs(dd) > spread
    return {"base": b, "cand": c, "delta": round(dd, 3), "delta_pct": round(100 * dd / mb, 2) if mb else None,
            "boot_spread": round(spread, 3), "verdict": "better" if better else "worse" if worse else "same"}


def tally(cells):
    v = [c["verdict"] for c in cells.values()]
    return {"cells": len(v), "better": v.count("better"), "same": v.count("same"), "worse": v.count("worse"),
            "delta_pct_range": [min(c["delta_pct"] for c in cells.values()), max(c["delta_pct"] for c in cells.values())]}


out = {}
# 1. identity (guard995.py section 1, ids995b.py)
for name, prefix in (("identity", "bench"), ("identity_refusals", "ids2")):
    runs = {b: jl(f"{d}/{prefix}-{b}.jsonl") for b in BOOTS}
    cells = {b: {(r["scenario"], r["reqs"], r["ctx"]): r for r in rs if r["kind"] == "identity"} for b, rs in runs.items()}
    allsha = {b: [r["sha"] for r in rs if r["kind"] == "identity_all"] for b, rs in runs.items()}
    keys = sorted(cells["base-boot1"])
    differ = [f"{k[0]}/c{k[1]}/{k[2]}" for k in keys
              if len({cells[b][k]["cell_sha"] for b in BOOTS}) != 1]
    steps = [s for k in keys for s in cells["base-boot1"][k]["steps"]]
    cov = {"steps": len(steps), "offered_rows": sum(s["offered_rows"] for s in steps),
           "copy_rows": sum(s["copy_rows"] for s in steps),
           "refused_rows": sum(s.get("refused_rows", 0) for s in steps) if "refused_rows" in steps[0] else None,
           "steps_with_doublings_ge_2": sum(s["max_doublings"] >= 2 for s in steps)}
    env = {b: [r for r in rs if r["kind"] == "identity_env"][0] for b, rs in runs.items()}
    out[name] = {"cells": len(keys), "steps_hashed_per_boot": len(steps), "all_cells_sha": allsha,
                 "cells_differing_between_any_boots": differ, "bit_exact": not differ and len({x[0] for x in allsha.values()}) == 1,
                 "coverage_base_boot1": cov,
                 "tilelang_copy_sha256": {b: env[b]["src_sha256"][:12] for b in BOOTS},
                 "tilelang_copy_path": env["cand-boot1"]["tilelang_copy"]}

# 2. timing (guard995.py section 2)
runs = {b: {(r["history"], r["reqs"], r["ctx"]): r for r in jl(f"{d}/bench-{b}.jsonl") if r["kind"] == "timing"} for b in BOOTS}
timing = {}
for metric in ("step_us", "alloc_apply_us", "eager_alloc_apply_us"):
    cells = {}
    for k in sorted(runs["base-boot1"]):
        cells[f"{k[0]}/c{k[1]}/{k[2]}"] = cell([runs["base-boot1"][k][metric], runs["base-boot2"][k][metric]],
                                                [runs["cand-boot1"][k][metric], runs["cand-boot2"][k][metric]])
    timing[metric] = {"summary": tally(cells), "cells": cells}
out["timing"] = timing
out["timing_found_all"] = all(r["found"] == r["reqs"] for b in BOOTS for r in runs[b].values())

# 3. gate: benchmark_copy_drafts.py --gate --cases copy (unmodified)
gates = {}
for b in BOOTS:
    rows = jl(f"{d}/gate-{b}.txt")
    gates[b] = {"summary": [r for r in rows if "summary" in r],
                "cases": {f'{r["history"]}/c{r["requests"]}/{r["context"]}': r["us"] for r in rows if r.get("case") == "copy"},
                "over_budget": [r for r in rows if r.get("case") == "copy" and r["slower"]]}
gcells = {k: cell([gates["base-boot1"]["cases"][k], gates["base-boot2"]["cases"][k]],
                  [gates["cand-boot1"]["cases"][k], gates["cand-boot2"]["cases"][k]]) for k in gates["base-boot1"]["cases"]}
out["gate"] = {"summary": tally(gcells), "budget_verdicts": {b: g["summary"] for b, g in gates.items()}, "cells": gcells}

# 3b. confirmation of the gate's copy cells at 201 replays (confirm995.py): three ABBA sets
# (boots 1-2, 3-4, 5-6 an arm), each set's verdicts, and the six boots an arm pooled: 'worse'
# only when every candidate boot is above every base boot and |delta| exceeds the larger
# within-arm range; a Welch 95% interval on the boot medians beside it.
import math
def pooled(b, c):
    mb, mc = statistics.mean(b), statistics.mean(c)
    dd = mc - mb
    spread = max(max(b) - min(b), max(c) - min(c))
    se = math.sqrt(statistics.variance(b) / len(b) + statistics.variance(c) / len(c))
    v = "better" if max(c) < min(b) and abs(dd) > spread else "worse" if min(c) > max(b) and abs(dd) > spread else "same"
    return {"base": b, "cand": c, "delta": round(dd, 3), "delta_pct": round(100 * dd / mb, 2),
            "range_spread": round(spread, 3), "welch95": [round(dd - 1.96 * se, 3), round(dd + 1.96 * se, 3)], "verdict": v}
def load_confirm(n, arm):
    return {f'{r["history"]}/c{r["requests"]}/{r["context"]}': r["us"] for r in jl(f"{d}/confirm-{arm}-boot{n}.txt") if r.get("case") == "copy"}
conf = {f"{a}{n}": load_confirm(n, a) for a in ("base", "cand") for n in range(1, 7) if os.path.exists(f"{d}/confirm-{a}-boot{n}.txt")}
sets = {}
for s_, (x, y) in enumerate(((1, 2), (3, 4), (5, 6)), 1):
    if f"base{y}" in conf and f"cand{y}" in conf:
        cells = {k: cell([conf[f"base{x}"][k], conf[f"base{y}"][k]], [conf[f"cand{x}"][k], conf[f"cand{y}"][k]]) for k in conf["base1"]}
        sets[f"set{s_}_boots{x}-{y}"] = {"summary": tally(cells), "flagged": {k: v for k, v in cells.items() if v["verdict"] != "same"}}
nb = sorted(int(k[4:]) for k in conf if k.startswith("base"))
pool = {k: pooled([conf[f"base{n}"][k] for n in nb], [conf[f"cand{n}"][k] for n in nb]) for k in conf["base1"]}
out["confirm201"] = {"sets": sets, "pooled_boots_per_arm": len(nb), "pooled_summary": tally(pool),
                     "budget_verdicts": {f"{a}-boot{n}": [r for r in jl(f"{d}/confirm-{a}-boot{n}.txt") if "summary" in r]
                                         for a in ("base", "cand") for n in nb},
                     "pooled_cells": pool}
# 3c. the guarded kernels with launch latency amortized (k32995.py), boots 3-6 an arm
k32 = {f"{a}{n}": {f'{r["history"]}/c{r["reqs"]}/{r["ctx"]}': r["pair_us"] for r in jl(f"{d}/k32-{a}-boot{n}.jsonl")}
       for a in ("base", "cand") for n in range(3, 7) if os.path.exists(f"{d}/k32-{a}-boot{n}.jsonl")}
if k32:
    kb = sorted(int(k[4:]) for k in k32 if k.startswith("base"))
    ksets = {}
    for x, y in ((3, 4), (5, 6)):
        if f"base{y}" in k32 and f"cand{y}" in k32:
            cells = {k: cell([k32[f"base{x}"][k], k32[f"base{y}"][k]], [k32[f"cand{x}"][k], k32[f"cand{y}"][k]]) for k in k32[f"base{x}"]}
            ksets[f"boots{x}-{y}"] = {"summary": tally(cells), "flagged": {k: v for k, v in cells.items() if v["verdict"] != "same"}}
    kpool = {k: pooled([k32[f"base{n}"][k] for n in kb], [k32[f"cand{n}"][k] for n in kb]) for k in k32[f"base{kb[0]}"]}
    out["k32_pair_us"] = {"sets": ksets, "pooled_boots_per_arm": len(kb), "pooled_summary": tally(kpool), "pooled_cells": kpool}

# 4. tests
tests = {}
for name in ("pytest-cand-boot1", "pytest-cand-boot2", "pytest-base-boot1", "canary-base-boot1"):
    p = f"{d}/{name}.txt"
    if not os.path.exists(p):
        continue
    txt = open(p).read()
    m = re.findall(r"=+ (.*?) in [0-9.]+s", txt)
    canary = re.findall(r"(\S*test_negative_slot_bounds_guard_in_alloc_and_apply\S*) (PASSED|FAILED|ERROR)", txt)
    fails = re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, re.M)
    assert_lines = [l.strip() for l in txt.splitlines() if l.startswith("E ")][:12]
    tests[name] = {"result": m[-1] if m else None, "canary": canary[:1], "failed": fails, "first_error_lines": assert_lines}
out["tests"] = tests
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("cells", "pooled_cells")} if isinstance(v, dict) else v
                  for k, v in out.items() if k not in ("timing", "identity", "identity_refusals")}, indent=1)[:9000])
for metric, t in timing.items():
    print(metric, json.dumps(t["summary"]))
