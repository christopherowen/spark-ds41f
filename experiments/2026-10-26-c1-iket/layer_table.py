#!/usr/bin/env python3
"""The c1 layer's kernels at one rank's shapes: cost, phases and what the waits are for.

usage: layer_table.py RUN_DIR --tp 4|3 [--shares step_shares.json] [--json OUT]

RUN_DIR is one rank's output of the ``layer`` bundle (``tp4-live3``): ``nsys-graph.sqlite``
(the off build's graph replays), ``table.json`` (IKET's phase table, iket_table.py) and
``sources/`` (the IKET site manifests).

Cost comes from the graph replays. Kernels on the chain launch early under PDL and
wait, so a kernel's duration overstates its cost; each main-chain kernel is charged its
end less the previous main-chain kernel's end (the shared expert's four kernels run on
their own stream, and the combine also waits for them, so the combine's cost includes
any of the shared expert that ends after FC2). A family's cost and duration are summed
over its launches in a layer, then the median over layers is taken. Passes are split at
the LM head and layers at the combine; the first layer of a pass is left out (it
follows the host).

Phases come from IKET's table for the same kernel and grid: each phase's share of a
warp's lifetime, the PDL wait (time spent resident before the predecessor finished)
taken out of the waits and the lifetime. Each wait range is named by what it waits
for, from its statement in the site manifests:
- ``loaded[s].wait`` in the routed GEMMs: the stage's TMA weight tile;
- ``cp_async_wait``: cp.async copies (the dense GEMMs' and the head's weight and
  activation tiles; the routed GEMMs' activation rows and scale codes);
- a ``__syncthreads`` right after a ``cp_async_wait``: warps waiting for the slowest
  warp's copies; other block syncs: the kernel's own steps;
- ``mbarrier`` in the dense GEMMs: TMA tiles.

``--shares`` adds each family's share of the TP4 production step (step_shares.py).

Each weight-streaming family also gets its bytes at the rank's shapes, its floor at the
attainable 266.6 GB/s (#598) and its excess: cost less floor, the part of its time that
is not the byte floor. The routed GEMMs' bytes follow each layer's distinct experts
(the off run's result.json). The shared expert runs beside the routed GEMMs, so its
bytes are counted with the MoE block (dispatch to combine) as a whole. The dense GEMMs
here read cold weights; production prefetches them into L2.
"""
import argparse
import glob
import json
import os
import sqlite3
import statistics
from collections import defaultdict

FAMILY_SHARES = {  # this table's family -> step_shares.py family (TP4 production)
    "routed FC1": "routed FC1", "routed FC2": "routed FC2",
    "Q-B GEMM": "Q-B GEMM", "Q-A/KV GEMM": "Q-A/KV GEMM", "WO-B GEMM": "WO-B and shared down GEMMs",
    "WO-A partials and reduce-cast": "WO-A (partials, reduce-cast)", "sparse MLA": "sparse MLA",
    "mHC project and finalize": "mHC (project, finalize, post)", "LM head": "LM head",
    "routed dispatch": "routed dispatch, SwiGLU, combine", "routed SwiGLU and cast": "routed dispatch, SwiGLU, combine",
    "routed combine": "routed dispatch, SwiGLU, combine", "q norm and Q-B cast": "casts and norms",
}


RATE = 266.6e3  # bytes a microsecond
HIDDEN = 5120
RANK = {4: dict(heads=16, groups=2, inter=576, vocab=32320), 3: dict(heads=24, groups=3, inter=768, vocab=43136)}
MOE_BLOCK = ("routed dispatch", "routed FC1", "routed SwiGLU and cast", "routed FC2", "routed combine")


def fp8(n, k):
    return n * k + (n // 32) * (k // 32)


def expert(n, k):
    return n * k // 2 + n * 2 * -(-k // 128) + n


def family_bytes(tp):
    """Bytes a layer of each weight-streaming family (the routed GEMMs per expert)."""
    s = RANK[tp]
    groups = s["groups"] * 1024
    return {
        "Q-A/KV GEMM": fp8(1792, HIDDEN), "Q-B GEMM": fp8(s["heads"] * 512, 1280),
        "WO-A partials and reduce-cast": fp8(groups, 4096), "WO-B GEMM": fp8(HIDDEN, groups),
        "mHC project and finalize": 2 * 4 * 24 * 4 * HIDDEN, "LM head": s["vocab"] * HIDDEN * 3 // 2,
        "shared": fp8(2 * s["inter"], HIDDEN) + fp8(HIDDEN, s["inter"]),
        "routed FC1": expert(2 * s["inter"], HIDDEN), "routed FC2": expert(HIDDEN, s["inter"]),
    }


def passes(db):
    rows = db.execute(
        "SELECT k.start, k.end, k.gridX, k.gridY, k.gridZ, k.blockX, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()
    heads = [i for i, r in enumerate(rows) if r[6].startswith("vocab_packed")]
    out = [rows[a + 1:b + 1] for a, b in zip(heads, heads[1:])]
    return out


def classify(layer):
    """(family, row, aux) for each launch of one layer, in start order."""
    names = [r[6] for r in layer]
    finals = [i for i, n in enumerate(names) if n.startswith("mhc_finalize")]
    moe_start = finals[1] + 1 if len(finals) > 1 else len(layer)
    act = next((i for i, n in enumerate(names) if n.startswith("activation_swiglu")), None)
    out = []
    gemms = 0
    for i, r in enumerate(layer):
        n, grid = r[6], (r[2], r[3], r[4])
        aux = False
        if n.startswith("mhc_"):
            fam = "mHC project and finalize"
        elif n.startswith("gemm_mxfp8_kernel") and i < moe_start:
            fam = ("Q-A/KV GEMM", "Q-B GEMM", "WO-B GEMM")[min(gemms, 2)]
            gemms += 1
        elif n.startswith(("gemm_fp8w_bf16_partials", "gemm_splitk_reduce_cast")):
            fam = "WO-A partials and reduce-cast"
        elif n.startswith("sparse_mla"):
            fam = "sparse MLA"
        elif n.startswith(("norm_forward", "per_token_cast")) and i < moe_start:
            fam = "q norm and Q-B cast"
        elif n.startswith("moe_dispatch"):
            fam = "routed dispatch"
        elif n.startswith("moe_grouped"):
            fam = "routed FC1" if not any(x[0] == "routed FC1" for x in out) else "routed FC2"
        elif n.startswith("moe_swiglu"):
            fam = "routed SwiGLU and cast"
        elif n.startswith("moe_combine"):
            fam = "routed combine"
        elif n.startswith("vocab_packed"):
            fam = "LM head"
        else:  # the shared expert, on its own stream
            aux = True
            if n.startswith("gemm_mxfp8_kernel"):
                fam = "shared gate/up GEMM" if act is None or i < act else "shared down GEMM"
            else:
                fam = "shared SwiGLU and cast"
        out.append((fam, r, aux))
    return out


def costs(all_passes, distinct, tp):
    """Per family, over layers: cost (main-chain increments), duration, and excess over
    the byte floor; the MoE block's cost and excess; each layer's span."""
    per = defaultdict(list)
    dur = defaultdict(list)
    excess = defaultdict(list)
    keys = {}
    layer_us = []
    nbytes = family_bytes(tp)
    for p in all_passes:
        layers, cur = [], []
        for r in p:
            cur.append(r)
            if r[6].startswith("moe_combine"):
                layers.append(cur)
                cur = []
        if cur and cur[-1][6].startswith("vocab_packed") and layers:
            head = cur[-1]
            dur["LM head"].append((head[1] - head[0]) / 1000)
            per["LM head"].append((head[1] - max(r[1] for r in layers[-1])) / 1000)
            excess["LM head"].append(per["LM head"][-1] - nbytes["LM head"] / RATE)
            keys.setdefault("LM head", set()).add((head[6], (head[2], head[3], head[4]), head[5]))
        for index, layer in enumerate(layers):
            if index == 0:
                continue
            # The layer's first kernel follows the previous layer's combine.
            layer_start = max(r[1] for r in layers[index - 1])
            prev_end = layer_start
            aux_end = 0
            cost, length = defaultdict(float), defaultdict(float)
            for fam, r, aux in classify(layer):
                length[fam] += (r[1] - r[0]) / 1000
                keys.setdefault(fam, set()).add((r[6], (r[2], r[3], r[4]), r[5]))
                if aux:
                    aux_end = max(aux_end, r[1])
                    continue
                end = max(r[1], aux_end) if fam == "routed combine" else r[1]
                if prev_end is not None:
                    cost[fam] += (end - prev_end) / 1000
                prev_end = end
            for fam, us in length.items():
                dur[fam].append(us)
            for fam, us in cost.items():
                per[fam].append(us)
            experts = distinct[index % len(distinct)]
            floors = {fam: b / RATE for fam, b in nbytes.items() if fam in cost}
            for fam in ("routed FC1", "routed FC2"):
                floors[fam] = experts * nbytes[fam] / RATE
            for fam, floor in floors.items():
                excess[fam].append(cost[fam] - floor)
            block = sum(cost[f] for f in MOE_BLOCK if f in cost)
            per["MoE block"].append(block)
            excess["MoE block"].append(block - floors["routed FC1"] - floors["routed FC2"] - nbytes["shared"] / RATE)
            layer_us.append((max(r[1] for r in layer) - layer_start) / 1000)
    return per, dur, excess, keys, layer_us


def wait_kinds(sources):
    """(kernel, range) -> what it waits for."""
    kinds = {}
    for path in glob.glob(os.path.join(sources, "*.json")):
        m = json.load(open(path))
        prev = None
        for s in sorted((s for s in m["sites"] if "range" in s), key=lambda s: s.get("first_line", 0)):
            st = s["statement"]
            if s["phase"] == "wait":
                if "loaded[" in st:
                    what = "TMA weight tile"
                elif "mbarrier[" in st:
                    what = "TMA tile"
                elif "cp_async_wait" in st:
                    what = "cp.async copies"
                elif "GridDependency" in st:
                    what = "PDL predecessor"
                elif "__syncthreads" in st:
                    what = "sync after cp.async (slowest warp's copies)" if prev and "cp_async_wait" in prev else "block sync"
                else:
                    what = "other"
                kinds[(m["kernel"], s["range"])] = what
            prev = st
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--shares")
    ap.add_argument("--json")
    args = ap.parse_args()
    db = sqlite3.connect(os.path.join(args.run, "nsys-graph.sqlite"))
    result_path = next(p for p in (os.path.join(args.run, "off-result.json"), os.path.join(args.run, "off", "result.json"))
                       if os.path.exists(p))
    off = json.load(open(result_path))
    per, dur, excess, keys, layer_us = costs(passes(db), off["distinct_experts"], args.tp)
    keys.setdefault("MoE block", set())
    dur.setdefault("MoE block", per["MoE block"])
    table = json.load(open(os.path.join(args.run, "table.json")))["groups"]
    by_key = {(g["kernel"], tuple(g["grid"]), g["block"][0]): g for g in table}
    kinds = wait_kinds(os.path.join(args.run, "sources"))
    shares = {}
    if args.shares:
        fams = json.load(open(args.shares))["greedy"]["families"]
        shares = {f["family"]: f["share"] for f in fams}
    rows = []
    for fam in sorted(dur, key=lambda f: -(statistics.median(per[f]) if per.get(f) else statistics.median(dur[f]))):
        phases = defaultdict(float)
        waits = defaultdict(float)
        life = 0.0
        for kernel, grid, block in keys[fam]:
            g = by_key.get((kernel, grid, block))
            if g is None:
                continue
            pdl = sum(r["us"] for n, r in g["ranges"].items() if kinds.get((kernel, n)) == "PDL predecessor")
            life += g["warp_us"] - pdl
            for n, r in g["ranges"].items():
                kind = kinds.get((kernel, n))
                if kind == "PDL predecessor":
                    continue
                phases[n.split("#")[0].split(".")[0]] += r["us"]
                if n.startswith("wait"):
                    waits[kind or "other"] += r["us"]
            phases["unattributed"] += g["phases"].get("unattributed", 0.0)
        split = {k: round(v / life, 3) for k, v in phases.items()} if life else {}
        rows.append({
            "family": fam,
            "cost_us": round(statistics.median(per[fam]), 2) if per.get(fam) else None,
            "duration_us": round(statistics.median(dur[fam]), 2),
            "excess_us": round(statistics.median(excess[fam]), 2) if excess.get(fam) else None,
            "phases": split, "waits": {k: round(v / life, 3) for k, v in sorted(waits.items(), key=lambda kv: -kv[1])} if life else {},
            "production_share_tp4": shares.get(FAMILY_SHARES.get(fam)),
        })
    result = {"tp": args.tp, "layer_us": round(statistics.median(layer_us), 1),
              "distinct_mean": off["distinct_mean"], "families": rows}
    print(f"TP{args.tp}: a layer {result['layer_us']} us (median, graph replays), "
          f"{off['distinct_mean']} distinct experts a layer\n")
    print("| kernel | TP4 production share | cost µs a layer | excess over bytes µs | duration µs | wait | load "
          "| compute | store | unattributed | the waits are for |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        p = r["phases"]
        w = "; ".join(f"{k} {v:.0%}" for k, v in list(r["waits"].items())[:3])
        share = f"{r['production_share_tp4']:.1%}" if r["production_share_tp4"] is not None else "-"
        cost = r["cost_us"] if r["cost_us"] is not None else "(aux)"
        ex = r["excess_us"] if r["excess_us"] is not None else "-"
        print(f"| {r['family']} | {share} | {cost} | {ex} | {r['duration_us']} | " + " | ".join(
            f"{p.get(k, 0):.0%}" if p else "-" for k in ("wait", "load", "compute", "store", "unattributed"))
              + f" | {w} |")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
