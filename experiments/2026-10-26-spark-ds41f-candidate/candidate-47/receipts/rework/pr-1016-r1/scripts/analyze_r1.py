#!/usr/bin/env python3
"""Digest of the PR #1016 rework (r1) ABBA: h1016r1-{no1016,r1}-boot{1,2}.json.

Per pair, M and L2 state (graph replay):
  x-img   image arm on the r1 overlay minus image arm on no1016 (1107's gemm.py),
          mean of the two boots each; ci = CI95 of that difference of means;
          boot range: gain if both r1 boots are below both no1016 boots, slower if above.
  paired  in-process r1 table minus 1107 table (same weights, interleaved windows),
          pooled over the four processes (mean of means; ci from the four CI95s).
Cell outcome: same-kernels when r1 compiles the 1107 kernels at that M; otherwise
gain / slower when the pooled paired delta is beyond the larger of its within-process
CI95 and its across-process CI95 (t, 3 df) and the x-img delta has the same sign;
"gain (x-img boot range ...)" when only the within-process CI and the x-img boot
ranges agree; else within. A/A floor: the largest |x-img delta| over same-kernels
cells of the pair's shape, per state.
usage: analyze_r1.py OUTDIR  (prints the table; writes OUTDIR/cells.json)
"""
import json
import math
import sys

D = sys.argv[1]
B = {f"{a}-{b}": json.load(open(f"{D}/h1016r1-{a}-{b}.json"))
     for a in ("no1016", "r1") for b in ("boot1", "boot2")}
STATES = ("graph_warm", "graph_cold", "graph_next")
T3 = 3.182  # t(0.975, 3 df): four processes
BASE, CAND = ("no1016-boot1", "no1016-boot2"), ("r1-boot1", "r1-boot2")


def arm_stat(run, pair, m, arm, st):
    row = run["pairs"][pair][m]
    return row["variants"][row["arm_variant"][arm]][st]


def paired(run, pair, m, arm, st):
    v = run["pairs"][pair][m]["paired_minus_1107_us"][arm]
    return None if v == "same-kernels" else v[st]


cells, lines = {}, []
for pair in B["r1-boot1"]["pairs"]:
    cells[pair] = {}
    lines.append(f"== {pair}")
    for m in B["r1-boot1"]["pairs"][pair]:
        same = all(B[k]["pairs"][pair][m]["arm_variant"]["r1"] == B[k]["pairs"][pair][m]["arm_variant"]["1107"]
                   for k in B)
        img_ok = (all(B[k]["pairs"][pair][m]["arm_variant"]["image"] == B[k]["pairs"][pair][m]["arm_variant"]["r1"]
                      for k in CAND)
                  and all(B[k]["pairs"][pair][m]["arm_variant"]["image"]
                          == B[k]["pairs"][pair][m]["arm_variant"]["1107"] for k in BASE))
        bits = {k: B[k]["pairs"][pair][m]["bits_sha256_16"]["image"] for k in B}
        eq_all = all(all(B[k]["pairs"][pair][m]["bit_equal_to_1107_arm"].values()) for k in B)
        cells[pair][m] = {"r1_same_kernels_as_1107": same, "image_arm_is_overlay_table": img_ok,
                          "bits_equal_across_images": len(set(bits.values())) == 1,
                          "all_arms_torch_equal_in_process": eq_all, "states": {}}
        for st in STATES:
            b = [arm_stat(B[k], pair, m, "image", st) for k in BASE]
            c = [arm_stat(B[k], pair, m, "image", st) for k in CAND]
            dx = (c[0]["mean"] + c[1]["mean"]) / 2 - (b[0]["mean"] + b[1]["mean"]) / 2
            cix = math.sqrt(sum(x["ci95"] ** 2 for x in b + c)) / 2
            rng = ("gain" if max(x["mean"] for x in c) < min(x["mean"] for x in b) else
                   "slower" if min(x["mean"] for x in c) > max(x["mean"] for x in b) else "within")
            cell = {"no1016_b1_b2": [f"{x['mean']:.2f}±{x['ci95']:.2f}" for x in b],
                    "r1_b1_b2": [f"{x['mean']:.2f}±{x['ci95']:.2f}" for x in c],
                    "x_img_delta_us": round(dx, 2), "x_img_ci95": round(cix, 2), "x_img_boot_range": rng}
            for arm in ("r1", "pr1016", "thr48"):
                ps = [paired(B[k], pair, m, arm, st) for k in B]
                if ps[0] is None:
                    cell[f"paired_{arm}_minus_1107_us"] = "same-kernels"
                    continue
                pm = sum(p["mean"] for p in ps) / 4
                pci = math.sqrt(sum(p["ci95"] ** 2 for p in ps)) / 4
                sd = math.sqrt(sum((p["mean"] - pm) ** 2 for p in ps) / 3)
                cell[f"paired_{arm}_minus_1107_us"] = {
                    "per_boot": [f"{p['mean']:+.2f}±{p['ci95']:.2f}" for p in ps],
                    "pooled": round(pm, 2), "pooled_ci95": round(pci, 2),
                    "process_ci95": round(T3 * sd / 2, 2)}
            pr = cell["paired_r1_minus_1107_us"]
            if same:
                out = "same-kernels"
            else:
                ci = max(pr["pooled_ci95"], pr["process_ci95"])
                if pr["pooled"] < -ci and dx < 0:
                    out = "gain"
                elif pr["pooled"] > ci and dx > 0:
                    out = "slower"
                elif pr["pooled"] < -pr["pooled_ci95"] and rng == "gain":
                    out = "gain (x-img boot range; one process paired off)"
                else:
                    out = "within"
            cell["outcome"] = out
            cells[pair][m]["states"][st] = cell
        s = cells[pair][m]["states"]

        def fmt(st):
            c = s[st]
            p = c["paired_r1_minus_1107_us"]
            ptxt = "=" if p == "same-kernels" else f"{p['pooled']:+.2f}±{p['pooled_ci95']:.2f}/{p['process_ci95']:.2f}"
            q = c["paired_pr1016_minus_1107_us"]
            qtxt = "=" if q == "same-kernels" else f"{q['pooled']:+.2f}"
            return (f"{st[6:]:4} x-img {c['x_img_delta_us']:+6.2f}±{c['x_img_ci95']:.2f} ({c['x_img_boot_range']:6})"
                    f" paired r1 {ptxt:>12} [{c['outcome']}] | orig #1016 paired {qtxt}")
        lines.append(f"  M{int(m):>3} bits-same {cells[pair][m]['bits_equal_across_images']} "
                     f"eq {eq_all} img-ok {img_ok}")
        for st in STATES:
            lines.append("       " + fmt(st))
floor = {}
for pair in cells:
    shape = pair.split("_")[1]
    for m, cm in cells[pair].items():
        if cm["r1_same_kernels_as_1107"]:
            for st, cs in cm["states"].items():
                k = f"{shape}/{st}"
                floor[k] = max(floor.get(k, 0.0), abs(cs["x_img_delta_us"]))
lines.append("== A/A floor: max |x-img delta| over same-kernels cells (us)")
lines.append("  " + "  ".join(f"{k} {v:.2f}" for k, v in sorted(floor.items())))
lc = {}
lines.append("== layer_chain (graph replay, x-img r1 - no1016)")
for key in B["r1-boot1"]["layer_chain"]:
    row = {}
    for st in ("graph_warm_us", "graph_cold_us"):
        b = [B[k]["layer_chain"][key][st] for k in BASE]
        c = [B[k]["layer_chain"][key][st] for k in CAND]
        dx = (c[0]["mean"] + c[1]["mean"]) / 2 - (b[0]["mean"] + b[1]["mean"]) / 2
        cix = math.sqrt(sum(x["ci95"] ** 2 for x in b + c)) / 2
        row[st] = {"no1016_b1_b2": [f"{x['mean']:.2f}±{x['ci95']:.2f}" for x in b],
                   "r1_b1_b2": [f"{x['mean']:.2f}±{x['ci95']:.2f}" for x in c],
                   "x_img_delta_us": round(dx, 2), "x_img_ci95": round(cix, 2)}
    dg = {k: B[k]["layer_chain"][key]["digests_sha256"] for k in B}
    row["digests_equal_across_images"] = len(set(dg.values())) == 1
    lc[key] = row
    lines.append(f"  {key:9} warm {row['graph_warm_us']['x_img_delta_us']:+7.2f}±{row['graph_warm_us']['x_img_ci95']:.2f}"
                 f"  cold {row['graph_cold_us']['x_img_delta_us']:+7.2f}±{row['graph_cold_us']['x_img_ci95']:.2f}"
                 f"  digests-equal {row['digests_equal_across_images']}")
print("\n".join(lines))
json.dump({"cells": cells, "aa_floor_x_img_us": floor, "layer_chain": lc}, open(f"{D}/cells.json", "w"), indent=1)
