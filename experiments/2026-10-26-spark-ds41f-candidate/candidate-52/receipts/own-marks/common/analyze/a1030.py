#!/usr/bin/env python3
"""a1030.py OUTDIR: summary of PR #1030's own-mark runs (PR bench + h1030 harness), ABBA on sp8.
Prints a table and writes OUTDIR/summary.json."""
import json
import os
import sys

O = sys.argv[1]
ARMS = [("base", 1), ("c52", 1), ("c52", 2), ("base", 2)]
SH = ["tp4_inter576", "tp3_inter768"]
TI = ["decode_blockM16_rows16", "mid_blockM32_rows96", "prefill_blockM64_rows2048"]


def ld(p):
    p = os.path.join(O, p)
    return json.load(open(p)) if os.path.exists(p) else None


def rng(xs):
    return [min(xs), max(xs)]


def verdict_slower(c, b):
    """'slower' when the candidate's two boots are both above the base's two boots (non-overlapping)."""
    if min(c) > max(b):
        return "slower"
    if max(c) < min(b):
        return "faster"
    return "within"


S = {"pr_bench": {}, "harness": {}, "e2e": {}}
lines = []
pb = {f"{a}-boot{k}": ld(f"bench-{a}-boot{k}.json") for a, k in ARMS}
hh = {f"{a}-boot{k}": ld(f"h1030-{a}-boot{k}.json") for a, k in ARMS}
lines.append("PR bench (eager per-launch CUDA events, p50 us; diffs vs the arm's test-file reference)")
for s in SH:
    for t in TI:
        row = {}
        for lab, d in pb.items():
            if not d:
                continue
            r = d["swiglu_cast"][s][t]
            row[lab] = {"cand_diffs": [r["candidate"]["fp8_byte_diffs_vs_deepseek_ref"], r["candidate"]["scale_byte_diffs_vs_deepseek_ref"]],
                        "old_diffs": [r["baseline_1107"]["fp8_byte_diffs_vs_deepseek_ref"], r["baseline_1107"]["scale_byte_diffs_vs_deepseek_ref"]],
                        "cand_p50": r["candidate"]["p50_us"], "old_p50": r["baseline_1107"]["p50_us"]}
        S["pr_bench"][f"{s}/{t}"] = row
        lines.append(f"  {s} {t}: " + " | ".join(f"{k}: img {v['cand_p50']} old {v['old_p50']} diffs img {v['cand_diffs']} old {v['old_diffs']}" for k, v in row.items()))
for lab, d in pb.items():
    if not d:
        continue
    e = d["e2e_moe"]
    S["pr_bench"].setdefault("e2e", {})[lab] = {
        "latency_p50": {r: [e["latency"][r]["baseline_1107"]["p50_us"], e["latency"][r]["candidate"]["p50_us"]] for r in ("rows12", "rows300")},
        "accuracy_ulp_gt0": {r: {n: [v["baseline_1107_vs_deepseek_ref"]["ulp_gt0"], v["candidate_vs_deepseek_ref"]["ulp_gt0"]]
                                 for n, v in e["accuracy_by_regime"][r].items()} for r in ("rows12", "rows300")},
        "accuracy_rel_l1": {r: {n: [v["baseline_1107_vs_deepseek_ref"]["rel_l1"], v["candidate_vs_deepseek_ref"]["rel_l1"]]
                                for n, v in e["accuracy_by_regime"][r].items()} for r in ("rows12", "rows300")},
        "stack_10_layer_ulp_gt0": [[x["layer"], x["baseline_1107_vs_deepseek_ref"]["ulp_gt0"], x["candidate_vs_deepseek_ref"]["ulp_gt0"]] for x in e["stack_10_layer"]],
    }
lines.append("  e2e (PR bench; [old1107, image]): " + json.dumps(S["pr_bench"].get("e2e")))

lines.append("")
lines.append("Harness (CUDA-graph replay us/launch, median of 60 windows; diffs vs candidate-52's DeepSeek reference in both arms)")
for s in SH:
    for t in TI:
        cell = {"image_graph": {}, "old_graph": {}, "image_eager": {}, "image_diffs": {}, "old_diffs": {},
                "ref_vs_torch": {}, "input_sha": {}, "image_equals_old": {}}
        for lab, d in hh.items():
            if not d:
                continue
            r = d["swiglu_cast"][s][t]
            cell["image_graph"][lab] = r["image"]["graph_us_per_launch"]["median"]
            cell["old_graph"][lab] = r["old1107"]["graph_us_per_launch"]["median"]
            cell["image_eager"][lab] = r["image"]["eager_us_per_launch"]["median"]
            cell["image_diffs"][lab] = r["image"]["diffs_vs_ds_ref"]
            cell["old_diffs"][lab] = r["old1107"]["diffs_vs_ds_ref"]
            cell["ref_vs_torch"][lab] = r["ref_ds_vs_torch_floor"]
            cell["input_sha"][lab] = r["input_sha"]
            cell["image_equals_old"][lab] = r["image_equals_old1107"]
            cell["live_rows"] = r["live_rows"]
            cell["live_rows_pr_mask"] = r["live_rows_pr_mask"]
        b = [cell["image_graph"][k] for k in cell["image_graph"] if k.startswith("base")]
        c = [cell["image_graph"][k] for k in cell["image_graph"] if k.startswith("c52")]
        if b and c:
            cell["graph_base_range"] = rng(b)
            cell["graph_c52_range"] = rng(c)
            cell["graph_cross_arm"] = verdict_slower(c, b)
            cell["graph_c52_over_base_mean"] = round((sum(c) / len(c)) / (sum(b) / len(b)), 4)
            # within-process paired ratio image/old1107 (on base an A/A, on c52 candidate vs 1107)
            cell["paired_image_over_old"] = {k: round(cell["image_graph"][k] / cell["old_graph"][k], 4) for k in cell["image_graph"]}
        S["harness"][f"{s}/{t}"] = cell
        lines.append(f"  {s} {t}: image graph {cell['image_graph']} old {cell['old_graph']} -> cross-arm {cell.get('graph_cross_arm')}"
                     f" (c52/base {cell.get('graph_c52_over_base_mean')}); paired img/old {cell.get('paired_image_over_old')}")
        lines.append(f"      image diffs {cell['image_diffs']} | old diffs {cell['old_diffs']} | ref vs torch {cell['ref_vs_torch']}"
                     f" | inputs {set(cell['input_sha'].values())} | live {cell.get('live_rows')} (PR mask {cell.get('live_rows_pr_mask')})")
for rows in ("rows12", "rows300"):
    cell = {"image_graph": {}, "old_graph": {}, "acc": {}, "capture_error": {}}
    for lab, d in hh.items():
        if not d:
            continue
        r = d["e2e_moe"][rows]
        tm = r["timing"]
        key = "graph_us_per_launch" if "graph_us_per_launch" in tm["image"] else "eager_us_per_launch"
        cell["timing_kind"] = key
        cell["image_graph"][lab] = tm["image"][key]["median"]
        cell["old_graph"][lab] = tm["old1107"][key]["median"]
        cell["capture_error"][lab] = r.get("graph_capture_error")
        cell["acc"][lab] = {n: {a: [v[a]["vs_ds_ref"]["ulp_gt0"], v[a]["vs_ds_ref"]["rel_l1"]] for a in ("image", "old1107")}
                            for n, v in r["accuracy"].items()}
    b = [cell["image_graph"][k] for k in cell["image_graph"] if k.startswith("base")]
    c = [cell["image_graph"][k] for k in cell["image_graph"] if k.startswith("c52")]
    if b and c:
        cell["base_range"], cell["c52_range"] = rng(b), rng(c)
        cell["cross_arm"] = verdict_slower(c, b)
        cell["c52_over_base_mean"] = round((sum(c) / len(c)) / (sum(b) / len(b)), 4)
        cell["paired_image_over_old"] = {k: round(cell["image_graph"][k] / cell["old_graph"][k], 4) for k in cell["image_graph"]}
    S["e2e"][rows] = cell
    lines.append(f"  e2e {rows} ({cell.get('timing_kind')}): image {cell['image_graph']} old {cell['old_graph']} -> {cell.get('cross_arm')} "
                 f"(c52/base {cell.get('c52_over_base_mean')}); paired {cell.get('paired_image_over_old')}")
    lines.append(f"      acc [ulp>0, rel_l1] vs DeepSeek ref: {json.dumps(cell['acc'])}")
print("\n".join(lines))
json.dump(S, open(os.path.join(O, "summary.json"), "w"), indent=1)
open(os.path.join(O, "summary.txt"), "w").write("\n".join(lines) + "\n")
