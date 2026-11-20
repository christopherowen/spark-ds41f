"""Tables of the grid-fidelity runs: routes, step time and the kernel split.

usage: python3 analyze.py OUT_DIR RESULT_JSON

Reads OUT_DIR/json/{census,step-*,nsys-*}.json, OUT_DIR/nsys/nsys-*.sqlite and
OUT_DIR/ref-routes.pt (torch only for the last), and writes one JSON of:
- census: per input and shape, the target's experts hit per layer (and rows routed), the
  drafter's per layer;
- reference: real text through the checkpoint's gates, experts hit per layer at each grid
  shape (r requests x k rows: the first r sequences' rows 64..64+k-1) and for token 0;
- timing: per shape and input, each boot's mean step (slow outliers dropped as servebench
  does), the input's mean over boots, and each input against the grid boots of its own
  ABBA block (sample: grid boots 1-2; routes: grid boots 3-4), with a Welch interval over
  the four boots' means;
- kernels: per shape and input, routed-expert and router kernel time a step (nsys, graph
  nodes traced) and their share of the timing boots' step.
"""

import glob
import json
import math
import os
import re
import sqlite3
import statistics
import sys

sys.path.insert(0, "/sb")
from servebench import stats  # noqa: E402

OUT, RESULT = sys.argv[1], sys.argv[2]
GRID = ["1x2", "4x8", "8x16", "16x32", "1x3", "4x12", "8x24", "16x48",
        "1x6", "2x12", "4x24", "8x48", "12x72", "16x96"]
ROUTED = re.compile(r"moe_(grouped|swiglu|dispatch|combine)")
ROUTER = re.compile(r"router_partials|topk_gate|routing_mask|route_partials")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def census():
    path = os.path.join(OUT, "json", "census.json")
    if not os.path.exists(path):
        return None
    data = load(path)
    out = {"moe_skip_padding": data["moe_skip_padding"], "eager": data["eager"], "rows": []}
    for r in data["results"]:
        target = [x for x in r["layers"] if x["router"].startswith("target")]
        drafter = [x for x in r["layers"] if x["router"].startswith("drafter")]
        hits = [x["experts_hit"] for x in target]
        out["rows"].append({
            "input": r["input"], "shape": r["shape"], "real_rows": r["real_rows"],
            "target_layers": len(target), "target_rows": target[0]["rows"] if target else None,
            "target_rows_routed": sorted({x["rows_routed"] for x in target}),
            "target_experts_hit_per_layer": hits,
            "target_experts_hit_mean": round(sum(hits) / len(hits), 2) if hits else None,
            "drafter_experts_hit_per_layer": [x["experts_hit"] for x in drafter],
            "drafter_rows_routed": [x["rows_routed"] for x in drafter],
        })
    return out


def reference():
    path = os.path.join(OUT, "ref-routes.pt")
    if not os.path.exists(path):
        return None
    import torch

    d = torch.load(path, weights_only=False)
    routes = d["routes"]  # [layers, sequences, positions, 6]
    c = d["context"]
    text = routes[:, :16]
    out = {"layers": int(routes.shape[0]), "experts": 384, "top_k": 6, "context": c,
           "seconds": d["seconds"], "bytes_read": d["bytes_read"], "shapes": {}}
    for shape in GRID:
        r, t = map(int, shape.split("x"))
        k = t // r
        per = [int(torch.unique(text[l, :r, c:c + k]).numel()) for l in range(routes.shape[0])]
        out["shapes"][shape] = {"experts_hit_per_layer": per, "mean": round(sum(per) / len(per), 2),
                                "min": min(per), "max": max(per)}
    for r, k, name in ((16, 3, "c16_16x3"),):
        per = [int(torch.unique(text[l, :r, c:c + k]).numel()) for l in range(routes.shape[0])]
        out[name] = {"experts_hit_per_layer": per, "mean": round(sum(per) / len(per), 2),
                     "min": min(per), "max": max(per)}
    zero = routes[:, 16]
    for k in (1, 3, 6):
        per = [int(torch.unique(zero[l, c:c + k]).numel()) for l in range(routes.shape[0])]
        out[f"token0_{k}_positions"] = {"experts_hit_per_layer": per, "mean": round(sum(per) / len(per), 2)}
    # Uniform-random routing of the same rows, for scale: 384 (1 - (1 - 6/384)^rows).
    out["uniform_random_expected"] = {s: round(384 * (1 - (1 - 6 / 384) ** int(s.split("x")[1])), 1) for s in GRID}
    return out


def boot_mean(samples):
    kept, _ = stats.trim_slow(samples)
    return sum(kept) / len(kept)


def timing():
    files = sorted(glob.glob(os.path.join(OUT, "json", "step-*-boot*.json")))
    boots = {}
    for path in files:
        data = load(path)
        name = os.path.basename(path)[len("step-"):-len(".json")]
        boots[name] = data
    if not boots:
        return None
    out = {"boots": {}, "shapes": []}
    for name, data in boots.items():
        out["boots"][name] = {"inputs": data["inputs"], "utc": data["utc"], "boot_s": data["boot_s"],
                              "real_rows": sorted({str(r["real_rows"]) for r in data["rows"]}),
                              "replays": data["spec"]["replays"]}
    blocks = {"sample": ("grid-boot1", "grid-boot2"), "routes": ("grid-boot3", "grid-boot4")}
    for shape in GRID:
        row = {"shape": shape}
        means, by_boot = {}, {}
        for name, data in boots.items():
            hit = next((r for r in data["rows"] if r["shape"] == shape), None)
            if hit is None:
                continue
            key = "routes" if data.get("routes") else data["inputs"]
            by_boot[name] = boot_mean(hit["step_ms"])
            means.setdefault(key, []).append(by_boot[name])
            row[f"{name}_ms"] = round(boot_mean(hit["step_ms"]), 3)
            row[f"{name}_median_ms"] = round(statistics.median(hit["step_ms"]), 3)
            if hit.get("forward_ms"):
                row[f"{name}_drafter_ms"] = round(statistics.median(hit["drafter_ms"]), 3)
        for inputs, values in means.items():
            row[f"{inputs}_mean_ms"] = round(sum(values) / len(values), 3)
        for inputs, grid_boots in blocks.items():
            base = [by_boot[b] for b in grid_boots if b in by_boot]
            cand = means.get(inputs, [])
            if len(base) >= 2 and len(cand) >= 2:
                test = stats.welch(base, cand)
                row[f"{inputs}_block_grid_ms"] = round(sum(base) / len(base), 3)
                row[f"{inputs}_vs_grid_pct"] = round(test["change_pct"], 2)
                row[f"{inputs}_vs_grid_ci95_pct"] = round(test["ci95_pct"], 2) if test["ci95_pct"] is not None else None
                row[f"{inputs}_minus_grid_ms"] = round(sum(cand) / len(cand) - sum(base) / len(base), 3)
        out["shapes"].append(row)
    for inputs in blocks:
        logs = [math.log(1 + r[f"{inputs}_vs_grid_pct"] / 100) for r in out["shapes"] if f"{inputs}_vs_grid_pct" in r]
        if logs:
            out[f"geomean_{inputs}_over_grid"] = round(math.exp(sum(logs) / len(logs)), 4)
    base = glob.glob(os.path.join(OUT, "json", "step-grid-base.json"))
    if base:
        data = load(base[0])
        ctl = {}
        for shape in GRID:
            hit = next((r for r in data["rows"] if r["shape"] == shape), None)
            if hit:
                ctl[shape] = round(boot_mean(hit["step_ms"]), 3)
        out["control_base_image_grid_ms"] = ctl
        logs = [math.log(ctl[r["shape"]] / r["grid_mean_ms"]) for r in out["shapes"]
                if r["shape"] in ctl and "grid_mean_ms" in r]
        out["control_base_over_patched_grid_geomean"] = round(math.exp(sum(logs) / len(logs)), 4) if logs else None
    return out


def kernels(step_means):
    out = {}
    for path in sorted(glob.glob(os.path.join(OUT, "nsys", "nsys-*.sqlite"))):
        inputs = os.path.basename(path)[len("nsys-"):-len(".sqlite")]
        meta = load(os.path.join(OUT, "json", f"nsys-{inputs}.json"))
        runs = {p["shape"]: p["runs"] for p in meta["profiled"]}
        con = sqlite3.connect(path)
        names = dict(con.execute("select id, value from StringIds"))
        ranges = con.execute(
            "select start, end, coalesce(text, (select value from StringIds where id = textId)) "
            "from NVTX_EVENTS where end is not null").fetchall()
        kern = con.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
        rows = {}
        for start, end, text in ranges:
            if not text or not text.startswith("shape "):
                continue
            shape = text[len("shape "):]
            totals = {"routed": 0.0, "router": 0.0, "other": 0.0}
            by_name = {}
            for ks, ke, name_id in kern:
                if ks >= start and ke <= end:
                    name = names.get(name_id, "?")
                    group = "routed" if ROUTED.search(name) else "router" if ROUTER.search(name) else "other"
                    totals[group] += (ke - ks) / 1e6
                    if group != "other":
                        by_name[name] = by_name.get(name, 0.0) + (ke - ks) / 1e6
            n = runs[shape]
            step = (step_means.get(inputs) or {}).get(shape)
            rows[shape] = {
                "runs": n,
                "routed_expert_ms_per_step": round(totals["routed"] / n, 3),
                "router_ms_per_step": round(totals["router"] / n, 3),
                "all_kernels_ms_per_step": round(sum(totals.values()) / n, 3),
                "moe_kernels_by_name_ms_per_step": {k: round(v / n, 4) for k, v in sorted(by_name.items())},
                "timed_step_ms": step,
                "routed_share_of_step_pct": round(100 * totals["routed"] / n / step, 2) if step else None,
                "routed_and_router_share_of_step_pct":
                    round(100 * (totals["routed"] + totals["router"]) / n / step, 2) if step else None,
            }
        out[inputs] = rows
    return out


result = {"census": census(), "reference": reference(), "timing": timing()}
step_means = {}
if result["timing"]:
    for row in result["timing"]["shapes"]:
        for inputs in ("grid", "sample", "routes"):
            if f"{inputs}_mean_ms" in row:
                step_means.setdefault(inputs, {})[row["shape"]] = row[f"{inputs}_mean_ms"]
result["kernels"] = kernels(step_means)
with open(RESULT, "w") as fh:
    json.dump(result, fh, indent=1)
print("wrote", RESULT)
