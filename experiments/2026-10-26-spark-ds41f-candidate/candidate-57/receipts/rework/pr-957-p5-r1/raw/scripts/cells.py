#!/usr/bin/env python3
"""Replay cells for result.json: per profile (#986 r1 fit, TP3 served fit), seed, concurrency and arm, the
budget change and the predicted step-time and throughput change; plus #986's model M (its blend of model
A and model B, weights calibrated on the 2026-10-18 row-cost A/B) at c1 and c8.
usage: cells.py <raw/out dir> <#986 replay-seed0.json> <#986 replay-seed1.json> <out.json>"""
import json
import sys

out_dir, f0, f1, dest = sys.argv[1:5]
weights = {s: json.load(open(p))["concurrency"] for s, p in ((0, f0), (1, f1))}
cells = {}
for prof in ("986fit", "tp3fit"):
    for seed in (0, 1):
        d = json.load(open(f"{out_dir}/replay-{prof}-seed{seed}.json"))
        for c, r in d["concurrency"].items():
            for arm, a in r["arms"].items():
                if arm in ("c54@tp3", "c54@tp4"):
                    continue
                p = a["prediction"]
                base = r["arms"][p["vs"]]
                cell = {
                    "profile": prof, "seed": seed, "concurrency": c, "arm": arm, "vs": p["vs"],
                    "effective_dead_rows_tau": round(a["effective_dead_rows_tau"], 4),
                    "draft_budget_per_step": [round(base["draft_budget_mean"], 3), round(a["draft_budget_mean"], 3)],
                    "rows_per_step": [round(base["rows_mean"], 3), round(a["rows_mean"], 3)],
                    "tokens_per_step": [round(base["tokens_mean"], 4), round(a["tokens_mean"], 4)],
                    "steps_budget_differs_pct": round(100 * a[f"steps_budget_differs_from_{p['vs']}"], 2),
                    "tokens_per_step_pct": round(p["tokens_per_step_pct"], 3),
                    "grid_gpu_step_ms": [round(base["grid_gpu_step_ms"], 3), round(a["grid_gpu_step_ms"], 3)],
                    "grid_gpu_step_pct": round(p["grid_gpu_step_pct"], 3),
                    "grid_interp_gpu_step_pct": round(p["interp_gpu_step_pct"], 3) if "interp_gpu_step_pct" in p else None,
                    "throughput_grid_gpu_only_pct": round(p["throughput_grid_gpu_only_pct"], 3),
                }
                if "throughput_model_A_pct" in p:
                    cell["served_step_ms_model_A"] = [round(r["anchor"]["overhead_A_ms"] + base["grid_gpu_step_ms"], 3),
                                                      round(p["served_step_ms_model_A"], 3)]
                    cell["served_step_pct_model_A"] = round(p["served_step_pct_model_A"], 3)
                    cell["throughput_model_A_pct"] = round(p["throughput_model_A_pct"], 3)
                    cell["throughput_model_A_pct_bootstrap95"] = [round(x, 3) for x in p["throughput_model_A_pct_bootstrap95"]]
                cell["throughput_model_B_tp4_regression_pct"] = round(p["throughput_model_B_tp4_regression_pct"], 3)
                if c in ("c1", "c8") and "throughput_model_A_pct" in p:
                    TA0 = cell["served_step_ms_model_A"][0]; TA1 = p["served_step_ms_model_A"]
                    TB0 = base["model_B_step_ms"]; TB1 = a["model_B_step_ms"]
                    E0 = base["tokens_mean"]; E1 = a["tokens_mean"]
                    m = {}
                    for wn, w in weights[seed][c]["model_M_weight_on_A"].items():
                        m[wn] = round(100 * ((E1 / (w * TA1 + (1 - w) * TB1)) / (E0 / (w * TA0 + (1 - w) * TB0)) - 1), 3)
                    cell["throughput_model_M_pct"] = m
                cells.setdefault(prof, {}).setdefault(c, {}).setdefault(arm, []).append(cell)
json.dump(cells, open(dest, "w"), indent=1)
for prof, pc in cells.items():
    for c, arms in pc.items():
        for arm, lst in arms.items():
            if arm not in ("p5@tp3", "p5-scale-only@tp3", "p5-floor-only@tp3"):
                continue
            s = " | ".join(
                f"s{x['seed']}: b {x['draft_budget_per_step'][0]}->{x['draft_budget_per_step'][1]} tok {x['tokens_per_step_pct']:+.2f}% "
                f"gpu {x['grid_gpu_step_pct']:+.2f}% thrGPU {x['throughput_grid_gpu_only_pct']:+.2f}% "
                f"thrA {x.get('throughput_model_A_pct', float('nan')):+.2f}% B {x['throughput_model_B_tp4_regression_pct']:+.2f}% "
                f"M {x.get('throughput_model_M_pct', {})}" for x in lst)
            print(prof, c, arm, s)
