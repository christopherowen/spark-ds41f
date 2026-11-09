"""#971 analysis: ab-*.jsonl (policy-driven image A/B) and script-c53-boot*.txt (the PR's screen)."""
import json, sys, os, statistics
sys.path.insert(0, os.path.dirname(__file__)); from common import cell, jl
d = sys.argv[1]
labs = ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")
ab = {l: jl(f"{d}/ab-{l}.jsonl") for l in labs}
pol = {l: {r["tau_name"]: r for r in ab[l] if r.get("kind") == "policy"} for l in labs}
gate = {l: [(r["offered"], r["admits_copy"]) for r in ab[l] if r.get("kind") == "gate_copy_apply"] for l in labs}
def t(l, world, cache, g, live):
    for r in ab[l]:
        if r.get("kind") == "timing_step" and r["world"] == world and r["cache_mode"] == cache and r["graph_rows"] == g and r["live_routed_rows"] == live:
            return r
out = {"gate_copy_apply": gate, "policy": {l: pol[l] for l in labs}, "cells": {}, "kernel_identity_same_shape_across_images": {}}
for tau_name in ("recipe", "pr"):
    for world in (3, 4):
        for cache in ("warm", "128MiB_wash"):
            for scen in ("initial_step", "shrunk_step"):
                def shape(l):
                    p = pol[l][tau_name]
                    if scen == "initial_step":
                        return p["initial_step"]["graph_rows"], p["initial_step"]["live_rows"]
                    return p["shrunk_step"]["graph_rows"], p["shrunk_step"]["unmasked_rows"]
                B = [t(l, world, cache, *shape(l)) for l in ("base-boot1", "base-boot2")]
                C = [t(l, world, cache, *shape(l)) for l in ("c53-boot1", "c53-boot2")]
                key = f"tau_{tau_name}/TP{world}/{cache}/{scen}"
                c = cell([x["est_58layer_step_ms"] for x in B], [x["est_58layer_step_ms"] for x in C],
                         samples_b=[s * 58 for x in B for s in x["samples_ms"]], samples_c=[s * 58 for x in C for s in x["samples_ms"]])
                c["base_shape"] = list(shape("base-boot1")); c["candidate_shape"] = list(shape("c53-boot1"))
                out["cells"][key] = c
for world in (3, 4):
    for cache in ("warm", "128MiB_wash"):
        for g, live in ((24, 17), (16, 16), (24, 6), (16, 6)):
            B = [t(l, world, cache, g, live)["est_58layer_step_ms"] for l in ("base-boot1", "base-boot2")]
            C = [t(l, world, cache, g, live)["est_58layer_step_ms"] for l in ("c53-boot1", "c53-boot2")]
            out["kernel_identity_same_shape_across_images"][f"TP{world}/{cache}/{g}x{live}"] = cell(B, C)
pr = {}
for k in (1, 2):
    p = f"{d}/script-c53-boot{k}.txt"
    if os.path.exists(p):
        rows = jl(p)
        pr[f"boot{k}"] = {"gate": [r for r in rows if r.get("kind") in ("gate_copy_apply_yield", "gate_adaptive_tile_and_dead_rows")],
                          "timing": [(r["world"], r["scenario"], r["cache_mode"], r["variant"], r["est_58layer_step_ms"]) for r in rows if r.get("kind") == "timing_step"]}
out["pr_script_candidate_image"] = pr
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
for l in labs: print(l, "gate", gate[l]); print("  policy recipe", json.dumps({k: v for k, v in pol[l]["recipe"].items() if k not in ("label", "kind")}))
for l in labs: print("  policy pr", json.dumps({k: v for k, v in pol[l]["pr"].items() if k not in ("label", "kind")}))
for k, v in out["cells"].items(): print(k, v["base_shape"], "->", v["candidate_shape"], v["base"], "->", v["candidate"], v["delta_pct"], "%", v["verdict"], v.get("welch95"))
for k, v in out["kernel_identity_same_shape_across_images"].items(): print("same-shape", k, v["base"], v["candidate"], v["delta_pct"], v["verdict"])
