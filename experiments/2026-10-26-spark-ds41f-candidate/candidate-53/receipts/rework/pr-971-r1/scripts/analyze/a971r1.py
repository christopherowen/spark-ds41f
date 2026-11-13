"""#971 round 1 analysis: ab-*.jsonl (policy-driven image A/B, three arms), script-r1-boot*.txt
(the PR's screen on r1) and pytest-r1-boot*.txt. Writes analysis.json beside the raw outputs.
Cell rule: analyze/common.py (the candidate-53 own-mark rule, unchanged)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from common import cell, jl
d = sys.argv[1]
labs = {"base": ("base-boot1", "base-boot2"), "r1": ("r1-boot1", "r1-boot2"), "orig": ("orig-boot1", "orig-boot2")}
ab = {l: jl(f"{d}/ab-{l}.jsonl") for arm in labs for l in labs[arm]}
kinds = lambda l, k: [r for r in ab[l] if r.get("kind") == k]
strip = lambda r: {k: v for k, v in r.items() if k != "label"}
out = {"identity": {l: strip(kinds(l, "identity")[0]) for l in ab}}
out["gate_copy_apply"] = {l: [(r["offered"], r["admits_copy"]) for r in kinds(l, "gate_copy_apply")] for l in ab}
pol = {l: {r["tau_name"]: strip(r) for r in kinds(l, "policy")} for l in ab}
out["policy"] = pol
def t(l, world, cache, g, live):
    for r in kinds(l, "timing_step"):
        if r["world"] == world and r["cache_mode"] == cache and r["graph_rows"] == g and r["live_routed_rows"] == live:
            return r
    raise KeyError((l, world, cache, g, live))
def shape(l, tau_name, scen):
    p = pol[l][tau_name]
    if scen == "initial_step":
        return p["initial_step"]["graph_rows"], p["initial_step"]["live_rows"]
    return p["shrunk_step"]["graph_rows"], p["shrunk_step"]["unmasked_rows"]
def compare(A, C, tau_name, world, cache, scen):
    B = [t(l, world, cache, *shape(l, tau_name, scen)) for l in labs[A]]
    Cc = [t(l, world, cache, *shape(l, tau_name, scen)) for l in labs[C]]
    c = cell([x["est_58layer_step_ms"] for x in B], [x["est_58layer_step_ms"] for x in Cc],
             samples_b=[s * 58 for x in B for s in x["samples_ms"]], samples_c=[s * 58 for x in Cc for s in x["samples_ms"]])
    c[f"{A}_shape_graph_x_live"] = list(shape(labs[A][0], tau_name, scen)); c[f"{C}_shape_graph_x_live"] = list(shape(labs[C][0], tau_name, scen))
    return c
for A, C in (("base", "r1"), ("orig", "r1"), ("base", "orig")):
    cells = {}
    for tau_name in ("recipe", "pr"):
        for world in (3, 4):
            for cache in ("warm", "128MiB_wash"):
                for scen in ("initial_step", "shrunk_step"):
                    cells[f"tau_{tau_name}/TP{world}/{cache}/{scen}"] = compare(A, C, tau_name, world, cache, scen)
    out[f"cells_{A}_to_{C}"] = cells
aa = {}
for A, C in (("base", "r1"), ("orig", "r1"), ("base", "orig")):
    for world in (3, 4):
        for cache in ("warm", "128MiB_wash"):
            for g, live in ((24, 17), (16, 16), (24, 6), (16, 6)):
                aa[f"{A}_vs_{C}/TP{world}/{cache}/{g}x{live}"] = cell(
                    [t(l, world, cache, g, live)["est_58layer_step_ms"] for l in labs[A]],
                    [t(l, world, cache, g, live)["est_58layer_step_ms"] for l in labs[C]])
out["same_shape_across_arms"] = aa
# policy identity, r1 vs orig, per tau setting (every boot)
out["policy_identity_r1_vs_orig"] = {tn: all(strip(pol[a][tn]) == strip(pol[b][tn]) for a in labs["r1"] for b in labs["orig"])
                                     for tn in ("recipe", "pr")}
out["policy_identity_r1_vs_base"] = {tn: all(strip(pol[a][tn]) == strip(pol[b][tn]) for a in labs["r1"] for b in labs["base"])
                                     for tn in ("recipe", "pr")}
# dead-row grid
keyf = lambda r: (r["tau_mode"], r["width"], r["admitted"], r["drafts"])
grid = {l: {keyf(r): strip(r) for r in kinds(l, "dead_row_case")} for l in ab}
def grid_diff(a, b):
    return [list(k) for k in grid[a] if grid[a][k] != grid[b].get(k)]
out["dead_row_grid"] = {
    "cases": len(grid["r1-boot1"]),
    "boot_to_boot_identical": {arm: grid[labs[arm][0]] == grid[labs[arm][1]] for arm in labs},
    "r1_vs_orig_differing_cases": grid_diff("r1-boot1", "orig-boot1"),
    "r1_vs_base_differing_cases": grid_diff("r1-boot1", "base-boot1"),
    "orig_vs_base_differing_cases": grid_diff("orig-boot1", "base-boot1"),
    "recipe_cases_r1_vs_orig_identical": all(grid["r1-boot1"][k] == grid["orig-boot1"][k] for k in grid["r1-boot1"] if k[0] != "zero"),
    "examples": {arm: {"/".join(map(str, k)): {x: grid[labs[arm][0]][k][x] for x in ("effective_tau", "dead_row_args", "log_tau", "cut", "unmasked_rows", "verified")}
                       for k in grid[labs[arm][0]] if k[3] == "shrunk_copy" and k[2]} for arm in labs},
}
out["assign_draft_budget_one_request"] = {l: {r["confidences"]: r["equals_budget"] for r in kinds(l, "assign_draft_budget_one_request")} for l in ab}
out["declare_dead_rows"] = {l: [strip(r) for r in kinds(l, "declare_dead_rows")] for l in ab}
# PR screen on r1
pr = {}
for k in (1, 2):
    p = f"{d}/script-r1-boot{k}.txt"
    if os.path.exists(p):
        txt = open(p).read()
        rows = jl(p)
        tail = [ln for ln in txt.splitlines() if ln.startswith(("AssertionError", "Traceback")) or "assert " in ln][-3:]
        pr[f"boot{k}"] = {"gate": [strip(r) for r in rows if r.get("kind") in ("gate_copy_apply_yield", "gate_adaptive_tile_and_dead_rows")],
                          "copy_apply_kernel_us": [(r["cache_mode"], r["mean_us"]) for r in rows if r.get("kind") == "timing_copy_apply_kernel"],
                          "stopped_at": tail}
out["pr_script_r1"] = pr
tests = {}
for k in (1, 2):
    p = f"{d}/pytest-r1-boot{k}.txt"
    if os.path.exists(p):
        last = [ln for ln in open(p).read().splitlines() if re.search(r"\d+ passed", ln)]
        tests[f"boot{k}"] = last[-1].strip("= ") if last else None
out["pytest_r1"] = tests
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
for l in ab: print(l, "gate", out["gate_copy_apply"][l])
for l in ab: print(l, "policy recipe", json.dumps({k: v for k, v in pol[l]["recipe"].items() if k not in ("kind",)}))
for A, C in (("base", "r1"), ("orig", "r1")):
    for k, v in out[f"cells_{A}_to_{C}"].items():
        print(f"{A}->{C}", k, v[f"{A}_shape_graph_x_live"], "->", v[f"{C}_shape_graph_x_live"], v["base"], "->", v["candidate"], v["delta_pct"], "%", v["verdict"], v.get("welch95"))
print("policy identity r1 vs orig", out["policy_identity_r1_vs_orig"])
print("dead-row grid", json.dumps({k: v for k, v in out["dead_row_grid"].items() if k != "examples"}))
print("assign", out["assign_draft_budget_one_request"])
for l in ab: print(l, "declare", out["declare_dead_rows"][l])
print("pr script", json.dumps(pr)[:1500])
print("pytest", tests)
