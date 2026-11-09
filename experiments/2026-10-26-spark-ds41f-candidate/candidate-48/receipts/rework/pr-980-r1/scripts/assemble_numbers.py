"""assemble_numbers.py <work-dir>: the measured numbers of the #980 r1 rework, from out/."""
import glob
import json
import math
import os
import re
import sys

W = sys.argv[1]
O = os.path.join(W, "out")
res = {}


def jl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def det(records):
    out = {}
    for r in records:
        if r.get("kind") == "determinism_1000_steps":
            out[r["mode"]] = {k: r[k] for k in r if k.startswith("mismatch") or k in (
                "unchecked_sampled_raw_steps",
                "seeds_equal_across_ranks", "broadcast_calls", "refits_completed",
                "max_conf_abs_diff", "max_slope_abs_diff", "max_intercept_abs_diff")}
    return out


def crash(label):
    p = os.path.join(O, f"{label}.txt")
    if not os.path.exists(p):
        return None
    t = open(p, errors="replace").read()
    m = re.search(r"File \"([^\"]+rejection_sampler_utils.py)\", line (\d+), in (\w+)", t)
    return {
        "illegal_memory_access": "illegal memory access" in t,
        "in": f"{os.path.basename(m.group(1))}:{m.group(2)} {m.group(3)}" if m else None,
    }


def exit_of(label):
    t = open(os.path.join(W, "commands.txt")).read()
    m = re.search(rf"## {re.escape(label)}\nstart_utc (\S+) end_utc (\S+) exit (\d+)", t)
    return int(m.group(3)) if m else None


# tests
res["pytest"] = {}
for b in (1, 2):
    p = os.path.join(O, f"pytest-r1-boot{b}.txt")
    if os.path.exists(p):
        res["pytest"][f"boot{b}"] = open(p).read().strip().splitlines()[-1].strip("= ")

# seed follow-up: tests and the harness-fixed screen with a configured seed
res["pytest_r1seed"] = {}
for b in (1, 2):
    p = os.path.join(O, f"pytest-r1seed-boot{b}.txt")
    if os.path.exists(p):
        res["pytest_r1seed"][f"boot{b}"] = open(p).read().strip().splitlines()[-1].strip("= ")
res["screen_r1seed"] = {}
for label in ("screen-r1seed-boot1-tp3", "screen-r1seed-boot1-tp4", "screen-r1seed-boot2-tp4", "screen-r1seed-boot2-tp3"):
    recs = jl(os.path.join(O, f"{label}.jsonl"))
    res["screen_r1seed"][label] = {"exit": exit_of(label), "sampling_seed": next((r.get("sampling_seed") for r in recs if "sampling_seed" in r), None), "modes": det(recs)}

# lockstep screen (the mark)
res["screen_mark"] = {}
for label in ("screen-r1-boot1-tp3", "screen-r1-boot1-tp4", "screen-r1-boot2-tp4", "screen-r1-boot2-tp3"):
    res["screen_mark"][label] = {"exit": exit_of(label), "modes": det(jl(os.path.join(O, f"{label}.jsonl")))}

# TP2 diagnosis
res["tp2"] = {}
for label in ("screen-tp2-c48-orig", "screen-tp2-r1-orig", "screen-tp2-c48-rowsfix",
              "screen-tp2-r1-rowsfix", "screen-tp3-r1-rowsfix", "screen-tp4-r1-rowsfix",
              "screen-tp2-r1-valid", "screen-tp2-c48-valid", "screen-tp3-r1-valid",
              "screen-tp4-r1-valid", "screen-tp2-r1-valid-again"):
    res["tp2"][label] = {"exit": exit_of(label), "modes": det(jl(os.path.join(O, f"{label}.jsonl"))),
                         "crash": crash(label)}
for img in ("1107", "c48"):
    p = os.path.join(O, f"tp2-oob-repro-{img}.txt")
    if os.path.exists(p):
        line = [l for l in open(p) if l.startswith("SUMMARY ")]
        res["tp2"][f"oob-repro-{img}-image"] = json.loads(line[-1][8:]) if line else None

# warmup tokens
wt = {}
for p in sorted(glob.glob(os.path.join(O, "warmup-tokens-*.txt"))):
    for l in open(p):
        if l.startswith("WARMUP_TOKENS "):
            d = json.loads(l[len("WARMUP_TOKENS "):])
            wt[f"{d['arm']}-{d['boot']}"] = {k: d[k] for k in ("flashinfer_path", "seeds_sha256_16", "tokens_sha256_16")}
            wt[f"{d['arm']}-{d['boot']}"]["first_step_tokens_head"] = d["first_step_tokens"][:6]
res["warmup_tokens"] = wt

# executed changed lines
res["executed_changed_lines"] = {}
for a in ("c48", "r1", "r1seed"):
    p = os.path.join(O, f"executed-changed-lines-{a}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        res["executed_changed_lines"][a] = {
            c: {f.split("worker/gpu/")[-1]: v for f, v in fs.items()}
            for c, fs in d["executed_changed_lines"].items()}

# per-process step cost
p = os.path.join(O, "stepcost-summary.json")
if os.path.exists(p):
    s = json.load(open(p))
    res["stepcost_per_process"] = {
        c: {k: v[k] for k in ("geomean_wall_us", "geomean_wall_us_by_boot", "c48_vs_1107_pct",
                              "r1_vs_1107_pct", "aa_spread_pct")}
        for c, v in s["components"].items()}

# in-process A/B/C step cost
ab = {}
for b in ("boot1", "boot2"):
    p = os.path.join(O, f"stepcost-ab-{b}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    comps = {}
    for r in d["results"]:
        comps.setdefault(r["component"], []).append(r)
    summ = {}
    for c, rows in comps.items():
        def gm(arm, key):
            vals = [max(r[arm][key], 1e-3) for r in rows]
            return math.exp(sum(math.log(v) for v in vals) / len(vals))
        dh = [r["c48_minus_1107_host_us"] for r in rows]
        dr = [r["r1_minus_1107_host_us"] for r in rows]
        summ[c] = {
            "host_us_geomean": {a: round(gm(a, "host_us_mean"), 3) for a in ("1107", "c48", "r1")},
            "gpu_us_geomean": {a: round(gm(a, "gpu_us_mean"), 3) for a in ("1107", "c48", "r1")},
            "c48_minus_1107_host_us_mean_over_shapes": round(sum(dh) / len(dh), 4),
            "c48_minus_1107_host_us_range": [round(min(dh), 4), round(max(dh), 4)],
            "r1_minus_1107_host_us_mean_over_shapes": round(sum(dr) / len(dr), 4),
            "r1_minus_1107_host_us_range": [round(min(dr), 4), round(max(dr), 4)],
            "max_ci95_us": round(max(max(r["c48_minus_1107_host_ci95_us"], r["r1_minus_1107_host_ci95_us"]) for r in rows), 4),
            "shapes": {f"{r['num_reqs']}x{r['rows']}": {
                "host_us": {a: r[a]["host_us_mean"] for a in ("1107", "c48", "r1")},
                "c48_minus_1107_host_us": [r["c48_minus_1107_host_us"], r["c48_minus_1107_host_ci95_us"]],
                "r1_minus_1107_host_us": [r["r1_minus_1107_host_us"], r["r1_minus_1107_host_ci95_us"]],
            } for r in rows},
        }
    ab[b] = {"iters": d["iters"], "broadcast_calls": d["broadcast_calls"],
             "sampler_use_flashinfer": d["sampler_use_flashinfer"], "components": summ}
res["stepcost_in_process"] = ab
print(json.dumps(res, indent=1))
