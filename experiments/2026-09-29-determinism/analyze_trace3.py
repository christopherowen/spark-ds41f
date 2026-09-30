#!/usr/bin/env python3
"""Compare a tagged target's rows across runs by what they compute: position and input token.

usage: analyze_trace3.py OUT_DIR PROMPT [--chain N]   (runs from trace_mixes.py on an attn-exact2
                                                       arm, which records row positions and tokens)

Steps cannot be paired by number once two runs accept drafts differently, so
each target row is keyed by (position, input token). A row is comparable only
if its whole prefix is the accepted text: prompt rows always, and a generated
row when every earlier row of its verify block was fed the token the run
finally generated there. For every pair of runs (each run against the first
solo run and the first run of its mix), rows comparable in both, with the same
key and a prefix both runs share (before their first differing output token),
are compared record by record in execution order (main model only): attention,
MoE input, router logits, shared expert, outputs. Reports output identity,
comparable rows, and the first (lowest position, earliest record) difference
with the records that differ after it in that row. A row computed twice in one
run (verified again after a truncated acceptance) is also checked against
itself.
"""
import glob
import json
import os
import re
import sys

import torch

OUT, PROMPT = sys.argv[1], sys.argv[2]
CHAIN = int(sys.argv[sys.argv.index("--chain") + 1]) if "--chain" in sys.argv else 8
SHARED = {0: "mlp_input", 1: "gate_up", 2: "act", 3: "down"}
ATTN = {10: "attn_input", 11: "q_latent", 12: "kv_latent", 13: "q_rotated", 14: "compressor_latent",
        15: "index_key", 16: "index_weights", 17: "selected_sum", 18: "selected_order", 19: "selected_len",
        20: "window_len", 21: "attn_output", 22: "o_proj", 24: "q_proj", 28: "q_raw", 29: "kv_raw",
        30: "router_logits"}
ORDER = (10, 28, 29, 11, 12, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def latest(node_dir, kind):
    files = sorted(glob.glob(os.path.join(node_dir, f"rank*-{kind}-*.pt")),
                   key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
    return torch.load(files[-1]) if files else None


def chronological(log):
    rows, count, capacity = log["rows"], log["count"], log["capacity"]
    if count > capacity:
        k = count % capacity
        rows = torch.cat([rows[k:], rows[:k]])
    return rows


def steps_of(log, marker_step_col, value_col, tag_col=None):
    out, current = {}, None
    if log is None:
        return out
    m = log["max_rows"]
    for r in chronological(log):
        if r[0].item() == -1.0:
            current = out.setdefault(int(r[marker_step_col].item()), [])
            continue
        if current is None:
            continue
        n = int(r[1].item())
        tag = int(r[tag_col].item()) if tag_col is not None else None
        if tag_col is None:
            sums = torch.stack([r[value_col + i * m: value_col + i * m + n] for i in range(3)], dim=1)
        else:
            sums = r[value_col: value_col + n].unsqueeze(1)
        current.append((int(r[0].item()), tag, n, sums))
    return out


def main_layer(name):
    if not name.startswith("language_model"):
        return None
    match = re.search(r"layers\.(\d+)", name)
    return int(match.group(1)) if match else None


def load_run(run_dir, node="dgx1"):
    target = json.load(open(os.path.join(run_dir, "target.json")))
    d = os.path.join(run_dir, node)
    runner, tags, attn, sched = (latest(d, k) for k in ("runner", "tags", "attn", "schedule"))
    host = json.load(open(sorted(glob.glob(os.path.join(d, "rank*-schedule-host-*.json")),
                                 key=lambda f: int(f.rsplit("-", 1)[1][:-5]))[-1]))
    host = {h["step"]: h for h in host}
    m, reqs = sched["max_rows"], sched["schedule_reqs"]
    dev = {int(row[0].item()): row for row in chronological(sched)}
    run_steps, tag_steps = steps_of(runner, 2, 2), steps_of(tags, 3, 3, tag_col=2)
    attn_steps = steps_of(attn, 3, 3, tag_col=2)
    prompt_len = target["prompt_tokens"]
    rows = []  # (position, token, comparable, {label: value}, step)
    for step in sorted(run_steps):
        h = host.get(step)
        if h is None or step not in dev:
            continue
        idx = next((i for i, rid in enumerate(h["req_ids"]) if target["tag"] in rid), None)
        if idx is None:
            continue
        row = dev[step]
        qsl = row[3: 3 + reqs]
        a, b = int(qsl[idx].item()), int(qsl[idx + 1].item())
        base = 3 + reqs + m
        if b > m:
            continue
        pos = [int(v) for v in row[base + a: base + b].tolist()]
        tok = [int(v) for v in row[base + m + a: base + m + b].tolist()]
        records = [{} for _ in range(b - a)]
        for slot, tag, n, sums in attn_steps.get(step, []):
            layer = main_layer(attn["names"][slot])
            if layer is None or b > n or tag not in ATTN:
                continue
            key = (layer, 1, 1) if tag == 30 else (layer, 0, ORDER.index(tag)) if tag in ORDER else None
            if key is None:
                continue
            for i in range(b - a):
                records[i][(key, ATTN[tag])] = float(sums[a + i, 0])
        for slot, _, n, sums in run_steps[step]:
            layer = main_layer(runner["names"][slot])
            if layer is None or b > n:
                continue
            for i in range(b - a):
                records[i][((layer, 1, 0), "moe_input")] = float(sums[a + i, 0])
                records[i][((layer, 3, 0), "shared_out")] = float(sums[a + i, 1])
                records[i][((layer, 4, 0), "routed_out")] = float(sums[a + i, 2])
        for slot, tag, n, sums in tag_steps.get(step, []):
            layer = main_layer(tags["names"][slot])
            if layer is None or b > n or tag not in SHARED:
                continue
            for i in range(b - a):
                records[i][((layer, 2, tag), SHARED[tag])] = float(sums[a + i, 0])
        generated = target["tokens_ids"] if "tokens_ids" in target else None
        for i in range(b - a):
            ok = True
            for j in range(i):  # earlier rows of this block fed the accepted token?
                if pos[j] >= prompt_len:
                    g = pos[j] - prompt_len
                    if generated is None or g >= len(generated) or generated[g] != tok[j]:
                        ok = False
                        break
            rows.append((pos[i], tok[i], ok, records[i], step))
    return target, rows


def first_difference(ra, rb):
    for key in sorted(set(ra) & set(rb)):
        if ra[key] != rb[key]:
            chain = [f"L{k[0][0]}:{k[1]}" for k in sorted(set(ra) & set(rb)) if ra[k] != rb[k]][:CHAIN]
            return key, chain
    return None, []


def compare(a, b):
    (ta, rowsa), (tb, rowsb) = a, b
    diff = next((i for i, (x, y) in enumerate(zip(ta["tokens"], tb["tokens"])) if x != y), None)
    same = diff is None and len(ta["tokens"]) == len(tb["tokens"])
    limit = ta["prompt_tokens"] + (diff if diff is not None else len(ta["tokens"]))
    first = {}
    for pos, tok, ok, rec, step in rowsb:
        if ok and pos < limit:
            first.setdefault((pos, tok), (rec, step))
    compared, found = 0, None
    for pos, tok, ok, rec, step in sorted(rowsa, key=lambda r: (r[0], r[4])):
        if not ok or pos >= limit or (pos, tok) not in first:
            continue
        other, step_b = first[(pos, tok)]
        compared += 1
        key, chain = first_difference(rec, other)
        if key is not None:
            found = (f"position {pos} (token {tok}, steps {step} vs {step_b}): layer {key[0][0]} {key[1]} "
                     f"differs; chain: {' '.join(chain)}")
            break
    return {"outputs_equal": same, "first_token_diff": diff, "rows_compared": compared,
            "first_difference": found or "none over comparable rows"}


def self_check(rows):
    seen, repeats, bad = {}, 0, []
    for pos, tok, ok, rec, step in rows:
        if not ok:
            continue
        if (pos, tok) in seen:
            repeats += 1
            key, _ = first_difference(seen[(pos, tok)][0], rec)
            if key is not None:
                bad.append(f"position {pos} steps {seen[(pos, tok)][1]}/{step}: L{key[0][0]} {key[1]}")
        else:
            seen[(pos, tok)] = (rec, step)
    return repeats, bad


runs = {os.path.basename(d): load_run(d) for d in sorted(glob.glob(os.path.join(OUT, f"{PROMPT}-m*-r*")))}
if runs and "tokens_ids" not in next(iter(runs.values()))[0]:
    print("note: target.json has no token ids; generated rows after the first are not checked for prefix")
for name, (target, rows) in runs.items():
    repeats, bad = self_check(rows)
    print(json.dumps({"run": name, "rows": len(rows), "comparable": sum(r[2] for r in rows),
                      "recomputed_rows": repeats, "recomputed_mismatch": bad[:3]}))
ref = f"{PROMPT}-m0-r0"
for name in sorted(runs):
    if name == ref:
        continue
    mix_ref = name.rsplit("-r", 1)[0] + "-r0"
    for base in dict.fromkeys((ref, mix_ref)):
        if base != name and base in runs:
            print(json.dumps({"run": name, "against": base, **compare(runs[base], runs[name])}))
