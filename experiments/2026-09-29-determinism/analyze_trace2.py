#!/usr/bin/env python3
"""First tensor where a tagged target's rows differ between two traced runs, attention included.

usage: analyze_trace2.py OUT_DIR PROMPT [--main | --drafter]   (runs from trace_mixes.py on an
                                                                 attn-trace arm)

analyze_trace.py, extended with the attn-trace overlay's attention log. For
every run of PROMPT and rank, the target's rows are found in each step from
the device schedule and host request ids; per appearance, the records are put
in execution order within each layer: attention (input, normalized q and kv
latents, rotated q, compressor latent and index key on emitted rows, index
head weights, selected positions and their order, lengths, attention output,
projected output), then the MoE (input, router logits, shared-expert
internals, shared and routed output); main-model layers first, then the drafter. Every run is
compared with the first run of mix 0 and every repeat with the first run of
its mix: output identity, the target's schedule, and the first record where
its rows differ while its row count matches, with the indexer's score width
and mode in that step (tag 23; it depends on the batch's longest sequence).
Each run's schedule is summarized: the target's rows, padded batch and
request count per appearance. --main compares only the main model's layers,
--drafter only the drafter's, so a drafter difference does not hide a later
main-model one.
"""
import glob
import hashlib
import json
import os
import re
import sys

import torch

OUT, PROMPT = sys.argv[1], sys.argv[2]
ONLY = 0 if "--main" in sys.argv else 1 if "--drafter" in sys.argv else None
SHARED = {0: "mlp_input", 1: "gate_up", 2: "act", 3: "down"}
ATTN = {10: "attn_input", 11: "q_latent", 12: "kv_latent", 13: "q_rotated", 14: "compressor_latent",
        15: "index_key", 16: "index_weights", 17: "selected_sum", 18: "selected_order", 19: "selected_len",
        20: "window_len", 21: "attn_output", 22: "o_proj", 24: "q_proj", 28: "q_raw", 29: "kv_raw",
        30: "router_logits"}
# Execution order within a layer: attention records, then the MoE input, the
# router logits (tag 30, recorded by the MoE runner), the shared expert, outputs.
ATTN_ORDER = (10, 28, 29, 11, 12, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
MODES = {0: "decode", 1: "prefill", 2: "prefill_short"}


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
    """step -> list of (slot, tag, rows, row values [rows, k]) in log order."""
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


def layer_key(name):
    match = re.search(r"layers\.(\d+)", name)
    return (0 if name.startswith("language_model") else 1, int(match.group(1)) if match else -1)


def load_run(run_dir):
    target = json.load(open(os.path.join(run_dir, "target.json")))
    ranks = {}
    for node in ("dgx1", "dgx2", "dgx3"):
        d = os.path.join(run_dir, node)
        runner, tags, attn, sched = (latest(d, k) for k in ("runner", "tags", "attn", "schedule"))
        host_files = sorted(glob.glob(os.path.join(d, "rank*-schedule-host-*.json")),
                            key=lambda f: int(f.rsplit("-", 1)[1][:-5]))
        if runner is None or sched is None or not host_files:
            continue
        host = {h["step"]: h for h in json.load(open(host_files[-1]))}
        dev = {int(row[0].item()): row for row in chronological(sched)}
        run_steps = steps_of(runner, 2, 2)
        tag_steps = steps_of(tags, 3, 3, tag_col=2)
        attn_steps = steps_of(attn, 3, 3, tag_col=2)
        names = runner["names"]
        tag_names = tags["names"] if tags is not None else []
        attn_names = attn["names"] if attn is not None else []
        appearances = []
        for step in sorted(run_steps):
            h = host.get(step)
            if h is None or step not in dev:
                continue
            idx = next((i for i, rid in enumerate(h["req_ids"]) if target["tag"] in rid), None)
            if idx is None:
                continue
            qsl = dev[step][3: 3 + 17]
            a, b = int(qsl[idx].item()), int(qsl[idx + 1].item())
            items, widths = [], {}
            for slot, tag, n, sums in attn_steps.get(step, []):
                name = attn_names[slot]
                if tag == 23:
                    v = sums[:3, 0].tolist()
                    widths[layer_key(name)] = (int(v[0]), MODES.get(int(v[1]), v[1]), int(v[2]))
                elif tag == 30 and b <= n:
                    items.append(((*layer_key(name), 1, 1), f"{name}:{ATTN[tag]}", sums[a:b, 0]))
                elif tag in ATTN and b <= n:
                    items.append(((*layer_key(name), 0, ATTN_ORDER.index(tag)), f"{name}:{ATTN[tag]}",
                                  sums[a:b, 0]))
            for slot, _, n, sums in run_steps[step]:
                if b <= n:
                    name = names[slot]
                    items.append(((*layer_key(name), 1, 0), f"{name} moe_input", sums[a:b, 0]))
                    items.append(((*layer_key(name), 3, 0), f"{name} shared_out", sums[a:b, 1]))
                    items.append(((*layer_key(name), 4, 0), f"{name} routed_out", sums[a:b, 2]))
            for slot, tag, n, sums in tag_steps.get(step, []):
                if tag in SHARED and b <= n:
                    name = tag_names[slot]
                    items.append(((*layer_key(name), 2, tag), f"{name}:{SHARED[tag]}", sums[a:b, 0]))
            items.sort(key=lambda item: item[0])
            seen = {}
            for i, (key, label, s) in enumerate(items):  # a module may record twice in a step
                seen[label] = seen.get(label, 0) + 1
                if seen[label] > 1:
                    items[i] = (key, f"{label}#{seen[label]}", s)
            appearances.append({"step": step, "rows": b - a, "offset": a, "padded": h["padded"],
                                "batch_reqs": len(h["req_ids"]), "items": items, "widths": widths,
                                "recorded": any(label.endswith("moe_input") for _, label, _ in items)})
        ranks[node] = appearances
    return target, ranks


def schedule(appearances):
    seq = [(p["rows"], p["padded"], p["batch_reqs"]) for p in appearances]
    digest = hashlib.sha256(json.dumps(seq).encode()).hexdigest()[:10]
    head = " ".join(f"{r}/{p}({q})" for r, p, q in seq[:6])
    return {"appearances": len(seq), "digest": digest, "first": head,
            "padded_sizes": sorted({p for _, p, _ in seq})}


def compare(a, b):
    ta, ra = a
    tb, rb = b
    diff = next((i for i, (x, y) in enumerate(zip(ta["tokens"], tb["tokens"])) if x != y), None)
    same_out = diff is None and len(ta["tokens"]) == len(tb["tokens"])
    result = {"outputs_equal": same_out, "first_token_diff": diff}
    for node in sorted(set(ra) & set(rb)):
        xa, xb = ra[node], rb[node]
        sched_same = [(p["rows"], p["padded"]) for p in xa] == [(p["rows"], p["padded"]) for p in xb]
        first = None
        for k, (pa, pb) in enumerate(zip(xa, xb)):
            if pa["rows"] != pb["rows"]:
                first = f"appearance {k}: target rows {pa['rows']} vs {pb['rows']} (stopped)"
                break
            if not (pa["recorded"] and pb["recorded"]):
                first = (f"appearance {k}: batch above the log's row limit (padded {pa['padded']} vs "
                         f"{pb['padded']}), not recorded (stopped)")
                break
            other = {label: s for _, label, s in pb["items"]}
            for key, label, sa in pa["items"]:
                sb = other.get(label)
                if sb is None or (ONLY is not None and key[0] != ONLY):
                    continue
                if not torch.equal(sa, sb):
                    rows = (sa != sb).nonzero().flatten().tolist()
                    lk = key[:2]
                    first = (f"appearance {k} (rows {pa['rows']} at offset {pa['offset']} vs {pb['offset']}, "
                             f"padded {pa['padded']} vs {pb['padded']}, batch {pa['batch_reqs']} vs "
                             f"{pb['batch_reqs']} reqs): {label} differs on target rows {rows[:8]}"
                             f"{' of ' + str(len(sa)) if len(rows) > 8 else ''}; indexer width/mode/chunk "
                             f"there {pa['widths'].get(lk)} vs {pb['widths'].get(lk)}")
                    break
            if first:
                break
        result[node] = {"schedule_equal": sched_same, "appearances": min(len(xa), len(xb)),
                        "first_difference": first or "none over matching appearances"}
    return result


runs = {os.path.basename(d): load_run(d) for d in sorted(glob.glob(os.path.join(OUT, f"{PROMPT}-m*-r*")))}
names = sorted(runs)
ref = f"{PROMPT}-m0-r0"
for name in names:
    target, ranks = runs[name]
    first_rank = ranks.get("dgx1") or next(iter(ranks.values()), [])
    print(json.dumps({"run": name, "schedule": schedule(first_rank)}))
for name in names:
    if name == ref:
        continue
    mix_ref = name.rsplit("-r", 1)[0] + "-r0"
    base = mix_ref if name != mix_ref else ref
    print(json.dumps({"run": name, "against": base, **compare(runs[base], runs[name])}))
