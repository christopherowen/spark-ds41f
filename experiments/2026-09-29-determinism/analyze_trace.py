#!/usr/bin/env python3
"""First tensor where a tagged target's rows differ between two traced runs.

usage: analyze_trace.py OUT_DIR PROMPT      (runs from trace_mixes.py)

For every run of PROMPT, per rank: splits the runner and tag logs into steps at
the schedule markers, finds the target's rows in each step from the device
schedule (query_start_loc after draft reallocation) and the host request ids,
and keeps, per appearance of the target, its rows' sums at every MoE layer
(input, shared and routed output) and inside the shared expert (input,
gate_up, activation, down). Compares every run with the first run of mix 0 and
every repeat with the first run of its mix: output identity, whether the
target's schedule (rows per appearance, padded batch size) matched, and the
first appearance, layer and tensor where the target's rows differ while its
row count matches, per rank.
"""
import glob
import json
import os
import sys

import torch

OUT, PROMPT = sys.argv[1], sys.argv[2]
TAGS = {0: "mlp_input", 1: "gate_up", 2: "act", 3: "down"}


def latest(node_dir, kind):
    files = sorted(glob.glob(os.path.join(node_dir, f"rank*-{kind}-*.pt")),
                   key=lambda f: int(f.rsplit("-", 1)[1][:-3]))
    return torch.load(files[-1]) if files else None


def steps_of(log, marker_step_col, value_col, tag_col=None):
    """step -> list of (slot, tag, rows, row_sums tensor [rows, k]) in log order."""
    rows = log["rows"][: min(log["count"], log["capacity"])]
    m = log["max_rows"]
    out, current = {}, None
    for r in rows:
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


def load_run(run_dir):
    target = json.load(open(os.path.join(run_dir, "target.json")))
    ranks = {}
    for node in ("dgx1", "dgx2", "dgx3"):
        d = os.path.join(run_dir, node)
        runner, tags, sched = latest(d, "runner"), latest(d, "tags"), latest(d, "schedule")
        host_files = sorted(glob.glob(os.path.join(d, "rank*-schedule-host-*.json")),
                            key=lambda f: int(f.rsplit("-", 1)[1][:-5]))
        if runner is None or sched is None or not host_files:
            continue
        host = {h["step"]: h for h in json.load(open(host_files[-1]))}
        dev = {}
        for row in sched["rows"][: min(sched["count"], sched["capacity"])]:
            dev[int(row[0].item())] = row
        run_steps = steps_of(runner, 2, 2)
        tag_steps = steps_of(tags, 3, 3, tag_col=2) if tags is not None else {}
        names = runner["names"]
        tag_names = tags["names"] if tags is not None else []
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
            layers = []
            for slot, _, n, sums in run_steps[step]:
                if b <= n:
                    layers.append((names[slot], sums[a:b]))
            shared = []
            for slot, tag, n, sums in tag_steps.get(step, []):
                if tag in TAGS and b <= n:
                    shared.append((tag_names[slot] + ":" + TAGS[tag], sums[a:b, 0]))
            appearances.append({"step": step, "rows": b - a, "padded": h["padded"],
                                "batch_reqs": len(h["req_ids"]), "layers": layers, "shared": shared})
        ranks[node] = appearances
    return target, ranks


def compare(a, b):
    ta, ra = a
    tb, rb = b
    diff = next((i for i, (x, y) in enumerate(zip(ta["tokens"], tb["tokens"])) if x != y), None)
    same_out = diff is None and len(ta["tokens"]) == len(tb["tokens"])
    lp0 = abs(ta["logprobs"][0] - tb["logprobs"][0]) if ta["logprobs"] and tb["logprobs"] else None
    result = {"outputs_equal": same_out, "first_token_diff": diff, "first_logprob_diff": lp0}
    for node in sorted(set(ra) & set(rb)):
        xa, xb = ra[node], rb[node]
        sched_same = [(p["rows"], p["padded"]) for p in xa] == [(p["rows"], p["padded"]) for p in xb]
        first = None
        for k, (pa, pb) in enumerate(zip(xa, xb)):
            if pa["rows"] != pb["rows"]:
                first = f"appearance {k}: target rows {pa['rows']} vs {pb['rows']} (stopped)"
                break
            for (na, sa), (nb, sb) in zip(pa["layers"], pb["layers"]):
                if na != nb:
                    first = f"appearance {k}: layer order differs ({na} vs {nb})"
                    break
                for col, label in enumerate(("moe_input", "shared_out", "routed_out")):
                    if not torch.equal(sa[:, col], sb[:, col]):
                        rows = (sa[:, col] != sb[:, col]).nonzero().flatten().tolist()
                        first = (f"appearance {k} (rows {pa['rows']}, padded {pa['padded']} vs "
                                 f"{pb['padded']}, batch {pa['batch_reqs']} vs {pb['batch_reqs']} reqs): "
                                 f"{na} {label} differs on target rows {rows[:8]}")
                        break
                if first:
                    break
            if first:
                inner = next(((n1, (s1 != s2).nonzero().flatten().tolist()[:8])
                              for (n1, s1), (n2, s2) in zip(pa["shared"], pb["shared"])
                              if n1 == n2 and not torch.equal(s1, s2)), None)
                if inner:
                    first += f"; first shared-expert difference: {inner[0]} rows {inner[1]}"
                break
        result[node] = {"schedule_equal": sched_same, "appearances": min(len(xa), len(xb)),
                        "first_difference": first or "none over matching appearances"}
    return result


runs = {os.path.basename(d): load_run(d) for d in sorted(glob.glob(os.path.join(OUT, f"{PROMPT}-m*-r*")))}
names = sorted(runs)
ref = f"{PROMPT}-m0-r0"
for name in names:
    if name == ref:
        continue
    mix_ref = name.rsplit("-r", 1)[0] + "-r0"
    base = mix_ref if name != mix_ref else ref
    print(json.dumps({"run": name, "against": base, **compare(runs[base], runs[name])}))
