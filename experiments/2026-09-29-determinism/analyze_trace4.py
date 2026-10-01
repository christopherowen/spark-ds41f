#!/usr/bin/env python3
"""Compare traced requests row by row: same position and input token, exact fingerprints.

usage: analyze_trace4.py OUT_DIR [--group G] [--node dgx1|dgx2|dgx3] [--chain N] [--all-pairs]
       (runs from trace_mixes.py, c8_trace.py or scenario_trace.py on an attn-exact2/3/4 arm)

Every target.json under OUT_DIR is one traced request (its run directory holds
each rank's logs, possibly shared with other requests of the same run). Requests
are grouped by the prompt they share ("group" in target.json, else the run
directory's name before "-m"); in each group every request is compared with the
group's first (and, with --all-pairs, every pair).

A request's rows are found per step from the schedule log (the request's row
range, each row's position, input token and dead/padding flag). A record
covers the step's rows [offset, offset + rows): offsets are explicit in layout
2 (attn-exact4; sequence-parallel WO output covers only the rank's rows),
zero before. A record whose rows match neither the step's token count nor its
padded count, at offset zero, is not aligned with the batch: it is counted
and left out instead of being compared row for row.

A row is comparable if it is not dead and every earlier row of its step block
was fed the token the request finally generated there (prompt rows always).
Two requests' rows with the same (position, token), before their first
differing output token, are compared on the records both have (main model
only). Each row is labelled with its step: token count, padded count, whether
prefill sequence parallelism ran (tag 32, attn-exact4) and the attention
index mode (tag 23: decode, prefill, prefill_short; "graph" when a captured
step left none).
"""
import glob
import json
import os
import re
import struct
import sys
from collections import Counter

import torch

SHARED = {0: "mlp_input", 1: "gate_up", 2: "act", 3: "down"}
ATTN = {10: "attn_input", 11: "q_latent", 12: "kv_latent", 13: "q_rotated", 14: "compressor_latent",
        15: "index_key", 16: "index_weights", 17: "selected_sum", 18: "selected_order", 19: "selected_len",
        20: "window_len", 21: "attn_output", 22: "o_proj", 24: "q_proj", 28: "q_raw", 29: "kv_raw",
        30: "router_logits"}
ORDER = (10, 28, 29, 11, 12, 24, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
MODES = {0: "decode", 1: "prefill", 2: "prefill_short"}
PRIME = 16777213


def fingerprint1(value):
    """The fingerprint attn-exact2/3 stored for a one-value row (float32 bits, weight 40510)."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return float((bits % PRIME) * ((1 * 40503 + 7) % 1048573) % PRIME)


FINGERPRINTED = {fingerprint1(v): v for v in range(4)}


def arg(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


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


def steps_of(log, runner):
    """{step: [(slot, tag, rows, offset, values[rows, k])]} for one log, either layout."""
    out, current = {}, None
    if log is None:
        return out
    m, layout = log["max_rows"], log.get("layout", 1)
    step_col = 3 if layout == 2 or not runner else 2
    for r in chronological(log):
        if r[0].item() == -1.0:
            current = out.setdefault(int(r[step_col].item()), [])
            continue
        if current is None:
            continue
        n = int(r[1].item())
        if layout == 2:
            tag, offset, base = int(r[2].item()), int(r[3].item()), 4
        elif runner:
            tag, offset, base = -2, 0, 2
        else:
            tag, offset, base = int(r[2].item()), 0, 3
        if runner:
            values = torch.stack([r[base + i * m: base + i * m + n] for i in range(3)], dim=1)
        else:
            values = r[base: base + n].unsqueeze(1)
        current.append((int(r[0].item()), tag, n, offset, values))
    return out


def main_layer(name):
    if not name.startswith("language_model"):
        return None
    match = re.search(r"layers\.(\d+)", name)
    return int(match.group(1)) if match else None


class NodeLogs:
    """One rank's logs for one run directory (cached: several requests may share them)."""

    _cache: dict = {}

    @classmethod
    def load(cls, node_dir):
        key = os.path.realpath(node_dir)
        if key not in cls._cache:
            cls._cache[key] = cls(node_dir)
        return cls._cache[key]

    def __init__(self, node_dir, sp_min_rows=205):
        self.sched = latest(node_dir, "schedule")
        host_files = sorted(glob.glob(os.path.join(node_dir, "rank*-schedule-host-*.json")),
                            key=lambda f: int(f.rsplit("-", 1)[1][:-5]))
        self.host = {h["step"]: h for h in json.load(open(host_files[-1]))} if host_files else {}
        self.dev = {int(row[0].item()): row for row in chronological(self.sched)} if self.sched else {}
        self.sources = []  # (kind, log, steps)
        for kind in ("runner", "tags", "attn"):
            for suffix in ("", "_wide"):
                log = latest(node_dir, kind + suffix)
                if log is not None:
                    self.sources.append((kind, log, steps_of(log, kind == "runner")))
        self.labels = {}
        for kind, log, steps in self.sources:
            if kind != "attn":
                continue
            # Layout 2 stores these values raw; attn-exact2/3 stored fingerprints.
            raw = log.get("layout", 1) == 2
            decode = (lambda v: v) if raw else (lambda v: FINGERPRINTED.get(v, -1))
            for step, records in steps.items():
                label = self.labels.setdefault(step, {"mode": "graph", "sp": None})
                for slot, tag, n, offset, values in records:
                    if tag == 23 and n == 3 and label["mode"] == "graph":
                        label["mode"] = MODES.get(int(decode(float(values[1, 0]))), "?")
                    if tag == 32 and n == 3 and label["sp"] is None:
                        label["sp"] = int(decode(float(values[0, 0]))) == 1
                        if raw and label["sp"]:
                            label["local_rows"] = int(values[2, 0])
        for step, h in self.host.items():  # no SP record (attn-exact2/3): infer it from the rows
            label = self.labels.setdefault(step, {"mode": "graph", "sp": None})
            if label["sp"] is None and label["mode"] != "graph":
                label["sp"] = h["num_tokens"] >= sp_min_rows


def load_request(run_dir, target, node):
    """[(position, token, comparable, records, step, label)], and counters."""
    logs = NodeLogs.load(os.path.join(run_dir, node))
    m, reqs = logs.sched["max_rows"], logs.sched["schedule_reqs"]
    prompt_len = target["prompt_tokens"]
    generated = target.get("tokens_ids")
    rows, stats = [], Counter()
    for step in sorted(logs.host):
        h = logs.host[step]
        idx = next((i for i, rid in enumerate(h["req_ids"]) if target["tag"] in rid), None)
        if idx is None or step not in logs.dev:
            continue
        row = logs.dev[step]
        qsl = row[3: 3 + reqs]
        a, b = int(qsl[idx].item()), int(qsl[idx + 1].item())
        if b > m:
            stats["rows_beyond_schedule_log"] += b - a
            continue
        base = 3 + reqs + m
        pos = [int(v) for v in row[base + a: base + b].tolist()]
        tok = [int(v) for v in row[base + m + a: base + m + b].tolist()]
        dead = [bool(v) for v in row[3 + reqs + a: 3 + reqs + b].tolist()]
        num_tokens, padded = h["num_tokens"], h["padded"]
        label = dict(logs.labels.get(step, {"mode": "graph", "sp": None}), rows=num_tokens, padded=padded)
        local_rows = label.pop("local_rows", None)  # rank 0's SP-local records start at offset 0
        records = [{} for _ in range(b - a)]
        for kind, log, steps in logs.sources:
            for slot, tag, n, offset, values in steps.get(step, []):
                layer = main_layer(log["names"][slot])
                if layer is None:
                    continue
                if kind == "attn":
                    if tag not in ATTN:
                        continue
                    key = (layer, 1, 1) if tag == 30 else (layer, 0, ORDER.index(tag))
                    names = [ATTN[tag]]
                    keys = [key]
                elif kind == "runner":
                    keys = [(layer, 1, 0), (layer, 3, 0), (layer, 4, 0)]
                    names = ["moe_input", "shared_out", "routed_out"]
                else:
                    if tag not in SHARED:
                        continue
                    keys, names = [(layer, 2, tag)], [SHARED[tag]]
                if offset == 0 and n not in (num_tokens, padded, local_rows):
                    stats["unaligned_records"] += 1
                    continue
                if offset > 0 and offset + n > max(padded, -(-num_tokens // 3) * 3):
                    stats["unaligned_records"] += 1
                    continue
                if offset > 0:
                    stats["offset_records"] += 1
                for i in range(b - a):
                    j = a + i - offset
                    if 0 <= j < n:
                        for c, (key, name) in enumerate(zip(keys, names)):
                            records[i][(key, name)] = float(values[j, c])
        prefix_ok = True
        for i in range(b - a):
            ok = not dead[i] and bool(records[i]) and prefix_ok
            stats["dead_rows"] += dead[i]
            rows.append((pos[i], tok[i], ok, records[i], step, label))
            if pos[i] >= prompt_len:
                g = pos[i] - prompt_len
                if generated is None or g >= len(generated) or generated[g] != tok[i]:
                    prefix_ok = False
    return rows, stats


def describe(label):
    sp = {True: " SP", False: "", None: ""}[label.get("sp")]
    return f"{label['rows']} rows (padded {label['padded']}{sp}, {label['mode']})"


def compare(a, b, chain):
    (ta, rowsa), (tb, rowsb) = a, b
    diff = next((i for i, (x, y) in enumerate(zip(ta["tokens_ids"], tb["tokens_ids"])) if x != y), None)
    same = diff is None and len(ta["tokens_ids"]) == len(tb["tokens_ids"])
    limit = ta["prompt_tokens"] + (diff if diff is not None else len(ta["tokens_ids"]))
    other = {}
    for pos, tok, ok, rec, step, label in rowsb:
        if ok and pos < limit:
            other.setdefault((pos, tok), (rec, step, label))
    compared, differing, first, kinds, labels = 0, 0, None, Counter(), Counter()
    for pos, tok, ok, rec, step, label in sorted(rowsa, key=lambda r: (r[0], r[4])):
        if not ok or pos >= limit or (pos, tok) not in other:
            continue
        rec_b, step_b, label_b = other[(pos, tok)]
        common = sorted(set(rec) & set(rec_b))
        bad = [k for k in common if rec[k] != rec_b[k]]
        compared += 1
        if not bad:
            continue
        differing += 1
        kinds[f"L{bad[0][0][0]}:{bad[0][1]}"] += 1
        labels[f"{describe(label)} vs {describe(label_b)}"] += 1
        if first is None:
            first = {"position": pos, "token": tok, "steps": [step, step_b],
                     "record": f"L{bad[0][0][0]}:{bad[0][1]}",
                     "chain": [f"L{k[0][0]}:{k[1]}" for k in bad[:chain]],
                     "records_compared": len(common), "records_differing": len(bad),
                     "label": describe(label), "label_other": describe(label_b)}
    return {"outputs_equal": same, "first_token_diff": diff, "rows_compared": compared,
            "rows_differing": differing, "first_difference": first,
            "first_records": dict(kinds.most_common(8)), "step_pairs": dict(labels.most_common(8))}


def self_check(rows):
    seen, repeats, bad = {}, 0, []
    for pos, tok, ok, rec, step, label in rows:
        if not ok:
            continue
        if (pos, tok) in seen:
            repeats += 1
            other = seen[(pos, tok)]
            diff = [k for k in sorted(set(rec) & set(other[0])) if rec[k] != other[0][k]]
            if diff:
                bad.append(f"position {pos} steps {other[1]}/{step}: L{diff[0][0][0]} {diff[0][1]}")
        else:
            seen[(pos, tok)] = (rec, step)
    return repeats, bad


def discover(out):
    requests = []
    for f in sorted(glob.glob(os.path.join(out, "**", "target.json"), recursive=True)):
        target = json.load(open(f))
        run_dir = os.path.dirname(f)
        name = os.path.basename(run_dir)
        target.setdefault("group", name.rsplit("-m", 1)[0] if "-m" in name else name)
        requests.append((name, run_dir, target))
    return requests


def main(argv):
    out = argv[1]
    node, chain = arg(argv, "--node", "dgx1"), int(arg(argv, "--chain", "8"))
    only = arg(argv, "--group", None)
    results = []
    groups = {}
    for name, run_dir, target in discover(out):
        if only is None or target["group"] == only:
            groups.setdefault(target["group"], []).append((name, run_dir, target))
    for group, members in groups.items():
        loaded = {}
        for name, run_dir, target in members:
            rows, stats = load_request(run_dir, target, node)
            loaded[name] = (target, rows)
            repeats, bad = self_check(rows)
            results.append({"group": group, "request": name, "rows": len(rows),
                            "comparable": sum(r[2] for r in rows), "recomputed_rows": repeats,
                            "recomputed_mismatch": bad[:3], **stats})
        names = [n for n, _, _ in members]
        pairs = ([(x, y) for i, x in enumerate(names) for y in names[i + 1:]] if "--all-pairs" in argv
                 else [(names[0], y) for y in names[1:]])
        for x, y in pairs:
            results.append({"group": group, "request": y, "against": x, **compare(loaded[x], loaded[y], chain)})
    return results


if __name__ == "__main__":
    for result in main(sys.argv):
        print(json.dumps(result))
