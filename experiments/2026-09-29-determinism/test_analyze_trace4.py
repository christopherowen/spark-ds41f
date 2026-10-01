#!/usr/bin/env python3
"""Self-test of analyze_trace4.py on logs written by the real attn-exact4 checksum_debug.py (CPU).

usage: CHECKSUM_DEBUG=/path/attn-exact4/checksum_debug.py python3 test_analyze_trace4.py

Two runs of one request (6 prompt tokens, then a three-row verify block):
run A alone; run B behind a background request's two rows, with WO's output
recorded as a sequence-parallel rank would (rows 4-7 of the step, offset 4),
one record whose rows do not match the step (must be counted, not compared),
a dead verification row (must be excluded) and one planted difference
(layer 1's router logits at prompt position 4). Every other row is built from
its (layer, tag, position, token) alone, so equal rows have equal fingerprints.
"""
import importlib.util
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS, WIDTH = 3, 16
GENERATED = [100, 101, 102, 103, 104]
PROMPT = [10, 11, 12, 13, 14, 15]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Owner:
    def __init__(self, prefix):
        self.prefix = prefix


def values(layer, tag, positions, tokens, plant=None):
    out = []
    for p, t in zip(positions, tokens):
        g = torch.Generator().manual_seed(layer * 1_000_003 + tag * 10_007 + p * 101 + t)
        row = torch.randn(WIDTH, generator=g).to(torch.bfloat16)
        if plant == (layer, tag, p):
            row[3] += 0.5
        out.append(row)
    return torch.stack(out)


def run(directory, steps, plant=None, sp_local=None, unaligned=False):
    os.environ.update(SPARK3_MOE_CHECKSUM_DIR=directory, SPARK3_MOE_CHECKSUM_ROWS="4",
                      SPARK3_MOE_CHECKSUM_WIDE_ROWS="16", SPARK3_MOE_CHECKSUM_WIDE_CAPACITY="256",
                      SPARK3_MOE_CHECKSUM_SCHEDULE_CAPACITY="16")
    cd = load("checksum_debug_test", os.environ["CHECKSUM_DEBUG"])
    cd._Log.row_sums = lambda self, x, out: cd._fingerprint_torch(x.reshape(x.shape[0], -1), out)
    cd.threading.Thread = lambda *a, **k: type("T", (), {"start": lambda self: None})()
    attn = [Owner(f"language_model.model.layers.{L}.self_attn") for L in range(LAYERS)]
    moe = [Owner(f"language_model.model.layers.{L}.mlp.experts") for L in range(LAYERS)]
    shared = [Owner(f"language_model.model.layers.{L}.mlp.shared_experts") for L in range(LAYERS)]
    for step, requests in enumerate(steps):
        req_ids = [r for r, _, _, _ in requests]
        positions = [p for _, ps, _, _ in requests for p in ps]
        tokens = [t for _, _, ts, _ in requests for t in ts]
        dead = [d for _, _, _, ds in requests for d in ds]
        qsl = [0]
        for _, ps, _, _ in requests:
            qsl.append(qsl[-1] + len(ps))
        n = len(positions)
        cd.record_schedule(req_ids, n, n, torch.tensor(qsl), len(requests), torch.tensor(dead, dtype=torch.float32),
                           torch.tensor(positions), torch.tensor(tokens))
        for L in range(LAYERS):
            for tag in (10, 21, 30):
                cd.record_attn(attn[L], tag, values(L, tag, positions, tokens, plant))
            if sp_local and step == 0:
                start, local = sp_local
                cd.record_attn(attn[L], 22, values(L, 22, positions[start:start + local],
                                                   tokens[start:start + local]), start)
            else:
                cd.record_attn(attn[L], 22, values(L, 22, positions, tokens))
            if unaligned and step == 0 and L == 0:
                cd.record_attn(attn[L], 11, values(L, 11, positions[:-1], tokens[:-1]))
            if step == 0 and L == 0:  # step labels: prefill index mode; SP in run B
                cd.record_attn_values(attn[L], 23, torch.tensor([-1.0, 1.0, 256.0]))
                cd.record_attn_values(attn[L], 32, torch.tensor([float(bool(sp_local)), 0.0, 0.0]))
            x = values(L, 40, positions, tokens)
            cd.record(moe[L], x, values(L, 41, positions, tokens), values(L, 42, positions, tokens))
            cd.record_tag(shared[L], 0, values(L, 43, positions, tokens))
    os.makedirs(directory, exist_ok=True)
    for log in cd._logs.values():
        count = int(log.count.item())
        torch.save({"count": count, "capacity": log.capacity, "rows": log.rows[: min(count, log.capacity)],
                    "names": list(log.names), "max_rows": log.max_rows, "layout": cd._LAYOUT,
                    "schedule_reqs": cd._SCHEDULE_REQS},
                   os.path.join(directory, f"rank0-{log.name}-{count}.pt"))
    with open(os.path.join(directory, f"rank0-schedule-host-{int(cd._logs['schedule'].count)}.json"), "w") as f:
        json.dump(list(cd._schedule_host), f)


def main():
    analyze = load("analyze_trace4", os.path.join(HERE, "analyze_trace4.py"))
    with tempfile.TemporaryDirectory() as out:
        target = "chatcmpl-trace-t-m{}-r0-x"
        prompt_rows = (list(range(6)), PROMPT, [0] * 6)
        verify = ([6, 7, 8], [100, 101, 999], [0, 0, 0])
        run(os.path.join(out, "t-m0-r0", "dgx1"),
            [[(target.format(0), *prompt_rows)], [(target.format(0), *verify)]])
        run(os.path.join(out, "t-m1-r0", "dgx1"),
            [[("bg", [50, 51], [7, 8], [0, 0]), (target.format(1), *prompt_rows)],
             [("bg", [52], [9], [0]), (target.format(1), [6, 7, 8], [100, 101, 999], [0, 1, 0])]],
            plant=(1, 30, 4), sp_local=(4, 4), unaligned=True)
        for m in (0, 1):
            json.dump({"tag": f"trace-t-m{m}-r0", "prompt_tokens": 6, "tokens_ids": GENERATED,
                       "tokens": [f"token_id:{t}" for t in GENERATED]},
                      open(os.path.join(out, f"t-m{m}-r0", "target.json"), "w"))
        results = analyze.main(["analyze_trace4.py", out, "--node", "dgx1"])
    by = {(r["request"], "against" in r): r for r in results}
    a, b, pair = by[("t-m0-r0", False)], by[("t-m1-r0", False)], by[("t-m1-r0", True)]
    checks = {
        "A: 9 comparable rows": a["comparable"] == 9,
        "B: dead row excluded (8 comparable)": b["comparable"] == 8 and b["dead_rows"] == 1,
        "B: misaligned record counted": b.get("unaligned_records") == 1,
        "B: offset records used": b.get("offset_records") == LAYERS,
        "8 rows compared": pair["rows_compared"] == 8,
        "exactly the planted row differs": pair["rows_differing"] == 1,
        "first difference at position 4, L1 router logits": pair["first_difference"] is not None
        and pair["first_difference"]["position"] == 4 and pair["first_difference"]["record"] == "L1:router_logits"
        and pair["first_difference"]["records_differing"] == 1,
        "WO compared at its offset rows without a false difference":
            pair["first_difference"]["chain"] == ["L1:router_logits"],
        "step labels (prefill alone against SP prefill)":
            pair["first_difference"]["label"] == "6 rows (padded 6, prefill)"
            and pair["first_difference"]["label_other"] == "8 rows (padded 8 SP, prefill)",
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    if not all(checks.values()):
        print(json.dumps(results, indent=1))
        sys.exit(1)


if __name__ == "__main__":
    main()
