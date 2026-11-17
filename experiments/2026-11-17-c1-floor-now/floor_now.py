#!/usr/bin/env python3
"""The byte floor of today's single-stream step, with the model of
experiments/2026-10-26-c1-100tps (model.py, floor.json) and a measured single stream.

usage: floor_now.py C1.json [C1.json ...]

    step = overhead + (dense + drafter_routed + 40 * N(live) * expert) / rate + dead * row

Everything but the step, its tokens and its rows is that record's: the bytes a rank reads
(floor.json), the rate (FC1's marginal rate at six rows, 266.6 GB/s), N(r) from DeepSeek's
reference routes and a dead row's 0.37 ms. That record solved the live rows from a traced
overhead; here the live rows are the anchor and the drafts a step verifies (the dead-row
cut is on the device now), and the overhead is what is left. The rows of the six-row block
that are not live are dead rows at most (``dead_all``) or cost nothing (``dead_none``).
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "..", "2026-10-26-c1-100tps")
spec = importlib.util.spec_from_file_location("model", os.path.join(REC, "model.py"))
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

BLOCK = 5


def main(paths):
    floor = json.load(open(os.path.join(REC, "floor.json")))
    N = model.distinct(model.ROUTES)
    g = floor["windows"]["greedy"]["measured"]
    rate = g["rate_gbs (FC1 marginal, six rows)"] * 1e6  # bytes a ms
    expert = floor["fc1_bytes_per_expert"] + floor["fc2_bytes_per_expert"]
    dense = sum(floor["dense_bytes"].values())
    drafter_routed = model.DRAFTER_LAYERS * g["experts_per_drafter_layer"] * expert
    runs = [json.load(open(p))["summary"] for p in paths]
    mean = lambda f: sum(f(s) for s in runs) / len(runs)
    tps = mean(lambda s: s["output_tps"]["mean"])
    steps_per_s = mean(lambda s: s["server"]["steps_per_s"]["mean"])
    tokens = mean(lambda s: s["server"]["output_tokens_per_step"]["mean"])
    drafts = mean(lambda s: s["server"]["draft_tokens_per_draft"]["mean"])
    step = 1e3 / steps_per_s
    live = 1 + drafts

    def stream_ms(rows):
        return (dense + drafter_routed + model.LAYERS * model.interp(N, rows) * expert) / rate

    out = {"inputs": {"runs": [os.path.relpath(p, HERE) for p in paths], "tok_s": round(tps, 2), "step_ms": round(step, 2),
                      "tokens_per_step": round(tokens, 4), "verified_drafts_per_step": round(drafts, 3)},
           "rate_gbs": rate / 1e6, "live_rows": round(live, 2), "experts_per_layer": round(model.interp(N, live), 2),
           "bytes_mb": {"dense and heads": round(dense / 1e6), "drafter routed": round(drafter_routed / 1e6),
                        "target routed": round(model.LAYERS * model.interp(N, live) * expert / 1e6)},
           "floor_ms": round(stream_ms(live), 2), "floor_share_of_step": round(stream_ms(live) / step, 3),
           "live_row_ms": round(stream_ms(live + 0.5) - stream_ms(live - 0.5), 2),
           "record_1022": {"step_ms": 30.55, "floor_ms": 21.41, "overhead_ms": 8.70, "tok_s": 74.3}, "cases": {}}
    for name, dead in (("dead_none", 0.0), ("dead_all", BLOCK - drafts)):
        dead_ms = dead * model.DEAD_ROW_MS
        overhead = step - stream_ms(live) - dead_ms
        case = {"dead_rows": round(dead, 2), "dead_rows_ms": round(dead_ms, 2), "overhead_ms": round(overhead, 2),
                "overhead_share_of_step": round(overhead / step, 3), "levers": {}}
        for label, cut in (("all of the overhead gone", 1.0), ("two thirds", 2 / 3), ("half", 0.5), ("a quarter", 0.25)):
            s = step - cut * overhead
            case["levers"][label] = {"step_ms": round(s, 2), "tok_s": round(tps * step / s, 1), "gain_pct": round(100 * (step / s - 1), 1),
                                     "tokens_per_step_for_100": round(tokens * 100 / (tps * step / s), 3)}
        case["tokens_per_step_for_100_today"] = round(tokens * 100 / tps, 3)
        out["cases"][name] = case
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
