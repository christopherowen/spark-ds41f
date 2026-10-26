#!/usr/bin/env python3
"""Ceilings of the c1 levers in tok/s, TP4 and TP3, one lever at a time.

usage: levers.py [--bound runs/bound-2/output.txt] [--tables runs/layer-1] [--json levers.json]

Each lever saves some milliseconds of a single-stream step; its ceiling is the
release suite's tokens a step over the shortened step (spark-ds41f-1025 single-stream
suites, candidate 6: TP4 33.003 steps/s at 2.284 tokens a step, TP3 26.730 at 2.269).
Savings are per step, over 43 MoE layers (40 target, 3 drafter) or 40 target layers.
Every saving is non-byte time in #598's terms, never byte floor.

Measured inputs (dgx1, one GPU at each rank's shapes, the reference routes):
- the bound (stream_bound.py, run bound-2): per layer, next-candidate's routed chain
  against an ideal one-launch read of the same expert bytes, at 3 live rows (13.25
  distinct experts a layer, the suite's operating point is 13.8);
- the layer tables (layer_table.py, run layer-1): each kernel family's cost and its
  excess over its bytes at 266.6 GB/s.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = {"TP4": {"steps_per_s": 33.003, "tokens": 2.284}, "TP3": {"steps_per_s": 26.730, "tokens": 2.269}}
MOE_LAYERS, LAYERS = 43, 40


def ceiling(tp, saved_ms):
    s = SUITE[tp]
    step = 1000 / s["steps_per_s"]
    return {"saved_ms": round(saved_ms, 3), "tok_s": round(s["tokens"] / (step - saved_ms) * 1000, 1),
            "gain_percent": round(100 * (step / (step - saved_ms) - 1), 2)}


def bound(path):
    """(tp, live) -> the bound's result line."""
    out = {}
    for line in open(path):
        if line.startswith('{"tp"'):
            r = json.loads(line)
            out[(r["tp"], r["live"])] = r
    return out


def families(path):
    return {f["family"]: f for f in json.load(open(path))["families"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound", default=os.path.join(HERE, "runs", "bound-2", "output.txt"))
    ap.add_argument("--tables", default=os.path.join(HERE, "runs", "layer-1"))
    ap.add_argument("--json")
    args = ap.parse_args()
    b = bound(args.bound)
    t = {tp: families(os.path.join(args.tables, f"tp{n}-live3", "layer_table.json")) for tp, n in (("TP4", 4), ("TP3", 3))}

    def per_layer(tp, us, layers=MOE_LAYERS):
        return us * layers / 1000

    routed = {tp: b[(n, 3)]["bound_us_per_layer"]["chain_minus_read_one"] for tp, n in (("TP4", 4), ("TP3", 3))}
    boundary = {tp: b[(n, 3)]["bound_us_per_layer"]["read_one_minus_read_all"] for tp, n in (("TP4", 4), ("TP3", 3))}
    small = {tp: sum(t[tp][f]["cost_us"] for f in ("sparse MLA", "q norm and Q-B cast", "routed dispatch",
                                                    "routed SwiGLU and cast"))
             + t[tp]["mHC project and finalize"]["excess_us"] for tp in ("TP4", "TP3")}
    head = {tp: 2 * t[tp]["LM head"]["excess_us"] for tp in ("TP4", "TP3")}
    levers = [
        {"lever": "L2 prefetch timed against its consumers", "owner": "l2-prefetch-timing session (#615)",
         "evidence": "1.67 ms of consumer excess (moe-overlap, PDL records); its prototype measured -1.5% c1",
         "TP4": ceiling("TP4", 1.67), "TP3": None},
        {"lever": "Latency-bound small kernels (mHC non-byte, sparse MLA, q-norm and cast, dispatch, SwiGLU)",
         "owner": "fusion survey (#583)", "evidence": "layer tables: their cost, mHC's excess over its fn bytes",
         "us_per_layer": {tp: round(v, 1) for tp, v in small.items()},
         "TP4": ceiling("TP4", per_layer("TP4", small["TP4"], LAYERS)),
         "TP3": ceiling("TP3", per_layer("TP3", small["TP3"], LAYERS))},
        {"lever": "All-reduce: peer lateness and staging, at most the DRAM-idle time under collectives",
         "owner": "sparknet; cluster only (ring: dgx4 last in 39% of collectives, 10-03)",
         "evidence": "#598: 0.95 ms DRAM idle under collectives; IKET: the time is the peer wait",
         "TP4": ceiling("TP4", 0.95), "TP3": ceiling("TP3", 0.95)},
        {"lever": "Stream boundaries' ramps and tails filled with known traffic (the next window's prefetch)",
         "owner": "l2-prefetch-timing session", "evidence": "bound: one launch a layer is this much slower "
         "than one launch for all layers, for an ideal read of the same bytes",
         "us_per_layer": {tp: round(v, 1) for tp, v in boundary.items()},
         "TP4": ceiling("TP4", per_layer("TP4", boundary["TP4"])),
         "TP3": ceiling("TP3", per_layer("TP3", boundary["TP3"]))},
        {"lever": "Routed experts restructured: continuous FC1-FC2, persistent or stream GEMMs, deeper "
                  "stages, warp-specialized producer, fused SwiGLU",
         "owner": "this study", "evidence": "bound: the routed chain less an ideal one-launch read of the "
         "same bytes", "us_per_layer": {tp: round(v, 1) for tp, v in routed.items()},
         "TP4": ceiling("TP4", per_layer("TP4", routed["TP4"])),
         "TP3": ceiling("TP3", per_layer("TP3", routed["TP3"]))},
        {"lever": "LM heads at the attainable rate", "owner": "unclaimed",
         "evidence": "layer tables: the head's excess over its bytes, two heads a step",
         "TP4": ceiling("TP4", head["TP4"] / 1000), "TP3": ceiling("TP3", head["TP3"] / 1000)},
    ]
    result = {"suite": SUITE, "levers": levers}
    print("| lever | owner | TP4 ms | TP4 tok/s | TP3 ms | TP3 tok/s |")
    print("|---|---|---:|---:|---:|---:|")
    for lv in levers:
        t3 = lv["TP3"]
        print(f"| {lv['lever']} | {lv['owner']} | {lv['TP4']['saved_ms']} | {lv['TP4']['tok_s']} "
              f"(+{lv['TP4']['gain_percent']}%) | {t3['saved_ms'] if t3 else '-'} | "
              f"{str(t3['tok_s']) + ' (+' + str(t3['gain_percent']) + '%)' if t3 else '-'} |")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
