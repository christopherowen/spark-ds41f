#!/usr/bin/env python3
"""Ceilings of the c1 levers in tok/s, TP4 and TP3, one lever at a time.

usage: levers.py [--chain routed_chain.json] [--json levers.json]

Each lever saves some milliseconds of a single-stream step; its ceiling is the
release suite's tokens a step over the shortened step (spark-ds41f-1025 single-stream
suites, candidate 6: TP4 33.003 steps/s at 2.284 tokens a step, TP3 26.730 at 2.269).
Savings are per step at the suite's routes (13.8 distinct experts a layer, #598) and
count 43 MoE launches (40 target layers, 3 drafter). Every saving is a part of #598's
non-byte time (busy below the attainable rate, or DRAM idle), never of the byte floor.

The routed experts' items come from routed_chain.json (TP4 production traces):
- the SwiGLU between the GEMMs and the idle gaps before it and before FC2;
- FC2's fixed cost, half of it its ramp (the half a continuous stream removes; the
  other half, its tail into the combine, stays);
- half of FC1's fixed cost (14.2 us at six rows, the expert-rate bundle), its tail;
- FC2's time an expert above the attainable rate, at the suite's experts.
TP3's experts are 768 columns a rank: FC2's per-expert excess scales by 4/3, the fixed
costs do not. The ramp/tail halves are assumptions until the c1 layer traces split
each launch (cta_timeline.py).
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = {"TP4": {"steps_per_s": 33.003, "tokens": 2.284}, "TP3": {"steps_per_s": 26.730, "tokens": 2.269}}
EXPERTS = 13.8  # distinct a layer at the suite's live rows (#598, model.json)
LAUNCHES = 43
FC1_FIXED = 14.2  # us, expert-rate bundle


def ceiling(tp, saved_ms):
    s = SUITE[tp]
    step = 1000 / s["steps_per_s"]
    return {"saved_ms": round(saved_ms, 3), "step_ms": round(step, 2),
            "tok_s": round(s["tokens"] / (step - saved_ms) * 1000, 1),
            "today_tok_s": round(s["tokens"] * s["steps_per_s"], 1),
            "gain_percent": round(100 * (step / (step - saved_ms) - 1), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=os.path.join(HERE, "routed_chain.json"))
    ap.add_argument("--json")
    args = ap.parse_args()
    chain = json.load(open(args.chain))["greedy"]
    launches = chain["launches"]
    swiglu = launches["SwiGLU and cast"]["dur_us"] + launches["SwiGLU and cast"]["gap_us"] + launches["FC2"]["gap_us"]
    fc2_ramp = chain["fc2_fixed_us"] / 2
    fc1_tail = FC1_FIXED / 2
    fc2_rate = (chain["fc2_us_per_expert"] - chain["fc2_us_per_expert_at_rate"]) * EXPERTS
    routed = {
        "SwiGLU kernel and its gaps": (swiglu, swiglu),
        "FC2 ramp (half its fixed cost)": (fc2_ramp, fc2_ramp),
        "FC1 tail (half its fixed cost)": (fc1_tail, fc1_tail),
        "FC2 above the attainable rate an expert": (fc2_rate, fc2_rate * 768 / 576),
    }
    per_layer = {tp: sum(v[i] for v in routed.values()) for i, tp in enumerate(("TP4", "TP3"))}
    levers = [
        {"lever": "Routed experts as one continuous FC1-FC2 stream (FC1 tail, SwiGLU, FC2 ramp, FC2 at rate)",
         "owner": "this study (prototype 1 is the stream GEMM, a part of it)",
         "parts_us_per_layer": {k: [round(v[0], 2), round(v[1], 2)] for k, v in routed.items()},
         "TP4": ceiling("TP4", per_layer["TP4"] * LAUNCHES / 1000),
         "TP3": ceiling("TP3", per_layer["TP3"] * LAUNCHES / 1000)},
        {"lever": "All-reduce: peer skew and in-kernel staging, at most the DRAM-idle time under collectives",
         "owner": "sparknet (cluster-only; not prototyped here)",
         "TP4": ceiling("TP4", 0.95), "TP3": ceiling("TP3", 0.95)},
        {"lever": "L2 prefetch timed against its consumers",
         "owner": "l2-prefetch-timing session (not counted here)",
         "TP4": ceiling("TP4", 1.67), "TP3": None},
        {"lever": "LM heads at the attainable rate (#598 row 12)", "owner": "unclaimed",
         "TP4": ceiling("TP4", 0.20), "TP3": ceiling("TP3", 0.20 * 4 / 3)},
        {"lever": "Routed GEMMs: deeper stages, warp-specialized producer, fewer block syncs (steady state)",
         "owner": "none: the steady stream is at 98% of the attainable rate",
         "TP4": ceiling("TP4", 0.0), "TP3": ceiling("TP3", 0.0)},
    ]
    combined = per_layer["TP4"] * LAUNCHES / 1000 + 0.95 + 0.20
    combined3 = per_layer["TP3"] * LAUNCHES / 1000 + 0.95 + 0.20 * 4 / 3
    result = {"suite": SUITE, "experts": EXPERTS, "launches": LAUNCHES, "levers": levers,
              "this_study_and_unclaimed_combined": {"TP4": ceiling("TP4", combined), "TP3": ceiling("TP3", combined3)}}
    print("| lever | owner | TP4 ms | TP4 tok/s | TP3 ms | TP3 tok/s |")
    print("|---|---|---:|---:|---:|---:|")
    for lv in levers:
        t3 = lv["TP3"]
        print(f"| {lv['lever']} | {lv['owner']} | {lv['TP4']['saved_ms']} | {lv['TP4']['tok_s']} "
              f"(+{lv['TP4']['gain_percent']}%) | {t3['saved_ms'] if t3 else '-'} | "
              f"{str(t3['tok_s']) + ' (+' + str(t3['gain_percent']) + '%)' if t3 else '-'} |")
    c = result["this_study_and_unclaimed_combined"]
    print(f"\ncombined (routed stream, all-reduce, heads): TP4 {c['TP4']['tok_s']} tok/s, TP3 {c['TP3']['tok_s']}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")


if __name__ == "__main__":
    main()
