#!/usr/bin/env python3
"""A one-stream TP4 decode step as bytes plus overhead, and what each lever can win.

usage: model.py [--floor floor.json] [--routes ROUTES.json] [--weights weight_stats.json] [--out model.json]

The model, per step and rank:

    step = overhead + (dense + drafter_routed + 40 * N(live) * expert) / rate + dead * row

- dense: every non-routed weight byte of the target, both LM heads and the drafter
  (floor.py's DENSE); expert: one routed expert's FC1 + FC2 bytes at TP4;
- N(r): distinct experts a target layer's r routed rows reach, from DeepSeek's reference
  forward on CPU (experiments/2026-10-18-json-count: the routes of every layer for eight
  consecutive tokens; windows of r consecutive tokens, averaged over layers and starts);
- live: rows that run the routed experts (the anchor plus verified drafts above the
  dead-row cut); dead: verified rows below the cut, which skip the routed experts and cost
  about a dummy row (the r6v static table: 13.66 to 15.49 ms over 1 to 6 identical rows);
- rate and overhead, in two bases from floor.json:
  - "measured": the expert-rate bundle's FC1 marginal rate at six rows, the traced
    window's experts from inverting its FC1 fit, overhead = DRAM-idle time plus the
    streaming below that rate;
  - "first estimate": the packed LM head's rate with FC1 assumed at it (by_eta 1.00).

Calibrated to the release suite's single stream (spark-ds41f-1022, manifests/suites/
2026-10-22-spark-ds41f-1022-tp4/c1.json: 32.73 steps a second, 2.271 tokens a step,
3.073 verified drafts a step), it solves for the live rows. Every lever is then the
step it leaves and the tokens a step it gives, in tok/s.
"""
import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = os.path.join(HERE, "..", "2026-10-18-json-count", "runs", "ref-woa1.json")

SUITE = {"steps_per_s": 32.733, "tokens_per_step": 2.2708, "verified_drafts": 3.0733}
DEAD_ROW_MS = (15.49 - 13.66) / 5  # r6v static table: identical rows reach no new experts
LAYERS, DRAFTER_LAYERS = 40, 3
TARGET = 100.0
SPEC = 273e6  # bytes a ms
QAKV_BYTES = 1792 * 5120 + (1792 // 32) * (5120 // 32)
ALLGATHER_US = 11.2  # one-shot all-gather at 10 KiB, sparknet wide-read probe (median, slowest rank)
HEAD_BYTES = 32320 * 5120 * 3 // 2
SCREEN_BYTES = 32320 * 5120 // 2  # a 4-bit screening copy of the head shard
FN_BYTES = 40 * 2 * 4 * 24 * 4 * 5120


def distinct(path):
    steps = json.load(open(path))["steps"]
    routes = [[s["routes_last_position"][str(layer)] for s in steps] for layer in range(LAYERS)]
    out = {}
    for r in range(1, len(steps) + 1):
        out[r] = statistics.mean(len({e for t in range(s, s + r) for e in routes[layer][t]})
                                 for layer in range(LAYERS) for s in range(len(steps) - r + 1))
    return out


def interp(table, r):
    lo = max(1, min(int(r), max(table) - 1))
    return table[lo] + (table[lo + 1] - table[lo]) * (r - lo)


def scenario(floor, N, basis, weights):
    greedy = floor["windows"]["greedy"]
    if basis == "measured":
        g = greedy["measured"]
        rate = g["rate_gbs (FC1 marginal, six rows)"] * 1e6  # bytes a ms
    else:
        g = greedy["by_eta"]["1.00"]
        rate = greedy["attainable_gbs (LM head)"] * 1e6
    expert = floor["fc1_bytes_per_expert"] + floor["fc2_bytes_per_expert"]
    dense = sum(floor["dense_bytes"].values())
    drafter_routed = DRAFTER_LAYERS * g["experts_per_drafter_layer"] * expert
    idle = sum(greedy["dram_idle_ms"].values())
    below = g["busy_minus_floor_ms"]
    overhead = idle + below
    step_today = 1e3 / SUITE["steps_per_s"]
    T_today = SUITE["tokens_per_step"]
    rows_today = 1 + SUITE["verified_drafts"]

    def stream_ms(live, extra_bytes=0.0):
        return (dense + drafter_routed + LAYERS * interp(N, live) * expert + extra_bytes) / rate

    lo, hi = 1.0, rows_today
    for _ in range(60):
        live = (lo + hi) / 2
        if overhead + stream_ms(live) + (rows_today - live) * DEAD_ROW_MS > step_today:
            hi = live
        else:
            lo = live
    dead = rows_today - live
    floor_today = stream_ms(live)
    tps = lambda T, step: 1e3 * T / step
    row_ms = (stream_ms(live + 0.5) - stream_ms(live - 0.5)) + DEAD_ROW_MS
    s = {"rate_gbs": round(rate / 1e6, 1), "overhead_ms": round(overhead, 2), "dram_idle_ms": round(idle, 2),
         "below_rate_ms": below, "live_rows_today": round(live, 2), "dead_rows_today": round(dead, 2),
         "experts_per_layer_today": round(interp(N, live), 2), "floor_ms_today": round(floor_today, 2),
         "floor_share": round(floor_today / step_today, 3), "live_row_ms": round(row_ms, 2),
         "tok_s_today": round(tps(T_today, step_today), 1), "levers": {}}
    L = s["levers"]

    def lever(name, T, step, note=""):
        L[name] = {"tokens_per_step": round(T, 3), "step_ms": round(step, 2), "tok_s": round(tps(T, step), 1),
                   "gain_pct": round(100 * (tps(T, step) / tps(T_today, step_today) - 1), 1), "note": note}

    base = step_today
    lever("everything but the bytes (overhead to 0)", T_today, base - overhead)
    lever("DRAM-idle time to 0 (fusion, PDL, launch gaps, collectives)", T_today, base - idle)
    for key, v in greedy["dram_idle_ms"].items():
        lever(f"DRAM idle: {key}", T_today, base - v)
    lever("streaming at the attainable rate (below-rate time to 0)", T_today, base - below)
    if basis == "measured":
        lever("routed experts' launch fixed costs to 0 (one continuous FC1+FC2 stream)", T_today,
              base - g["moe_launch_fixed_ms"])
        lever("both LM heads at the expert GEMM's marginal rate", T_today, base - g["head_excess_ms"])
    lever("FC2 at FC1's per-byte rate", T_today,
          base - LAYERS * interp(N, live) * floor["fc2_bytes_per_expert"] / rate * (1 / greedy["fc2_rate_vs_fc1"] - 1))
    lever("every byte at GB10's 273 GB/s specification", T_today, base - floor_today * (1 - rate / SPEC))
    qakv = (LAYERS + DRAFTER_LAYERS) * QAKV_BYTES * 3 / 4 / rate - (LAYERS + DRAFTER_LAYERS) * ALLGATHER_US / 1e3
    lever("Q-A/KV column-sharded, one all-gather a layer", T_today, base - qakv)
    fn_exact = weights.get("fn_exactly_bf16") if weights else None
    fn = FN_BYTES / 2 / rate if fn_exact is not False else 0.0
    lever("mHC fn read as BF16" + (" (measured: not exactly BF16, so nothing)" if fn_exact is False else ""),
          T_today, base - fn)
    heads = 2 * (HEAD_BYTES - SCREEN_BYTES) / rate * 0.9
    lever("both heads through a certified exact screen (owner decision)", T_today, base - heads)
    fp8_bytes = sum(v for k, v in floor["dense_bytes"].items() if "head" not in k and "router" not in k
                    and "fn" not in k and "compressor" not in k and "Markov" not in k)
    fp4_bytes = LAYERS * interp(N, live) * expert + drafter_routed
    h8, h4 = ((weights["entropy_bits"]["fp8 (of 8)"], weights["entropy_bits"]["fp4 nibble (of 4)"])
              if weights else (7.0, 3.8))
    entropy = (fp8_bytes * (1 - h8 / 8) + fp4_bytes * (1 - h4 / 4)) / rate
    lever(f"entropy-coded weights at {h8}/8 and {h4}/4 bits" + (" (measured)" if weights else " (assumed)"),
          T_today, base - entropy)
    # tokens a step: today's drafts verified in full give T5; the policy's cut leaves
    # 1.93 positions whose survival is below the marginal admitted one
    for s_tail in (0.07, 0.17):
        T5 = T_today + (5 - SUITE["verified_drafts"]) * s_tail
        lever(f"oracle verification policy, T5 = {T5:.2f}", T5, overhead + stream_ms(T5),
              "every verified row is accepted; no dead rows")
        lever(f"every draft verified, T5 = {T5:.2f}", T5, overhead + stream_ms(6))
    lever("oracle drafter at 5 drafts", 6.0, overhead + stream_ms(6))
    for p_rej, p_alt in ((0.25, 1.0), (0.25, 0.5)):
        lever(f"tree: one second-candidate row at position 1 (P(reject) {p_rej}, right {p_alt:.0%})",
              T_today + p_rej * p_alt, base + row_ms)
    lever("two more drafts (6-7), dead rows, tail survival 0.06 + 0.04", T_today + 0.10, base + 2 * DEAD_ROW_MS)
    combos = s["combinations"] = {}
    for name, T, step in (
            ("credible: half the overhead, Q-A/KV sharded", T_today, base - 0.5 * overhead - qakv - fn),
            ("credible, and 3% more tokens a step (policy, drafter window fix)", 1.03 * T_today,
             base - 0.5 * overhead - qakv - fn),
            ("two thirds of the overhead, Q-A/KV sharded", T_today, base - 2 / 3 * overhead - qakv - fn),
            ("stretch: 75% of the overhead, the exact byte cuts, the head screen, entropy coding", T_today,
             base - 0.75 * overhead - qakv - fn - heads - entropy),
            ("the whole floor and the exact byte cuts", T_today, base - overhead - qakv - fn),
            ("the whole floor and the oracle verification policy (T5 2.60)", 2.60, stream_ms(2.60) - qakv - fn)):
        combos[name] = {"tokens_per_step": round(T, 3), "step_ms": round(step, 2), "tok_s": round(tps(T, step), 1)}
    frontier = []
    for cut in (0.0, 0.25, 0.5, 0.75, 1.0):
        T = T_today
        for _ in range(200):
            step = (1 - cut) * overhead + stream_ms(T + (live - T_today)) + dead * DEAD_ROW_MS
            if tps(T, step) >= TARGET:
                break
            T += 0.01
        frontier.append({"overhead_removed": cut, "tokens_per_step_for_100": round(T, 2),
                         "vs_today_pct": round(100 * (T / T_today - 1), 1), "step_ms": round(step, 2)})
    s["frontier_100"] = frontier
    # the overhead share that 100 tok/s needs at today's tokens a step
    need = (base - 1e3 * T_today / TARGET - qakv) / overhead
    s["overhead_share_for_100_at_today_tokens"] = round(need, 3)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", default=os.path.join(HERE, "floor.json"))
    ap.add_argument("--routes", default=ROUTES)
    ap.add_argument("--weights", default=os.path.join(HERE, "runs", "weight_stats-dgx4.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "model.json"))
    args = ap.parse_args()
    floor = json.load(open(args.floor))
    weights = json.load(open(args.weights)) if os.path.exists(args.weights) else None
    N = distinct(args.routes)
    out = {"inputs": {"suite_c1": SUITE, "step_ms": round(1e3 / SUITE["steps_per_s"], 3),
                      "dense_mb": round(sum(floor["dense_bytes"].values()) / 1e6, 1),
                      "expert_mb": round((floor["fc1_bytes_per_expert"] + floor["fc2_bytes_per_expert"]) / 1e6, 3),
                      "distinct_experts_by_rows": {r: round(v, 2) for r, v in N.items()},
                      "dead_row_ms": round(DEAD_ROW_MS, 3), "weight_stats": bool(weights)},
           "scenarios": {}}
    bases = (["measured"] if "measured" in floor["windows"]["greedy"] else []) + ["first estimate"]
    for basis in bases:
        out["scenarios"][basis] = scenario(floor, N, basis, weights)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    for name, s in out["scenarios"].items():
        print(f"{name}: {s['rate_gbs']} GB/s, live rows {s['live_rows_today']}, dead {s['dead_rows_today']}, "
              f"floor {s['floor_ms_today']} ms ({s['floor_share']:.0%}), overhead {s['overhead_ms']} ms, "
              f"a live row {s['live_row_ms']} ms; 100 at today's tokens needs "
              f"{s['overhead_share_for_100_at_today_tokens']:.0%} of the overhead removed")
        for k, v in s["levers"].items():
            print(f"  {v['tok_s']:6.1f} tok/s ({v['gain_pct']:+5.1f}%)  T {v['tokens_per_step']:.2f} step {v['step_ms']:5.2f}  {k}")
        for k, v in s["combinations"].items():
            print(f"  = {v['tok_s']:6.1f} tok/s  T {v['tokens_per_step']:.2f} step {v['step_ms']:5.2f}  {k}")
        for f in s["frontier_100"]:
            print(f"  100 tok/s with {f['overhead_removed']:.0%} of the overhead removed: "
                  f"{f['tokens_per_step_for_100']} tokens a step ({f['vs_today_pct']:+.0f}%)")


if __name__ == "__main__":
    main()
