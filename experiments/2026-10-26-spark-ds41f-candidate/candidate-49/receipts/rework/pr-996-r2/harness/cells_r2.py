#!/usr/bin/env python3
"""PR #996 rework r2 (ours): print the per-cell table of analysis-r2.json, and every
non-tail cell's spread across arms (in microseconds), for the report.

usage: cells_r2.py <analysis-r2.json>
"""
import json
import sys

a = json.load(open(sys.argv[1]))
for metric, name in (("cached_calls_us_median", "per-call"), ("pass_us_per_layer_median", "20-layer pass")):
    print(f"== {name} (us): 1107 rec | r2 b1/b2 | r2 vs 1107 | r1 rec vs 1107 | r1 ctl vs 1107 | r1 ctl vs r1 rec | r2 vs r1 ctl")
    for c in a["bench"]:
        m = c[metric]
        f = lambda v: "/".join(f"{x:.1f}" for x in v)  # noqa: E731
        print(f"TP{c['tp']} T{c['T']:<5} r{c['rank']} {'tail' if c['tail'] else '    '} {f(m['1107']):>13} | "
              f"{f(m['r2']):>13} | {m['r2_vs_1107']:<6} {m['r2_vs_1107_margin_us']:>6} x{m['r2_speedup_vs_1107']:<5} | "
              f"{m['r1_recorded_vs_1107']:<6} | {m['r1_control_vs_1107']:<6} {f(m['r1_control']):>13} | "
              f"{m['r1_control_vs_r1_recorded']:<6} | {m['r2_vs_r1_control_in_session']}")
