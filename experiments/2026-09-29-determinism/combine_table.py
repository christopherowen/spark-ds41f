#!/usr/bin/env python3
"""Tabulate moe_combine_bench.py timing lines.  usage: combine_table.py FILE"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])
        if line.startswith('{"mode"') and "us_per_call" in line]
by = {}
for r in rows:
    by.setdefault((r["rows"], r["dead"]), {})[r["mode"]] = r


def g(m, mode, key):
    return m.get(mode, {}).get(key, float("nan"))


print("rows dead | per call us: atomic collapsed slices masked | topk_sum us: collapsed slices masked | fused us: atomic slices masked")
for k in sorted(by):
    m = by[k]
    print(f"{k[0]:4d} {k[1]:4.2f} | {g(m,'atomic','us_per_call'):8.1f} {g(m,'collapsed','us_per_call'):8.1f} "
          f"{g(m,'slices','us_per_call'):8.1f} {g(m,'masked','us_per_call'):8.1f} | "
          f"{g(m,'collapsed','topk_sum_us'):6.2f} {g(m,'slices','topk_sum_us'):6.2f} {g(m,'masked','topk_sum_us'):6.2f} | "
          f"{g(m,'atomic','fused_moe_us'):8.1f} {g(m,'slices','fused_moe_us'):8.1f} {g(m,'masked','fused_moe_us'):8.1f}")
