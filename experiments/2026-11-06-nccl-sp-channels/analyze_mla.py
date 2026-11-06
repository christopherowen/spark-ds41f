#!/usr/bin/env python3
"""Summarize 2-boot standalone GPU Sparse-MLA 0731 receipts."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
RECEIPTS_DIR = EXP_DIR / "receipts"


def summarize(prefix: str = "screen-mla") -> dict:
    records = []
    for boot in ("boot1", "boot2"):
        p = RECEIPTS_DIR / f"{prefix}-{boot}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    bit_checks = [r for r in records if r.get("kind") == "bit_identity"]
    timings = [r for r in records if r.get("kind") == "timing"]

    assert all(r["passed"] and r["int16_differ"] == 0 for r in bit_checks)
    print(f"[{prefix}] Total bit_identity checks across boot1 & boot2: {len(bit_checks)}, all passed=True")

    grouped = defaultdict(dict)
    for r in timings:
        key = (r["heads"], r["idx_width"], r["rows"], r["context_mode"], r["cache_mode"])
        grouped[key][(r["boot"], r["variant"])] = r

    print(
        f"{'heads':>5} {'idx':>4} {'rows':>5} {'context':>14} {'cache':>10} | "
        f"{'ctrl_rt':>7} {'cand_rt':>7} | {'ctrl_b1':>8} {'ctrl_b2':>8} | "
        f"{'cand_b1':>8} {'cand_b2':>8} | {'delta%':>7} {'speedup':>7}"
    )
    print("-" * 126)

    summary_rows = []
    for key in sorted(grouped.keys()):
        heads, idx_w, rows, ctx, cache = key
        g = grouped[key]
        c_b1 = g[("boot1", "control_pre0731")]["median_ms"]
        c_b2 = g[("boot2", "control_pre0731")]["median_ms"]
        n_b1 = g[("boot1", "candidate_0731")]["median_ms"]
        n_b2 = g[("boot2", "candidate_0731")]["median_ms"]
        ctrl_rt = g[("boot1", "control_pre0731")]["route"]
        cand_rt = g[("boot1", "candidate_0731")]["route"]
        c_mean = 0.5 * (c_b1 + c_b2)
        n_mean = 0.5 * (n_b1 + n_b2)
        delta_pct = (n_mean - c_mean) / c_mean * 100.0
        speedup = c_mean / n_mean
        print(
            f"{heads:5d} {idx_w:4d} {rows:5d} {ctx:>14} {cache:>10} | "
            f"{ctrl_rt:>7} {cand_rt:>7} | {c_b1:8.4f} {c_b2:8.4f} | "
            f"{n_b1:8.4f} {n_b2:8.4f} | {delta_pct:+6.2f}% {speedup:6.3f}x"
        )
        summary_rows.append(
            {
                "heads": heads,
                "idx_width": idx_w,
                "rows": rows,
                "context_mode": ctx,
                "cache_mode": cache,
                "control_route": ctrl_rt,
                "candidate_route": cand_rt,
                "control_boot1_ms": round(c_b1, 4),
                "control_boot2_ms": round(c_b2, 4),
                "control_mean_ms": round(c_mean, 4),
                "candidate_boot1_ms": round(n_b1, 4),
                "candidate_boot2_ms": round(n_b2, 4),
                "candidate_mean_ms": round(n_mean, 4),
                "delta_pct": round(delta_pct, 2),
                "speedup_x": round(speedup, 3),
            }
        )
    return {
        "bit_identity_checks": len(bit_checks),
        "all_bit_identical": True,
        "comparisons": summary_rows,
    }


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "screen-mla"
    summarize(prefix)
