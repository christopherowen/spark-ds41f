#!/usr/bin/env python3
"""Summarize 2-boot 3-rank TP3 NCCL collective benchmark receipts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
RECEIPTS_DIR = EXP_DIR / "receipts"


def summarize() -> dict:
    records = []
    for boot in ("boot1", "boot2"):
        p = RECEIPTS_DIR / f"screen-nccl-tp3-{boot}.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))

    init_recs = [r for r in records if r.get("kind") == "nccl_init"]
    bit_recs = [r for r in records if r.get("kind") == "bit_identity"]
    timings = [r for r in records if r.get("kind") == "timing"]

    assert all(r["passed"] and r["differ_ag"] == 0 and r["differ_a2a"] == 0 and r["differ_rank_order_ar"] == 0 for r in bit_recs)
    print(f"Total bit_identity checks across boot1 & boot2: {len(bit_recs)}, all passed=True")

    print("\n=== NCCL Init Topology Summary (Rank 0) ===")
    topo_by_arm = {}
    for r in init_recs:
        topo_by_arm[(r["arm"], r["boot"])] = r
        if r["boot"] == "boot1":
            print(
                f"  {r['arm']:20s}: coll_channels={r['coll_channels']}, "
                f"p2p_channels={r['p2p_channels']}, p2p_chunk={r['p2p_chunksize']}, "
                f"trees={r['trees_count']}, buff_sizes={r['buff_sizes']}"
            )
            for ring in r["rings"]:
                print(f"    {ring}")

    grouped = defaultdict(dict)
    arms_seen = []
    for r in timings:
        if r["arm"] not in arms_seen:
            arms_seen.append(r["arm"])
        key = (r["T"], r["name"])
        grouped[key][(r["boot"], r["arm"])] = r

    arms_order = [
        "control_c8_1m",
        "cand_c4_1m",
        "cand_c4_4m_bidir",
        "cand_c4_8m",
        "cand_c4_8m_bidir",
        "cand_c6_8m",
        "cand_c6_8m_bidir",
    ]
    arms_order = [a for a in arms_order if a in arms_seen]

    print("\n=== 2-Boot Collective Latency Summary (Median ms across 3 ranks, boot1 / boot2 mean) ===")
    header = f"{'T':>5} {'Collective':>28} {'MiB':>6} | {'c8_1m(ms)':>10}"
    for arm in arms_order[1:]:
        short = arm.replace("cand_", "")
        header += f" | {short:>18}"
    print(header)
    print("-" * len(header))

    comparisons = []
    totals_by_T_arm = defaultdict(lambda: defaultdict(float))
    for (T, name) in sorted(grouped.keys()):
        g = grouped[(T, name)]
        c_b1 = g[("boot1", "control_c8_1m")]["median_ms"]
        c_b2 = g[("boot2", "control_c8_1m")]["median_ms"]
        c_mean = 0.5 * (c_b1 + c_b2)
        mib = g[("boot1", "control_c8_1m")]["payload_bytes"] / (1024 * 1024)
        totals_by_T_arm[T]["control_c8_1m"] += c_mean

        row_str = f"{T:5d} {name:>28} {mib:6.2f} | {c_mean:10.4f}"
        arm_metrics = {
            "control_c8_1m": {
                "boot1_ms": c_b1,
                "boot2_ms": c_b2,
                "mean_ms": round(c_mean, 4),
                "busbw_gbps": round(
                    0.5 * (g[("boot1", "control_c8_1m")]["busbw_gbps"] + g[("boot2", "control_c8_1m")]["busbw_gbps"]),
                    3,
                ),
            }
        }
        for arm in arms_order[1:]:
            a_b1 = g[("boot1", arm)]["median_ms"]
            a_b2 = g[("boot2", arm)]["median_ms"]
            a_mean = 0.5 * (a_b1 + a_b2)
            totals_by_T_arm[T][arm] += a_mean
            delta_pct = (a_mean - c_mean) / c_mean * 100.0
            speedup = c_mean / a_mean
            row_str += f" | {a_mean:7.4f} ({delta_pct:+6.1f}%)"
            arm_metrics[arm] = {
                "boot1_ms": a_b1,
                "boot2_ms": a_b2,
                "mean_ms": round(a_mean, 4),
                "delta_pct": round(delta_pct, 2),
                "speedup_x": round(speedup, 3),
                "busbw_gbps": round(
                    0.5 * (g[("boot1", arm)]["busbw_gbps"] + g[("boot2", arm)]["busbw_gbps"]),
                    3,
                ),
            }
        print(row_str)
        comparisons.append(
            {
                "T": T,
                "name": name,
                "payload_bytes": g[("boot1", "control_c8_1m")]["payload_bytes"],
                "arms": arm_metrics,
            }
        )

    print("\n=== Aggregate Per-Token-Shape Latency Across All 8 Collectives ===")
    for T in sorted(totals_by_T_arm.keys()):
        c_tot = totals_by_T_arm[T]["control_c8_1m"]
        line = f"T={T:4d}: control_c8_1m={c_tot:8.4f} ms"
        for arm in arms_order[1:]:
            a_tot = totals_by_T_arm[T][arm]
            dp = (a_tot - c_tot) / c_tot * 100.0
            sp = c_tot / a_tot
            line += f" | {arm.replace('cand_', '')}={a_tot:8.4f} ms ({dp:+5.1f}%, {sp:.3f}x)"
        print(line)

    return {
        "bit_identity_checks": len(bit_recs),
        "all_bit_identical": True,
        "init_records": [r for r in init_recs if r["boot"] == "boot1"],
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    summarize()
