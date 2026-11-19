#!/usr/bin/env python3
"""Analyze 2-boot 3-rank RoCE NCCL collective and overlap benchmark receipts."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE / "receipts"


def load_records() -> list[dict]:
    records: list[dict] = []
    for boot_file in sorted(RECEIPTS.glob("screen-tp3-nccl-boot*.jsonl")):
        for line in boot_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize() -> dict[str, object]:
    records = load_records()
    inits = [r for r in records if r.get("kind") == "nccl_init"]
    checks = [r for r in records if r.get("kind") == "bit_identity"]
    timings = [r for r in records if r.get("kind") == "timing"]

    grouped: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    meta_by_case: dict[tuple[str, int], dict[str, object]] = {}
    for row in timings:
        key = (row["arm"], row["name"], int(row["T"]))
        grouped[key][row["boot"]] = row
        meta_by_case[(row["name"], int(row["T"]))] = {
            "L": int(row["L"]),
            "op_kind": row["op_kind"],
            "payload_bytes": int(row["payload_bytes"]),
        }

    cases_order = [
        "engram_sp_ag_bf16",
        "moe_sp_gather_mxfp8",
        "attn_front_ag_u8",
        "wo_local_exchange_mxfp8_a2a",
        "wo_local_exchange_slice_a2a",
        "mlp_all_to_all_bf16",
        "moe_reduce_scatter_bf16",
        "mhc_norm_all_reduce_bf16",
        "rank_order_all_reduce_bf16",
        "overlap_proj_rs_bf16",
        "overlap_engram_proj_a2a_bf16",
        "overlap_moe_ag_shared_gemm",
    ]
    tokens_order = [16, 256, 1024, 6144]
    arms = ["control_c8_1m", "cand_c6_8m_bidir", "cand_c6_8m_bidir_p2p512k"]

    comparisons: list[dict[str, object]] = []
    for tokens in tokens_order:
        for case in cases_order:
            base_boots = grouped.get(("control_c8_1m", case, tokens), {})
            c256_boots = grouped.get(("cand_c6_8m_bidir", case, tokens), {})
            c512_boots = grouped.get(("cand_c6_8m_bidir_p2p512k", case, tokens), {})
            if not base_boots or not c256_boots:
                continue
            base_vals = [float(r["median_ms"]) for r in base_boots.values()]
            c256_vals = [float(r["median_ms"]) for r in c256_boots.values()]
            c512_vals = [float(r["median_ms"]) for r in c512_boots.values()] if c512_boots else []

            base_mean = statistics.mean(base_vals)
            c256_mean = statistics.mean(c256_vals)
            c512_mean = statistics.mean(c512_vals) if c512_vals else None

            base_busbw = statistics.mean(float(r["busbw_gbps"]) for r in base_boots.values())
            c256_busbw = statistics.mean(float(r["busbw_gbps"]) for r in c256_boots.values())
            c512_busbw = statistics.mean(float(r["busbw_gbps"]) for r in c512_boots.values()) if c512_boots else None

            meta = meta_by_case[(case, tokens)]
            entry: dict[str, object] = {
                "T": tokens,
                "L": meta["L"],
                "name": case,
                "op_kind": meta["op_kind"],
                "payload_bytes": meta["payload_bytes"],
                "control_c8_1m": {
                    "boot1_ms": round(float(base_boots["boot1"]["median_ms"]), 4),
                    "boot2_ms": round(float(base_boots["boot2"]["median_ms"]), 4),
                    "mean_ms": round(base_mean, 4),
                    "half_range_ms": round(abs(float(base_boots["boot1"]["median_ms"]) - float(base_boots["boot2"]["median_ms"])) / 2.0, 4),
                    "busbw_gbps": round(base_busbw, 3),
                },
                "cand_c6_8m_bidir": {
                    "boot1_ms": round(float(c256_boots["boot1"]["median_ms"]), 4),
                    "boot2_ms": round(float(c256_boots["boot2"]["median_ms"]), 4),
                    "mean_ms": round(c256_mean, 4),
                    "half_range_ms": round(abs(float(c256_boots["boot1"]["median_ms"]) - float(c256_boots["boot2"]["median_ms"])) / 2.0, 4),
                    "busbw_gbps": round(c256_busbw, 3),
                    "delta_ms_vs_control": round(c256_mean - base_mean, 4),
                    "delta_pct_vs_control": round((c256_mean - base_mean) / base_mean * 100.0, 2),
                    "speedup_x_vs_control": round(base_mean / c256_mean, 3),
                },
            }
            if c512_mean is not None and c512_busbw is not None:
                entry["cand_c6_8m_bidir_p2p512k"] = {
                    "boot1_ms": round(float(c512_boots["boot1"]["median_ms"]), 4),
                    "boot2_ms": round(float(c512_boots["boot2"]["median_ms"]), 4),
                    "mean_ms": round(c512_mean, 4),
                    "half_range_ms": round(abs(float(c512_boots["boot1"]["median_ms"]) - float(c512_boots["boot2"]["median_ms"])) / 2.0, 4),
                    "busbw_gbps": round(c512_busbw, 3),
                    "delta_ms_vs_control": round(c512_mean - base_mean, 4),
                    "delta_pct_vs_control": round((c512_mean - base_mean) / base_mean * 100.0, 2),
                    "speedup_x_vs_control": round(base_mean / c512_mean, 3),
                    "delta_pct_vs_p2p256k": round((c512_mean - c256_mean) / c256_mean * 100.0, 2),
                }
            comparisons.append(entry)

    bundle_totals: list[dict[str, object]] = []
    standalone_ops = [c for c in cases_order if not c.startswith("overlap_") and c != "mhc_norm_all_reduce_bf16"]
    overlap_ops = [c for c in cases_order if c.startswith("overlap_")]
    for tokens in tokens_order:
        rows_t = [r for r in comparisons if r["T"] == tokens]
        st_rows = [r for r in rows_t if r["name"] in standalone_ops]
        ov_rows = [r for r in rows_t if r["name"] in overlap_ops]

        st_ctrl = sum(float(r["control_c8_1m"]["mean_ms"]) for r in st_rows)
        st_c256 = sum(float(r["cand_c6_8m_bidir"]["mean_ms"]) for r in st_rows)
        st_c512 = sum(float(r["cand_c6_8m_bidir_p2p512k"]["mean_ms"]) for r in st_rows)

        ov_ctrl = sum(float(r["control_c8_1m"]["mean_ms"]) for r in ov_rows)
        ov_c256 = sum(float(r["cand_c6_8m_bidir"]["mean_ms"]) for r in ov_rows)
        ov_c512 = sum(float(r["cand_c6_8m_bidir_p2p512k"]["mean_ms"]) for r in ov_rows)

        bundle_totals.append({
            "T": tokens,
            "L": rows_t[0]["L"] if rows_t else 0,
            "standalone_sp_bundle_ms": {
                "control_c8_1m": round(st_ctrl, 4),
                "cand_c6_8m_bidir": round(st_c256, 4),
                "cand_c6_8m_bidir_p2p512k": round(st_c512, 4),
                "c256_delta_pct_vs_control": round((st_c256 - st_ctrl) / st_ctrl * 100.0, 2),
                "c256_speedup_x_vs_control": round(st_ctrl / st_c256, 3),
                "c512_delta_pct_vs_control": round((st_c512 - st_ctrl) / st_ctrl * 100.0, 2),
                "c512_speedup_x_vs_control": round(st_ctrl / st_c512, 3),
            },
            "overlapped_gemm_plus_coll_bundle_ms": {
                "control_c8_1m": round(ov_ctrl, 4),
                "cand_c6_8m_bidir": round(ov_c256, 4),
                "cand_c6_8m_bidir_p2p512k": round(ov_c512, 4),
                "c256_delta_pct_vs_control": round((ov_c256 - ov_ctrl) / ov_ctrl * 100.0, 2),
                "c256_speedup_x_vs_control": round(ov_ctrl / ov_c256, 3),
                "c512_delta_pct_vs_control": round((ov_c512 - ov_ctrl) / ov_ctrl * 100.0, 2),
                "c512_speedup_x_vs_control": round(ov_ctrl / ov_c512, 3),
            },
        })

    return {
        "arms": arms,
        "nccl_inits": inits,
        "bit_identity_check_count": len(checks),
        "failed_bit_identity_checks": [c for c in checks if not c.get("passed", False)],
        "bundle_totals": bundle_totals,
        "comparisons": comparisons,
    }


def main() -> int:
    summary = summarize()
    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({summary['bit_identity_check_count']} bit-identity checks, {len(summary['failed_bit_identity_checks'])} failures)")
    print("\n=== Bundle Totals across T in (16, 256, 1024, 6144) ===")
    for b in summary["bundle_totals"]:
        st = b["standalone_sp_bundle_ms"]
        ov = b["overlapped_gemm_plus_coll_bundle_ms"]
        print(
            f"T={b['T']:4d} (L={b['L']:4d}) | "
            f"Standalone SP bundle: ctrl={st['control_c8_1m']:.4f}ms, c256={st['cand_c6_8m_bidir']:.4f}ms ({st['c256_delta_pct_vs_control']:+.2f}%, {st['c256_speedup_x_vs_control']:.2f}x), "
            f"c512={st['cand_c6_8m_bidir_p2p512k']:.4f}ms ({st['c512_delta_pct_vs_control']:+.2f}%, {st['c512_speedup_x_vs_control']:.2f}x) | "
            f"Overlapped GEMM+coll: ctrl={ov['control_c8_1m']:.4f}ms, c256={ov['cand_c6_8m_bidir']:.4f}ms ({ov['c256_delta_pct_vs_control']:+.2f}%, {ov['c256_speedup_x_vs_control']:.2f}x), "
            f"c512={ov['cand_c6_8m_bidir_p2p512k']:.4f}ms ({ov['c512_delta_pct_vs_control']:+.2f}%, {ov['c512_speedup_x_vs_control']:.2f}x)"
        )
    print("\n=== Per-Op 2-Boot Means ===")
    print(f"{'T':>5s} {'name':30s} {'ctrl_ms':>9s} {'c256_ms':>9s} {'d%':>8s} {'spd':>6s} {'c512_ms':>9s} {'d%':>8s}")
    for row in summary["comparisons"]:
        ctrl = row["control_c8_1m"]["mean_ms"]
        c256 = row["cand_c6_8m_bidir"]["mean_ms"]
        d256 = row["cand_c6_8m_bidir"]["delta_pct_vs_control"]
        s256 = row["cand_c6_8m_bidir"]["speedup_x_vs_control"]
        c512 = row["cand_c6_8m_bidir_p2p512k"]["mean_ms"]
        d512 = row["cand_c6_8m_bidir_p2p512k"]["delta_pct_vs_control"]
        print(f"{row['T']:5d} {row['name']:30s} {ctrl:9.4f} {c256:9.4f} {d256:+7.2f}% {s256:5.2f}x {c512:9.4f} {d512:+7.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
