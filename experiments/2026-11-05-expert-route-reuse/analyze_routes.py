#!/usr/bin/env python3
"""Account for grouped-expert payloads from saved routes, without running a model.

Input: JSONL, one object per actual layer invocation (never pooled across steps).
See README.md for the trace contract and what the byte counts do not measure.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def ceildiv(n, d):
    return (n + d - 1) // d


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def tile_height(rows, experts, topk, policy):
    if rows * topk >= 32 * experts:
        return 64
    if policy == "1030" or rows * topk >= 16 * experts:
        return 32
    return 16


def projection(n, k, bits):
    """Logical load payload for one expert M tile, across all N CTAs.

    TP4's down projection has an exact 64-column weight tail. Its last scale
    code still covers a 128-column word; count those unused code bits separately.
    Row bases are loaded once per N CTA, not again in each K iteration.
    """
    if n % 128 or k < 128 or k % 64 or bits not in (2, 4):
        raise ValueError("expected N multiple of 128, K multiple of 64, 2/4-bit scales")
    weight = n * k // 2
    codes = n * ceildiv(k, 128) * bits // 2
    used_codes = n * (k // 32) * bits // 8
    return {
        "fp4_weight": weight,
        "scale_codes": codes,
        "scale_bases": n,
        "unused_scale_code_bits_as_bytes": codes - used_codes,
        "total": weight + codes + n,
    }


def validate(record):
    for field in ("workload", "capture", "step", "layer", "source_tree", "model_revision"):
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"missing identity: {field}")
    experts = positive(record["experts"], "experts")
    topk = positive(record["topk"], "topk")
    hidden = positive(record.get("hidden", 5120), "hidden")
    if topk > experts or hidden % 128:
        raise ValueError("invalid topk or hidden width")
    rows = record["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must contain every row, including masked/padded rows")
    for row in rows:
        ids = row["ids"]
        if len(ids) != topk or any(type(e) is not int or not -1 <= e < experts for e in ids):
            raise ValueError("invalid route IDs")
        live = [e for e in ids if e >= 0]
        if len(live) != len(set(live)):
            raise ValueError("a row may select each expert only once")
        if row.get("kind") not in ("prefill", "anchor", "draft", "padding", "dead"):
            raise ValueError("row kind is required")
        if row["kind"] in ("padding", "dead") and live:
            raise ValueError("masked rows must contain the IDs actually consumed: -1")
        if live and not isinstance(row.get("request"), str):
            raise ValueError("live rows require an anonymized request identity")
    return experts, topk, hidden, rows


def summarize(record, tp, policy):
    if tp not in (3, 4) or policy not in ("1030", "candidate-38"):
        raise ValueError("only TP3/TP4 and the recorded tile policies are supported")
    experts, topk, hidden, rows = validate(record)
    # TP3 and TP4 divide the native 2304 intermediate dimension exactly.
    intermediate = 2304 // tp
    counts = Counter(e for row in rows for e in row["ids"] if e >= 0)
    by_request = defaultdict(set)
    anchors = set()
    for row in rows:
        live = {e for e in row["ids"] if e >= 0}
        if live:
            by_request[row["request"]].update(live)
        if row["kind"] == "anchor":
            anchors.update(live)
    widths = record.get("scale_bits", {}).get(str(tp))
    if widths is not None:
        for name in ("gate_up", "down"):
            if len(widths[name]) != experts or any(b not in (2, 4) for b in widths[name]):
                raise ValueError("scale_bits must supply one 2/4-bit width per expert and projection")
    height = tile_height(len(rows), experts, topk, policy)
    tiles = {e: ceildiv(c, height) for e, c in counts.items()}
    used_tiles = sum(tiles.values())
    allocated_tiles = min(len(rows) * topk, ceildiv(len(rows) * topk, height) + experts)
    if used_tiles > allocated_tiles:
        raise ValueError("route counts exceed dispatch capacity")
    byte_cases = {}
    for scenario, fallback in (("exact", None),) if widths else (("all_2bit", 2), ("all_4bit", 4)):
        totals = Counter({key: 0 for key in (
            "unique_expert_payload_bytes", "grouped_load_payload_bytes",
            "independent_row_payload_bytes", "fp4_weight_load_bytes",
            "scale_code_load_bytes", "scale_base_load_bytes",
            "unused_scale_code_bits_as_bytes",
        )})
        for e, count in counts.items():
            for name, n, k in (("gate_up", 2 * intermediate, hidden), ("down", hidden, intermediate)):
                b = projection(n, k, widths[name][e] if widths else fallback)
                totals["unique_expert_payload_bytes"] += b["total"]
                totals["grouped_load_payload_bytes"] += tiles[e] * b["total"]
                totals["independent_row_payload_bytes"] += count * b["total"]
                totals["fp4_weight_load_bytes"] += tiles[e] * b["fp4_weight"]
                totals["scale_code_load_bytes"] += tiles[e] * b["scale_codes"]
                totals["scale_base_load_bytes"] += tiles[e] * b["scale_bases"]
                totals["unused_scale_code_bits_as_bytes"] += tiles[e] * b["unused_scale_code_bits_as_bytes"]
        totals["repeated_expert_payload_bytes"] = totals["grouped_load_payload_bytes"] - totals["unique_expert_payload_bytes"]
        totals["already_saved_by_grouping_bytes"] = totals["independent_row_payload_bytes"] - totals["grouped_load_payload_bytes"]
        byte_cases[scenario] = dict(totals)
    union = set(counts)
    request_pairs = []
    names = sorted(by_request)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = by_request[first], by_request[second]
            request_pairs.append({
                "requests": [first, second], "shared_experts": len(a & b),
                "union_experts": len(a | b), "jaccard": len(a & b) / len(a | b),
            })
    return {
        **{k: record[k] for k in ("workload", "capture", "step", "layer", "source_tree", "model_revision")},
        "evidence_kind": record.get("evidence_kind", "recorded"),
        "tp": tp, "policy": policy, "call_rows": len(rows), "topk": topk,
        "live_rows": sum(any(e >= 0 for e in r["ids"]) for r in rows),
        "live_routes": sum(counts.values()), "distinct_experts": len(union),
        "expert_counts": dict(sorted(counts.items())),
        "tile_height": height, "used_tiles": used_tiles,
        "allocated_tiles": allocated_tiles, "empty_tiles": allocated_tiles - used_tiles,
        "repeated_expert_tiles": used_tiles - len(union),
        "padding_rows_in_used_tiles": used_tiles * height - sum(counts.values()),
        "active_row_fraction": sum(counts.values()) / (used_tiles * height) if used_tiles else 0.0,
        "gemm_active_ctas": used_tiles * (2 * intermediate + hidden) // 128,
        "gemm_empty_ctas": (allocated_tiles - used_tiles) * (2 * intermediate + hidden) // 128,
        "per_request_distinct_experts": {k: len(v) for k, v in sorted(by_request.items())},
        "cross_request_duplicate_experts": sum(map(len, by_request.values())) - len(union),
        "anchor_experts": len(anchors), "experts_beyond_anchors": len(union - anchors),
        "request_pair_overlap": request_pairs, "payload": byte_cases,
        "other_tile_heights": {
            str(m): {"used_tiles": sum(ceildiv(c, m) for c in counts.values()),
                     "padding_rows": sum(ceildiv(c, m) * m - c for c in counts.values())}
            for m in (16, 32, 64)
        },
    }


def analyze(path):
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not records:
        raise ValueError("the trace contains no layer invocations")
    seen = set()
    for r in records:
        key = (r["capture"], r["step"], r["layer"])
        if key in seen:
            raise ValueError(f"duplicate layer invocation: {key}")
        seen.add(key)
    return {
        "schema_version": 1, "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_calls": len(records),
        "scope": "logical candidate-38 routed-weight and scale payloads; not measured DRAM/L2 traffic or timing",
        "policy_scope": "1030 changes only tile height; it does not simulate the full released runtime",
        "tp_projection": "same saved IDs evaluated at TP3/TP4 shard geometry; not cross-TP routing equivalence",
        "excluded": ["shared expert", "activation and dispatch traffic", "narrow-flag loads", "cache-line/sector overfetch"],
        "calls": [summarize(r, tp, policy) for r in records for tp in (3, 4) for policy in ("1030", "candidate-38")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
