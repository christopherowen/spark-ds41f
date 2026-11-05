"""Bounds and a linearized-grid model; never hardware timing or cache counters."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from analyze_routes import ceildiv, projection, validate

PANELS = (0, 1, 2, 3, 4, 8)


def cta_order(nx, ny, panel):
    if nx < 1 or ny < 1 or panel < 0:
        raise ValueError("positive grid dimensions and a nonnegative panel required")
    for linear in range(nx * ny):
        if panel == 0:
            yield linear % nx, linear // nx
            continue
        group, within = divmod(linear, panel * nx)
        height = min(panel, ny - group * panel)
        col, row = divmod(within, height)
        yield (nx - 1 - col if group % 2 else col), group * panel + row


def distribution(values):
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    values = sorted(values)
    return {"count": len(values), "min": values[0],
            "median": statistics.median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}


def ordering_metrics(tile_experts, n, k, bits, panel):
    """Model only grid-linearized accesses, not the GPU's execution order.

    Keys identify one expert/N slab or one input M tile. The activation metric
    ignores partial row overlap between different expert tiles. Empty CTAs
    remain in the grid-index distances but issue no modeled weight requests.
    """
    nx = n // 128
    weight_last, input_last = {}, {}
    weight_distances, input_distances = [], []
    payload = projection(128, k, bits)["total"]
    active = 0
    for i, (x, y) in enumerate(cta_order(nx, len(tile_experts), panel)):
        expert = tile_experts[y]
        if expert < 0:
            continue
        active += 1
        key = expert, x
        if key in weight_last:
            weight_distances.append(i - weight_last[key])
        weight_last[key] = i
        if y in input_last:
            input_distances.append(i - input_last[y])
        input_last[y] = i
    return {"panel": panel, "active_ctas": active,
            "unique_weight_payload_bytes": len(weight_last) * payload,
            "requested_weight_payload_bytes": active * payload,
            "repeated_weight_payload_bytes": len(weight_distances) * payload,
            "weight_reuse_grid_index_distance": distribution(weight_distances),
            "same_input_tile_reuse_grid_index_distance": distribution(input_distances)}


def compare(record):
    experts, topk, hidden, rows = validate(record)
    counts = Counter(e for row in rows for e in row["ids"] if e >= 0)
    results = []
    for tp in (3, 4):
        intermediate = 2304 // tp
        for m in (16, 32, 64):
            tiles = [e for e, c in sorted(counts.items()) for _ in range(ceildiv(c, m))]
            used = len(tiles)
            allocated = min(len(rows) * topk, ceildiv(len(rows) * topk, m) + experts)
            tiles += [-1] * (allocated - used)
            useful_flops = 6 * hidden * intermediate * sum(counts.values())
            padded_flops = 6 * hidden * intermediate * used * m
            # Scratch intermediates only: dispatch and alignment excluded here.
            scratch_per_row = 5 * intermediate + 4 * ceildiv(intermediate, 128) + 2 * hidden
            geometry = {"tp": tp, "m": m, "used_tiles": used,
                        "allocated_tiles": allocated,
                        "padding_rows_in_used_tiles": used * m - sum(counts.values()),
                        "nominal_live_gemm_flops": useful_flops,
                        "nominal_issued_gemm_flops": padded_flops,
                        "gemm_intermediate_scratch_bytes": allocated * m * scratch_per_row}
            loads = {}
            for bits in (2, 4):
                for name, n, k in (("gate_up", 2 * intermediate, hidden),
                                   ("down", hidden, intermediate)):
                    loads[f"{name}_{bits}bit"] = [ordering_metrics(tiles, n, k, bits, panel)
                                                  for panel in PANELS]
            results.append({**geometry, "ordering": loads})
    return {"workload": record["workload"],
            "evidence_kind": record.get("evidence_kind", "recorded"),
            "call_rows": len(rows), "live_routes": sum(counts.values()),
            "distinct_experts": len(counts), "alternatives": results}


def scale_bounds():
    out = []
    for tp in (3, 4):
        intermediate = 2304 // tp
        values = {}
        for bits in (2, 4):
            parts = [projection(2 * intermediate, 5120, bits),
                     projection(5120, intermediate, bits)]
            values[bits] = {key: sum(p[key] for p in parts) for key in parts[0]}
        wide, narrow = values[4], values[2]
        out.append({"tp": tp, "per_expert_m_tile": values,
                    "all_4_to_all_2_payload_saving_ceiling":
                        (wide["total"] - narrow["total"]) / wide["total"],
                    "remove_all_scales_payload_saving_ceiling":
                        (wide["total"] - wide["fp4_weight"]) / wide["total"],
                    "remove_unused_4bit_tail_payload_saving_ceiling":
                        wide["unused_scale_code_bits_as_bytes"] / wide["total"]})
    return out


def synthetic_records():
    def record(name, ids, requests=None):
        return {"workload": name, "evidence_kind": "synthetic", "capture": "bounds",
                "step": 0, "layer": "target.0", "source_tree": "synthetic",
                "model_revision": "synthetic", "experts": 384, "topk": 6,
                "rows": [{"ids": v, "kind": "draft",
                          "request": requests[i] if requests else "r0"}
                         for i, v in enumerate(ids)]}
    yield record("synthetic-disjoint-48", [list(range(i * 6, i * 6 + 6)) for i in range(48)])
    yield record("synthetic-eight-independent-groups",
                 [list(range(i * 6, i * 6 + 6)) for i in range(8) for _ in range(6)],
                 [f"r{i}" for i in range(8) for _ in range(6)])
    yield record("synthetic-hot-17", [list(range(6)) for _ in range(17)])
    for rows in (2048, 8192):
        yield record(f"synthetic-balanced-{rows}",
                     [[(i * 6 + j) % 384 for j in range(6)] for i in range(rows)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {"scope": "synthetic mechanism bounds and linearized-grid ordering only",
              "not_measured": ["real workload overlap", "physical CTA execution order",
                               "DRAM/cache traffic", "latency", "throughput"],
              "scale_bounds": scale_bounds(),
              "cases": [compare(record) for record in synthetic_records()]}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
