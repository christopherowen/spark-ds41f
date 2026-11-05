#!/usr/bin/env python3
"""Offline cost sensitivity, never a serving throughput or policy qualification.

Use the historical decoder's validated trace parser and allocation, holding the
observed proposals fixed. Compare two cost objectives only at measured grid
cells: do not invent costs at graph or tile boundaries. The linear control is
also restricted to those cells; it is NOT the unrestricted serving policy.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/2026-09-26-dspark-policy/replay.py"
spec = importlib.util.spec_from_file_location("historical_replay", OLD)
historical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(historical)


def receipt(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def grid_cells(grid, concurrency, context):
    """No interpolation, extrapolation, cross-concurrency or context pooling."""
    cells = {}
    for entry in grid["engine"]:
        shape = entry["shape"]
        if (shape["kind"], shape["num_reqs"], shape["context_len"]) != (
            "decode", concurrency, context
        ):
            continue
        rows = shape["num_tokens"]
        if rows in cells:
            raise ValueError(f"duplicate grid cell: c{concurrency}, {rows} rows")
        cost = entry["step_ms"]["mean"]
        if not 0 < cost < float("inf"):
            raise ValueError("grid cost must be finite and positive")
        cells[rows] = cost
    return cells


def score(rows, cells, depth=5):
    """Identical eligible measured row counts and global ranking for both arms."""
    concurrency = len(rows)
    confidences = [conf for conf, _ in rows]
    expected = historical.expected_curve(confidences, depth)
    eligible = sorted(n for n in cells if concurrency <= n <= concurrency * (depth + 1))
    if not eligible:
        return None
    budgets = [n - concurrency for n in eligible]
    control = max(budgets, key=lambda k: expected[k] / (1 + .035 * (concurrency + k)))
    measured = max(budgets, key=lambda k: expected[k] / cells[concurrency + k])
    unconstrained = max(range(len(expected)), key=lambda k: expected[k] / (1 + .035 * (concurrency + k)))
    out = {"unrestricted_control_covered": concurrency + unconstrained in cells}
    for name, budget in (("linear035_at_measured_cells", control), ("grid_at_measured_cells", measured)):
        counts = historical.allocate(confidences, budget, depth)
        out[name] = {
            "draft_rows": budget,
            "matched_tokens": sum(1 + min(match, count) for (_, match), count in zip(rows, counts)),
            "expected_tokens": float(expected[budget]),
            "model_ms": cells[concurrency + budget],
        }
    return out


def analyze(trace_path, grid_paths, context=4096):
    raw = historical.load_trace(trace_path)
    # Startup dummy records are not user requests and may reuse IDs.
    raw = [[r for r in step if not r["r"].startswith("_warmup")] for step in raw]
    raw = [step for step in raw if step]
    for step in raw:
        if len(step) != step[0]["b"] or len({r["r"] for r in step}) != len(step):
            raise ValueError("incomplete or duplicate request records in a step")
        for record in step:
            if len(record.get("c", ())) != 5 or len(record["d"]) != 5:
                raise ValueError("this historical screen requires five observed drafts and confidences")
            if any(not 0 <= c <= 1 for c in record["c"]):
                raise ValueError("invalid confidence")
    streams, anchors = historical.committed_streams(raw)
    agree, total = historical.check_acceptance(raw, streams, anchors)
    if agree != total or total == 0:
        raise ValueError(f"observed acceptance does not replay: {agree}/{total}")
    # Drop terminal proposals when their matching prefix extends beyond the
    # recorded output; never score unobserved tokens as rejected drafts.
    steps = historical.decode_steps(raw, streams, anchors, 5, 1, 16)
    counts = Counter(len(step) for step in steps)
    output = {
        "schema": "verification-budget-cell-screen/1",
        "evidence_kind": "historical-trace-cross-source-cost-sensitivity",
        "performance_verdict": "unsupported",
        "trace": receipt(trace_path),
        "historical_decoder": receipt(OLD.relative_to(ROOT)),
        "acceptance_replay": {"matched": agree, "checked": total},
        "observed_steps": len(raw),
        "scorable_steps": len(steps),
        "excluded_steps": len(raw) - len(steps),
        "scorable_steps_by_concurrency": dict(sorted(counts.items())),
        "limitations": [
            "The trace is r4a/2026-09-26; current source, checkpoint arithmetic and workloads differ.",
            "Both objectives are restricted to measured cells; neither is the unrestricted current policy.",
            "Observed fresh confidences are used for both objectives; stale host scheduling and future batch trajectories are not replayed.",
            "The grid uses dummy inputs and full live rows; no fresh dead-row or expert-union cost is available.",
            "The trace rounds confidences to five decimal places; this is not a bit-exact GPU allocator replay.",
            "No sampled, copy, tool/grammar, current agent-traffic or c16 confidence trace is present.",
            "No policy boot or repeat boot was run. Grid confidence intervals do not quantify the counterfactual error.",
        ],
        "grids": [],
    }
    for grid_path in grid_paths:
        grid = json.loads(Path(grid_path).read_text())
        result = {"input": receipt(grid_path), "label": grid.get("label"), "context": context, "concurrency": {}}
        for concurrency in (1, 4, 8, 16):
            cells = grid_cells(grid, concurrency, context)
            eligible_steps = [s for s in steps if len(s) == concurrency]
            scored = [score(s, cells) for s in eligible_steps]
            scored = [s for s in scored if s is not None]
            item = {"measured_rows": sorted(cells), "trace_steps": len(eligible_steps), "scored_steps": len(scored)}
            result["concurrency"][str(concurrency)] = item
            if not scored:
                item["status"] = "missing trace or grid"
                continue
            item["unrestricted_linear_control_covered_steps"] = sum(s["unrestricted_control_covered"] for s in scored)
            for policy in ("linear035_at_measured_cells", "grid_at_measured_cells"):
                totals = {k: sum(s[policy][k] for s in scored) for k in scored[0][policy]}
                totals["modeled_tokens_per_ms"] = totals["matched_tokens"] / totals["model_ms"]
                item[policy] = totals
            item["modeled_rate_delta_pct"] = 100 * (item["grid_at_measured_cells"]["modeled_tokens_per_ms"] / item["linear035_at_measured_cells"]["modeled_tokens_per_ms"] - 1)
        output["grids"].append(result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--grid", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(analyze(args.trace, args.grid), indent=2) + "\n")
