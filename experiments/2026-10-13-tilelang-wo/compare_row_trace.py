#!/usr/bin/env python3
"""Where a prompt's prefill rows first differ between its own step and a shared one.

usage: compare_row_trace.py TRACE_DIR

Reads the step files of a VLLM_DS41_ROW_TRACE run (vLLM branch wo-row-trace, DEBUG
ONLY): per forward, every attention layer's row hashes at four stages, with the
rows' positions: "in" (the layer's input), "o" (attention output, before WO), "wo"
(this rank's WO partial, before the TP sum) and "ar" (after it). A prompt's prefill
is the run of rows at positions 0, 1, ... in a forward; prompts are told apart by
their first layer's input hashes. For each prompt prefilled both alone (the forward
holds only its rows) and beside other rows, prints the first (layer, stage) whose
hashes differ, in execution order, and the shared forward's size.

Standard library only.
"""
import glob
import os
import pickle
import sys


def prefill_run(positions):
    """(start, length) of the rows at positions 0, 1, ...; None if no row is at 0."""
    try:
        start = positions.index(0)
    except ValueError:
        return None
    length = 1
    while start + length < len(positions) and positions[start + length] == length:
        length += 1
    return start, length


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    trace = sys.argv[1]
    steps = []
    for path in sorted(glob.glob(os.path.join(trace, "step-*.pkl"))):
        records = load(path)
        first = records[0]
        run = prefill_run(first[2])
        if run is None:
            continue
        start, length = run
        key = tuple(first[3][start : start + length])
        steps.append((os.path.basename(path), key, len(first[2]), records))
    by_key = {}
    for name, key, rows, records in steps:
        by_key.setdefault(key, []).append((name, rows, records))
    print(f"{len(steps)} forwards with a prefill, {len(by_key)} distinct prompts")
    for key, group in by_key.items():
        alone = [g for g in group if g[1] == len(key)]
        shared = [g for g in group if g[1] != len(key)]
        if not alone or not shared:
            continue
        ref_name, _, ref = alone[0]
        for other_name, _, other in alone[1:]:
            if first_difference(ref, other) is not None:
                print(f"  prompt of {len(key)} rows: alone {ref_name} and {other_name} differ")
        for name, rows, records in shared:
            diff = first_difference(ref, records)
            meta = next((r[4] for r in records if r[1] == "o"), {})
            where = "identical" if diff is None else diff
            print(f"  prompt of {len(key)} rows: {name} ({rows} rows, sp={meta.get('sp')}) "
                  f"against {ref_name}: {where}")


def first_difference(ref, other):
    """The first (prefix, stage) whose prefill-run hashes differ, in ref's order."""
    index = {}
    for prefix, stage, positions, hashes, _ in other:
        index.setdefault((prefix, stage), []).append((positions, hashes))
    seen = {}
    for prefix, stage, positions, hashes, _ in ref:
        n = seen.get((prefix, stage), 0)
        seen[(prefix, stage)] = n + 1
        candidates = index.get((prefix, stage), [])
        if n >= len(candidates):
            return f"{prefix} {stage}: missing in the shared forward"
        run_a, run_b = prefill_run(positions), prefill_run(candidates[n][0])
        if run_a is None or run_b is None:
            continue
        a = hashes[run_a[0] : run_a[0] + run_a[1]]
        b = candidates[n][1][run_b[0] : run_b[0] + run_b[1]]
        if run_a[1] != run_b[1]:
            return f"{prefix} {stage}: run lengths {run_a[1]} vs {run_b[1]}"
        rows = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if rows:
            return f"{prefix} {stage}: {len(rows)} of {len(a)} rows differ (first at position {rows[0]})"
    return None


if __name__ == "__main__":
    main()
