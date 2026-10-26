#!/usr/bin/env python3
"""Fingerprint CED's decoder replay under the scheduler's chunkings, per-chunk against
prompt-end anchoring.

usage: anchor_sim.py NEXT_TREE ANCHOR_TREE [--json OUT]

NEXT_TREE is a vLLM checkout at next-candidate (the per-chunk anchor and the nonfinal
skip); ANCHOR_TREE one at ced-prompt-anchor. The planning functions load from each tree
through AST, without the GPU runtime: the kept rows and limits, the scheduler's aligned
chunks and, for next-candidate, the skip policy. The requests kernel's replay start and
the window metadata's SWA window are written out below, one line each.

A stacked decoder row depends on the rows its window sees, and on everything they saw
in turn. A prompt's result is its signature: the window start of every row in the
closure of the rows the first decode step and the drafter read (the last CED_WINDOW
rows), with None for a row no step kept. Two runs give the same decoder outputs when
their signatures match and no kept row's window reaches a row no step kept. The
reference replays the prompt's last CED_WINDOW rows in one pass from max(prefix,
prompt end - CED_WINDOW), as the V4.1 paper defines bounded replay.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from bisect import bisect_left
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

# The scheduler's chunk thresholds: budget - max_num_seqs x 6 (scripts/recipe.py).
THRESHOLDS = {
    "tp3-6144": 6144 - 8 * 6,
    "tp3-8192": 8192 - 8 * 6,
    "tp3-12288": 12288 - 8 * 6,
    "tp4-16384": 16384 - 16 * 6,
}


def load(tree: str, sources: dict) -> dict:
    ns = {"bisect_left": bisect_left, "dataclass": dataclass, "replace": replace,
          "Any": Any, "cdiv": lambda a, b: -(-a // b)}
    for path, names in sources.items():
        keep = []
        for node in ast.parse(open(os.path.join(tree, path)).read()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                keep.append(node)
            elif isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) in names for t in node.targets):
                keep.append(node)
            elif isinstance(node, ast.ClassDef):
                keep += [s for s in node.body if isinstance(s, ast.FunctionDef)
                         and (node.name, s.name) in names]
        found = {getattr(n, "name", None) or n.targets[0].id for n in keep}
        missing = {n if isinstance(n, str) else n[1] for n in names} - found
        assert not missing, f"{tree}/{path}: missing {missing}"
        for node in keep:
            for fn in [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]:
                fn.returns = None
                for a in fn.args.args + fn.args.kwonlyargs:
                    a.annotation = None
        exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


CED = "vllm/models/deepseek_v4_1/ced.py"
SCHEDULER = {"vllm/v1/core/sched/scheduler.py": {("Scheduler", "_aligned_prefill_chunk")}}


def chunks(ns, threshold: int, start: int, end: int) -> list[tuple[int, int]]:
    sched = SimpleNamespace(
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=threshold))
    out = []
    while start < end:
        rows = ns["_aligned_prefill_chunk"](sched, start, min(end - start, threshold))
        out.append((start, rows))
        start += rows
    return out


def next_candidate_rows(ns, threshold, prompt_end, prefix):
    """next-candidate's (kept, replay) per chunk: nonfinal_limits with the recipe's
    DSpark skip policy (k = 5, the 128-row windows), and its requests kernel's
    ``replay = max(prefix, end - kept if q > kept else 0)``."""
    W = ns["CED_WINDOW"]
    min_rows = ns["skip_min_rows"]
    for start, q in chunks(ns, threshold, prefix, prompt_end):
        limits = ns["nonfinal_limits"]([q], [start], [prompt_end], [False], [True],
                                       min_remaining=min_rows, chunk_align=threshold)
        plan = ns["plan_ced"]([q], [False], (1 << 31,), limits)
        kept = q if plan is None else min(q, W, (limits or [W])[0])
        end = start + q
        yield start, q, kept, max(prefix, end - kept if q > kept else 0)


def anchored_rows(ns, threshold, prompt_end, prefix):
    """ced-prompt-anchor's (kept, replay) per chunk: replay_rows, and its requests
    kernel's ``replay = min(floor, end - kept)``."""
    W = ns["CED_WINDOW"]
    for start, q in chunks(ns, threshold, prefix, prompt_end):
        floors, limits = ns["replay_rows"]([q], [start], [prompt_end], [prefix],
                                           [False])
        plan = ns["plan_ced"]([q], [False], (1 << 31,), limits)
        kept = q if plan is None else min(q, W, limits[0])
        end = start + q
        yield start, q, kept, min(floors[0], end - kept)


def windows(W, steps) -> dict[int, int]:
    """Each kept row's first visible row: the window metadata's SWA window from the
    step's replay start. A later step's kept row replaces an earlier one's."""
    lo = {}
    for start, q, kept, replay in steps:
        end = start + q
        for pos in range(end - kept, end):
            lo[pos] = max(pos + 1 - W, replay)
    return lo


def draft_unwritten(W, steps) -> int:
    """Steps whose draft window (the drafter's context rows are the kept rows, from
    ``max(end - W, replay)`` to the step's end) reaches a row no step kept yet."""
    kept, bad = set(), 0
    for start, q, n, replay in steps:
        end = start + q
        kept.update(range(end - n, end))
        bad += any(r not in kept for r in range(max(end - W, replay), end))
    return bad


def signature(lo, first, prompt_end):
    """(closure start, window starts over [closure start, prompt end), unwritten
    reads) of the rows from ``first`` on."""
    start = first
    while True:
        reach = min([start, *(lo[p] for p in range(start, prompt_end) if p in lo)])
        if reach == start:
            break
        start = reach
    rows = range(start, prompt_end)
    kept = [0]
    for p in rows:
        kept.append(kept[-1] + (p in lo))
    unwritten = sum(1 for p in rows if p in lo
                    and kept[p - start] - kept[lo[p] - start] != p - lo[p])
    return start, tuple(lo.get(p) for p in rows), unwritten


def main() -> int:
    next_tree, anchor_tree = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    old = load(next_tree, {CED: {"CED_WINDOW", "KEEP_ALL", "CEDPlan", "plan_ced",
                                 "nonfinal_limits"}, **SCHEDULER})
    new = load(anchor_tree, {CED: {"CED_WINDOW", "KEEP_ALL", "CEDPlan", "plan_ced",
                                   "replay_rows"}, **SCHEDULER})
    W = new["CED_WINDOW"]
    assert old["CED_WINDOW"] == W == 128
    # The recipe's skip policy (next-candidate's _nonfinal_skip_policy at k = 5):
    # reach 192 (the 133-wide draft window in 64s) + 2 + 5.
    old["skip_min_rows"] = 199
    report = {"window": W, "shapes": {}, "cross_shape": {}}
    bad = 0
    lengths = {}
    for name, T in THRESHOLDS.items():
        near = {k * T + r for k in (1, 2, 3) for r in range(-2 * W, 2 * W + 64)}
        lengths[name] = sorted({*range(1, 4 * W), *near})
    for name, T in THRESHOLDS.items():
        tally = dict(cases=0, prefix_cases=0, final_chunk_le_window=0,
                     next_differs=0, next_differs_final_over_window=0,
                     next_unwritten=0, next_draft_unwritten=0, anchored_differs=0,
                     anchored_unwritten=0, anchored_draft_unwritten=0,
                     anchored_steps_changed_final_over_window=0)
        examples = []
        for L in lengths[name]:
            hits = [h for h in range(64, L - W + 1, 64)
                    if any(abs(h - k * T) <= 2 * W for k in (1, 2, 3))]
            for H in [0, *hits[:: max(1, len(hits) // 6)]]:
                floor = max(H, L - W)
                ref = signature(windows(W, [(H, L - H, L - floor, floor)]), floor, L)
                a_steps = list(anchored_rows(new, T, L, H))
                n_steps = list(next_candidate_rows(old, T, L, H))
                anchored = signature(windows(W, a_steps), floor, L)
                per_chunk = signature(windows(W, n_steps), floor, L)
                final_short = len(n_steps) > 1 and n_steps[-1][1] <= W
                tally["cases"] += 1
                tally["prefix_cases"] += H > 0
                tally["final_chunk_le_window"] += final_short
                tally["anchored_unwritten"] += anchored[2] > 0
                tally["next_unwritten"] += per_chunk[2] > 0
                tally["anchored_draft_unwritten"] += draft_unwritten(W, a_steps) > 0
                tally["next_draft_unwritten"] += draft_unwritten(W, n_steps) > 0
                tally["anchored_differs"] += anchored != ref
                if per_chunk != ref:
                    tally["next_differs"] += 1
                    tally["next_differs_final_over_window"] += not final_short
                    if len(examples) < 4:
                        examples.append(dict(prompt=L, prefix=H, chunks=[
                            [s, q] for s, q, _, _ in n_steps]))
                # Bit-identical prompts: the final step's kept rows and replay start
                # are next-candidate's.
                if not final_short:
                    tally["anchored_steps_changed_final_over_window"] += (
                        a_steps[-1] != n_steps[-1])
        bad += tally["anchored_differs"] + tally["anchored_unwritten"]
        bad += tally["anchored_draft_unwritten"]
        bad += tally["anchored_steps_changed_final_over_window"]
        share = W / T
        report["shapes"][name] = dict(threshold=T, **tally, examples=examples,
                                      affected_share_of_long_prompts=share)
        print(f"{name} (threshold {T}): {tally}; prompts longer than one chunk with "
              f"a final chunk of 1-{W} rows: {100 * share:.2f}%", flush=True)
    # The same prompt under two shapes' chunkings: next-candidate's results differ
    # where either shape leaves a short final chunk; anchored results never do.
    names = list(THRESHOLDS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(lengths[a]) | set(lengths[b]))
            differ = {"next": 0, "anchored": 0}
            for L in common:
                floor = max(0, L - W)
                for tree, rows, key in ((old, next_candidate_rows, "next"),
                                        (new, anchored_rows, "anchored")):
                    sa = signature(windows(W, rows(tree, THRESHOLDS[a], L, 0)), floor, L)
                    sb = signature(windows(W, rows(tree, THRESHOLDS[b], L, 0)), floor, L)
                    differ[key] += sa != sb
            bad += differ["anchored"]
            report["cross_shape"][f"{a} vs {b}"] = dict(prompts=len(common), **differ)
            print(f"{a} vs {b}: {len(common)} prompts, results differ: {differ}",
                  flush=True)
    report["pass"] = bad == 0
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad})")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
