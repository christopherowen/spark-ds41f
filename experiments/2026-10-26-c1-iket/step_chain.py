#!/usr/bin/env python3
"""Each kernel's cost on a decode step's critical chain, from a step bundle's nsys
timeline (IKET's step tool, the run without IKET).

usage: step_chain.py NSYS_GRAPHS_SQLITE [--json OUT]

The timeline holds one prefill and the decode steps of one request; a step begins at
each ``prepare_pos_seq_lens`` launch, and the step that prepares prefill inputs is the
prefill. Within a decode step, a launch's cost is how far it moves the end of the
step's work so far: ``max(0, end - latest end before it)``, launches in start order on
every stream. That charges a kernel only for the time it alone keeps the GPU busy,
the critical chain's share, whatever stream it is on. Prints, per kernel name, the
launches a step, the summed duration and the summed cost a step (medians over the
decode steps), the most costly first, and the step's span.
"""
import argparse
import json
import sqlite3
import statistics
from collections import defaultdict


def launches(db):
    return db.execute(
        "SELECT k.start, k.end, k.streamId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()


def steps(rows):
    starts = [i for i, r in enumerate(rows) if r[3].startswith("prepare_pos_seq_lens")]
    out = []
    for a, b in zip(starts, starts[1:] + [len(rows)]):
        step = rows[a:b]
        if any(r[3].startswith("prepare_prefill_inputs") for r in step):
            continue
        out.append(step)
    return out


def costs(step):
    per = defaultdict(lambda: [0, 0.0, 0.0])  # launches, duration us, cost us
    latest = step[0][0]
    for start, end, _stream, name in step:
        name = name.removesuffix("_kernel")
        entry = per[name]
        entry[0] += 1
        entry[1] += (end - start) / 1000
        entry[2] += max(0, end - latest) / 1000
        latest = max(latest, end)
    span = (max(r[1] for r in step) - step[0][0]) / 1000
    return per, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--json")
    args = ap.parse_args()
    db = sqlite3.connect(args.sqlite)
    decode = steps(launches(db))
    tables = [costs(step) for step in decode]
    names = sorted({n for per, _ in tables for n in per})
    rows = []
    for name in names:
        vals = [per.get(name, [0, 0.0, 0.0]) for per, _ in tables]
        rows.append(dict(
            kernel=name,
            launches=statistics.median(v[0] for v in vals),
            duration_us=round(statistics.median(v[1] for v in vals), 1),
            cost_us=round(statistics.median(v[2] for v in vals), 1),
        ))
    rows.sort(key=lambda r: -r["cost_us"])
    span = statistics.median(s for _, s in tables)
    total = sum(r["cost_us"] for r in rows)
    print(f"{len(decode)} decode steps; median span {span:.1f} us; summed costs {total:.1f} us")
    print(f"{'kernel':48s} {'launches':>8s} {'duration':>10s} {'cost':>9s}")
    for r in rows:
        print(f"{r['kernel'][:48]:48s} {r['launches']:8g} {r['duration_us']:10.1f} {r['cost_us']:9.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(steps=len(decode), span_us=span, kernels=rows), f, indent=1)


if __name__ == "__main__":
    main()
