#!/usr/bin/env python3
"""What a request's decode steps add to the engine grid's step, kernel by kernel.

usage: step_diff.py TRACE.sqlite|TRACE.json --grid-runs N [--table TABLE.json]
                    [--step-kernel K] [--prefill-kernel K] [--top N] [--json OUT]

From either an nsys export (``*.sqlite``) or a ``run-iket --warp-trace`` decoded JSON
(``*.json`` / ``iket.decoded_results.json``) of ``iket_step.py off --grid N``:
- the request's decode steps: the launches between the two ``iket_step_marker`` launches,
  split at each ``--step-kernel`` launch (prepare_pos_seq_lens_kernel), the steps that
  prepare prefill inputs left out;
- the grid's runs: the launches between the two ``iket_grid_marker`` launches, ``N`` runs
  (one warm-up and the timed ones), the cost profile's dummy step at the request's shape.

For each kernel name: launches and GPU time a step in each, their difference, and its
exposed time a step in the request (time it is the only kernel on the GPU, so on the
step's critical path; time it shares with another kernel is overlapped). The totals are
the GPU's busy time a step (the union of its kernels) and the kernels' summed time.

``--table`` is iket_table.py's ``--json`` of the IKET trace of the same steps: each
kernel's last-warp phase split, and the range its last warp waits in most, beside it.
"""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def load_rows(path):
    """Return sorted ``(start_ns, end_ns, kernel_name)`` rows from either an nsys SQLite
    database or a run-iket decoded JSON file."""
    p = Path(path)
    if p.suffix == ".json" or p.name.endswith(".json"):
        data = json.loads(p.read_text())
        launches = list(data.get("launches") or [])
        graphs = data.get("graphLaunches") or {}
        for value in graphs.values() if isinstance(graphs, dict) else graphs:
            items = value if isinstance(value, list) else value.get("launches", []) if isinstance(value, dict) else []
            launches.extend(item for item in items if isinstance(item, dict) and "kernelName" in item)
        rows = [
            (min(w["startTs"] for w in l["warpLifetimes"]),
             max(w["endTs"] for w in l["warpLifetimes"]),
             l["kernelName"])
            for l in launches if l.get("warpLifetimes")
        ]
        rows.sort(key=lambda r: r[0])
        return rows
    return sqlite3.connect(str(p)).execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
    ).fetchall()


def window(rows, marker):
    marks = [i for i, row in enumerate(rows) if row[2] == marker]
    if len(marks) < 2:
        raise SystemExit(f"{marker}: {len(marks)} launches, need two")
    return rows[marks[0] + 1 : marks[-1]]


def decode_steps(rows, step_kernel, prefill_kernel):
    starts = [i for i, row in enumerate(rows) if row[2] == step_kernel]
    if not starts:
        raise SystemExit(f"{step_kernel}: not launched between the request's markers")
    starts[0] = 0  # the window opens on the first step (its prefill inputs come first)
    steps = [rows[a:b] for a, b in zip(starts, starts[1:] + [len(rows)])]
    return [step for step in steps if not any(row[2] == prefill_kernel for row in step)]


def busy_and_exposed(rows):
    """The union of the launches' spans, and each kernel's time alone on the GPU."""
    events = sorted([(start, 1, i) for i, (start, _, _) in enumerate(rows)]
                    + [(end, -1, i) for i, (_, end, _) in enumerate(rows)])
    active, exposed, busy, last = set(), defaultdict(float), 0, None
    for time, kind, index in events:
        if last is not None and active:
            busy += time - last
            if len(active) == 1:
                exposed[rows[next(iter(active))][2]] += time - last
        (active.add if kind == 1 else active.discard)(index)
        last = time
    return busy, exposed


def per_step(steps):
    """Each kernel's launches, time and exposed time a step, and the step's busy and
    summed kernel time, as means over ``steps`` (lists of (start, end, name))."""
    launches, time, exposed = defaultdict(float), defaultdict(float), defaultdict(float)
    busy_total = 0
    for step in steps:
        busy, alone = busy_and_exposed(step)
        busy_total += busy
        for start, end, name in step:
            launches[name] += 1
            time[name] += end - start
        for name, ns in alone.items():
            exposed[name] += ns
    n = len(steps)
    return {
        "steps": n,
        "busy_us": busy_total / n / 1000,
        "kernel_us": sum(time.values()) / n / 1000,
        "kernels": {name: {"launches": launches[name] / n, "us": time[name] / n / 1000,
                           "exposed_us": exposed[name] / n / 1000} for name in time},
    }


def phases(table_path):
    """kernel name -> (last-warp phase shares, the wait range with most last-warp time),
    from the kernel shape with the most span."""
    groups = json.load(open(table_path))["groups"]
    out = {}
    for g in sorted(groups, key=lambda g: g["span_total_us"]):
        last = g["last_warp"]
        shares = {p: us / last["us"] for p, us in last["phases"].items() if last["us"]}
        waits = {name: us for name, us in last["ranges_total_us"].items() if name.startswith("wait")}
        out[g["kernel"]] = (shares, max(waits, key=waits.get) if waits else "")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="nsys SQLite export (*.sqlite) or run-iket decoded JSON (*.json)")
    parser.add_argument("--grid-runs", type=int, required=True)
    parser.add_argument("--table")
    parser.add_argument("--step-kernel", default="prepare_pos_seq_lens_kernel")
    parser.add_argument("--prefill-kernel", default="prepare_prefill_inputs_kernel")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json")
    args = parser.parse_args()
    rows = load_rows(args.trace)
    request = per_step(decode_steps(window(rows, "iket_step_marker_kernel"), args.step_kernel,
                                    args.prefill_kernel))
    grid_rows = window(rows, "iket_grid_marker_kernel")
    grid = per_step([grid_rows])
    for item in grid["kernels"].values():
        for key in ("launches", "us", "exposed_us"):
            item[key] /= args.grid_runs
    for key in ("busy_us", "kernel_us"):
        grid[key] /= args.grid_runs
    grid["steps"] = args.grid_runs
    split = phases(args.table) if args.table and Path(args.table).exists() else {}
    if args.table and not split:
        print(f"no phase split: {args.table} is missing", flush=True)
    names = set(request["kernels"]) | set(grid["kernels"])
    empty = {"launches": 0, "us": 0.0, "exposed_us": 0.0}
    diff = sorted(((request["kernels"].get(n, empty), grid["kernels"].get(n, empty), n) for n in names),
                  key=lambda item: -(item[0]["us"] - item[1]["us"]))
    lines = [f"## A decode step against the grid's step ({request['steps']} decode steps, "
             f"{grid['steps']} grid runs)\n",
             f"GPU busy a step: {request['busy_us']:.1f} us against {grid['busy_us']:.1f} us "
             f"(+{request['busy_us'] - grid['busy_us']:.1f}); kernel time a step: "
             f"{request['kernel_us']:.1f} us against {grid['kernel_us']:.1f} us "
             f"(+{request['kernel_us'] - grid['kernel_us']:.1f}).\n",
             "| kernel | launches a step | grid | us a step | grid | difference us | exposed us "
             "| last warp: wait / load / compute / store / other | longest last-warp wait |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for mine, theirs, name in diff[: args.top] + [d for d in diff[args.top:] if d[0]["us"] - d[1]["us"] < -50]:
        shares, wait = split.get(name, ({}, ""))
        text = " / ".join(f"{shares.get(p, 0):.0%}" for p in ("wait", "load", "compute", "store", "unattributed"))
        lines.append(f"| {name} | {mine['launches']:g} | {theirs['launches']:g} | {mine['us']:.1f} | "
                     f"{theirs['us']:.1f} | {mine['us'] - theirs['us']:+.1f} | {mine['exposed_us']:.1f} | "
                     f"{text if shares else ''} | {wait} |")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as out:
            json.dump({"request": request, "grid": grid}, out, indent=1)


if __name__ == "__main__":
    main()
