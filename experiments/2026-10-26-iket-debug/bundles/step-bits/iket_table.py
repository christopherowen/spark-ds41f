#!/usr/bin/env python3
"""A per-phase table from an IKET trace (run-iket's ``*.trace.json``).

usage: iket_table.py TRACE.json|RUN_DIR [--sources DIR] [--nsys NAME=FILE.sqlite ...]
                     [--kernels REGEX] [--between MARKER_KERNEL] [--json OUT.json]

Launches (those inside CUDA graph launches too) are ordered by their first warp's
start; ``--between`` keeps the ones between the first and last launch of a marker
kernel (iket_step.py brackets its measured request with ``iket_step_marker``).

An overview comes first: kernel shapes by total span, with the phase split of their
launches' last warps, and the waits those last warps spend most time in, summed over
the launches. A launch ends when its last warp does, so these are the waits that
lengthen the step.

Then launches are grouped by kernel name, grid and block. For each group:

- ``span``: first warp start to last warp end of a launch (median over launches),
  which is what nsys reports as the kernel's duration less its launch latency;
- ``warp``: a warp's lifetime (mean over warps, median over launches);
- per range: the time a warp spends in it (mean over all warps of a launch,
  warps that never enter it counting zero; median over launches), its share of
  the warp lifetime, the ranges a warp enters, and the longest warp's time in it;
- per phase (the range name before the dot): the same summed, plus
  ``unattributed`` (lifetime less every range), and the phase split of the warp
  that ends last, the one the launch waits for.

``--sources`` is the IKET mode's ``sources`` directory: each range's first
statement is printed beside it. ``--nsys`` adds, per group, the median kernel
duration of nsys exports (``nsys export --type sqlite``) of the same kernel
(name, grid and block); NAME labels the column, e.g. ``off=nsys-off.sqlite``.

Prints Markdown; ``--json`` also writes the numbers.
"""

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else 0.0


def phase_of(name):
    return name.split("#")[0].split(".")[0]


def sites(directory):
    """(kernel, range) -> first statement, from the IKET mode's site manifests."""
    found = defaultdict(set)
    for path in sorted(Path(directory).glob("*.json")):
        manifest = json.loads(path.read_text())
        for site in manifest["sites"]:
            if "range" in site:
                found[(site["kernel"], site["range"])].add(site["statement"])
    return {key: " | ".join(sorted(statements)) for key, statements in found.items()}


def nsys_durations(path):
    """(kernel, grid, block) -> kernel durations in us, from an nsys SQLite export."""
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.end - k.start "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName"
    ).fetchall()
    out = defaultdict(list)
    for name, gx, gy, gz, bx, by, bz, ns in rows:
        out[(name, (gx, gy, gz), (bx, by, bz))].append(ns / 1000)
    return out


def launch_numbers(launch, names, locations):
    """One launch: span, warp lifetimes and each warp's time per range."""
    lifetimes = {w["locIdx"]: (w["startTs"], w["endTs"]) for w in launch["warpLifetimes"]}
    if not lifetimes:
        return None
    start = min(s for s, _ in lifetimes.values())
    end = max(e for _, e in lifetimes.values())
    per_warp = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for r in launch["ranges"]:
        warp = r["warpLocIdxs"][0]
        name = names[r["rangeNameIdx"]]
        per_warp[warp][name] += (r["endTs"] - r["startTs"]) / 1000
        counts[warp][name] += 1
    warps = list(lifetimes)
    life = {w: (lifetimes[w][1] - lifetimes[w][0]) / 1000 for w in warps}
    last = max(warps, key=lambda w: lifetimes[w][1])
    return {
        "span": (end - start) / 1000,
        "warps": len(warps),
        "life": sum(life.values()) / len(warps),
        "ranges": {
            name: {
                "mean": sum(per_warp[w].get(name, 0.0) for w in warps) / len(warps),
                "max": max(per_warp[w].get(name, 0.0) for w in warps),
                "count": sum(counts[w].get(name, 0) for w in warps) / len(warps),
            }
            for name in {n for w in warps for n in per_warp[w]}
        },
        "last": {name: per_warp[last].get(name, 0.0) for name in per_warp[last]},
        "last_life": life[last],
        "ctas": len({tuple(locations[w]["ctaId"]) for w in warps}),
    }


def all_launches(trace):
    """The trace's launches, those of CUDA graph launches included, by first warp start."""
    launches = list(trace["launches"])
    graphs = trace.get("graphLaunches") or {}
    for value in graphs.values() if isinstance(graphs, dict) else graphs:
        items = value if isinstance(value, list) else value.get("launches", []) if isinstance(value, dict) else []
        launches.extend(item for item in items if isinstance(item, dict) and "kernelName" in item)
    launches = [launch for launch in launches if launch.get("warpLifetimes")]
    return sorted(launches, key=lambda launch: min(w["startTs"] for w in launch["warpLifetimes"]))


def window(launches, marker):
    """The launches strictly between the first and the last launch of ``marker``."""
    marks = [i for i, launch in enumerate(launches) if launch["kernelName"] == marker]
    if len(marks) < 2:
        raise SystemExit(f"{marker}: {len(marks)} launches in the trace, need two")
    return launches[marks[0] + 1 : marks[-1]]


def summarize(trace, kernels=None, between=None):
    names, locations = trace["stringTable"], trace["locationTable"]
    groups = defaultdict(list)
    launches = all_launches(trace)
    if between:
        launches = window(launches, between)
    for launch in launches:
        if kernels and not re.search(kernels, launch["kernelName"]):
            continue
        key = (launch["kernelName"],
               (launch["gridDimX"], launch["gridDimY"], launch["gridDimZ"]),
               (launch["blockDimX"], launch["blockDimY"], launch["blockDimZ"]))
        numbers = launch_numbers(launch, names, locations)
        if numbers is not None:
            groups[key].append(numbers)
    out = []
    for (kernel, grid, block), launches in groups.items():
        life = median([n["life"] for n in launches])
        range_names = sorted({name for n in launches for name in n["ranges"]})
        ranges = {}
        for name in range_names:
            mean = median([n["ranges"].get(name, {}).get("mean", 0.0) for n in launches])
            ranges[name] = {
                "us": mean,
                "share": mean / life if life else 0.0,
                "count": median([n["ranges"].get(name, {}).get("count", 0.0) for n in launches]),
                "max_us": median([n["ranges"].get(name, {}).get("max", 0.0) for n in launches]),
            }
        phases = defaultdict(float)
        last = defaultdict(float)
        for name, r in ranges.items():
            phases[phase_of(name)] += r["us"]
        # The last warp's split and lifetime are means over the same launches.
        for n in launches:
            for name, us in n["last"].items():
                last[phase_of(name)] += us / len(launches)
        last_life = sum(n["last_life"] for n in launches) / len(launches)
        phases["unattributed"] = max(0.0, life - sum(phases.values()))
        last["unattributed"] = max(0.0, last_life - sum(last.values()))
        last_ranges = defaultdict(float)
        for n in launches:
            for name, us in n["last"].items():
                last_ranges[name] += us
        out.append({
            "kernel": kernel, "grid": grid, "block": block, "launches": len(launches),
            "warps": launches[0]["warps"], "ctas": launches[0]["ctas"],
            "span_us": median([n["span"] for n in launches]), "warp_us": life,
            "span_total_us": sum(n["span"] for n in launches),
            "ranges": ranges, "phases": dict(phases),
            "last_warp": {"us": last_life, "phases": dict(last), "ranges_total_us": dict(last_ranges)},
        })
    return sorted(out, key=lambda g: -g["span_total_us"])


def overview(groups, statements, top=20):
    """Markdown: kernels by their total span, and the waits the launches' last warps spend
    most time in (a launch ends with its last warp)."""
    total = sum(g["span_total_us"] for g in groups)
    lines = [f"\n## Overview: {sum(g['launches'] for g in groups)} launches of {len(groups)} kernel "
             f"shapes, spans summing to {total:.1f} us\n",
             "| kernel | grid | launches | span total us | share | last warp: wait | load | compute "
             "| store | unattributed |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups[:top]:
        last = g["last_warp"]
        shares = [last["phases"].get(p, 0.0) / last["us"] if last["us"] else 0.0
                  for p in ("wait", "load", "compute", "store", "unattributed")]
        lines.append(f"| {g['kernel']} | {g['grid']} | {g['launches']} | {g['span_total_us']:.1f} | "
                     f"{g['span_total_us'] / total:.1%} | " + " | ".join(f"{s:.0%}" for s in shares) + " |")
    waits = []
    for g in groups:
        for name, us in g["last_warp"]["ranges_total_us"].items():
            if phase_of(name) == "wait":
                waits.append((us, g, name))
    lines += ["\n| wait | kernel | grid | last-warp us, all launches | per launch | first statement |",
              "|---|---|---|---:|---:|---|"]
    for us, g, name in sorted(waits, key=lambda item: -item[0])[:top]:
        statement = statements.get((g["kernel"], name), "")[:80].replace("|", "\\|")
        lines.append(f"| {name} | {g['kernel']} | {g['grid']} | {us:.1f} | {us / g['launches']:.2f} | "
                     f"`{statement}` |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--sources")
    parser.add_argument("--nsys", nargs="*", default=[])
    parser.add_argument("--kernels")
    parser.add_argument("--between")
    parser.add_argument("--json")
    args = parser.parse_args()
    trace = Path(args.trace)
    if trace.is_dir():  # run-iket's output directory: its largest trace
        trace = max(trace.glob("*.trace.json"), key=lambda p: p.stat().st_size)
    data = json.loads(trace.read_text())
    graphs = data.get("graphLaunches")
    print(f"trace {trace.name}: {len(data['launches'])} launches, graphLaunches "
          f"{type(graphs).__name__} of {len(graphs or [])}")
    groups = summarize(data, args.kernels, args.between)
    statements = sites(args.sources) if args.sources else {}
    nsys = {}
    for item in args.nsys:
        label, path = item.split("=", 1)
        nsys[label] = nsys_durations(path)
    for g in groups:
        key = (g["kernel"], tuple(g["grid"]), tuple(g["block"]))
        g["nsys_us"] = {label: median(d.get(key, [])) for label, d in nsys.items()}
        g["nsys_launches"] = {label: len(d.get(key, [])) for label, d in nsys.items()}
    lines = overview(groups, statements)
    for g in groups:
        nsys_text = "".join(f", nsys {label} {us:.2f} us" for label, us in g["nsys_us"].items())
        lines.append(f"\n### {g['kernel']} grid {g['grid']} block {g['block']}\n")
        lines.append(f"{g['launches']} launches, {g['ctas']} CTAs, {g['warps']} warps; "
                     f"span {g['span_us']:.2f} us, warp {g['warp_us']:.2f} us{nsys_text}\n")
        lines.append("| phase | us a warp | share | last warp us | last warp share |")
        lines.append("|---|---:|---:|---:|---:|")
        last = g["last_warp"]
        for phase in sorted(g["phases"], key=lambda p: -g["phases"][p]):
            us = g["phases"][phase]
            lu = last["phases"].get(phase, 0.0)
            lines.append(f"| {phase} | {us:.2f} | {us / g['warp_us']:.1%} | {lu:.2f} | "
                         f"{lu / last['us'] if last['us'] else 0:.1%} |")
        lines.append("\n| range | us a warp | share | entries | longest warp us | first statement |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, r in sorted(g["ranges"].items(), key=lambda item: -item[1]["us"]):
            statement = statements.get((g["kernel"], name), "")[:90].replace("|", "\\|")
            lines.append(f"| {name} | {r['us']:.2f} | {r['share']:.1%} | {r['count']:.1f} | "
                         f"{r['max_us']:.2f} | `{statement}` |")
    print("\n".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps(groups, indent=1) + "\n")


if __name__ == "__main__":
    main()
