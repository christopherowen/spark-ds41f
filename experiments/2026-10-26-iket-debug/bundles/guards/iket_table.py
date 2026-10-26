#!/usr/bin/env python3
"""A per-phase table from an IKET trace (run-iket's ``*.trace.json``).

usage: iket_table.py TRACE.json|RUN_DIR [--sources DIR] [--nsys NAME=FILE.sqlite ...]
                     [--kernels REGEX] [--between MARKER_KERNEL [--gap-us US]]
                     [--json OUT.json]

Launches (those inside CUDA graph launches too) are ordered by their first warp's
start; ``--between`` keeps the ones between the first and last launch of a marker
kernel (iket_step.py brackets its measured request with ``iket_step_marker``), and
starts with their timeline: wall time, the GPU's busy time (the union of the launches'
spans) and idle gaps, and the segments that gaps over ``--gap-us`` (100) separate.

An overview follows: kernel shapes by total span, with the phase split of their
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
statement is printed beside it, and the manifests say how a kernel was sampled. For a
kernel with sampled CTAs (``VLLM_DS41_IKET_CTAS``), warp, range and last-warp numbers
come from the CTAs that recorded ranges (the span stays the whole launch's). With
sampled iterations (``VLLM_DS41_IKET_ITERS``), a range inside a loop holds only its
sampled iterations; it is marked, and the loop's other iterations count as
unattributed. ``--nsys`` adds, per group, the median kernel
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
    """From the IKET mode's site manifests: (kernel, range) -> first statement, and
    kernel -> its sampling: CTAs and iterations (0: all) and the ranges inside a loop."""
    found = defaultdict(set)
    sampling = {}
    for path in sorted(Path(directory).glob("*.json")):
        manifest = json.loads(path.read_text())
        kernel = sampling.setdefault(manifest["kernel"], {
            "ctas": manifest.get("ctas", 0), "iters": manifest.get("iters", 0), "in_loops": set()})
        for site in manifest["sites"]:
            if "range" in site:
                found[(site["kernel"], site["range"])].add(site["statement"])
                if site.get("loops"):
                    kernel["in_loops"].add(site["range"])
    return {key: " | ".join(sorted(statements)) for key, statements in found.items()}, sampling


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


def launch_numbers(launch, names, locations, sampled=False):
    """One launch: span, warp lifetimes and each warp's time per range. With sampled CTAs,
    lifetimes, ranges and the last warp are those of the CTAs that recorded ranges; the
    span is always the whole launch's."""
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
    if sampled and per_warp:
        recording = {tuple(locations[w]["ctaId"]) for w in per_warp}
        warps = [w for w in warps if tuple(locations[w]["ctaId"]) in recording]
    life = {w: (lifetimes[w][1] - lifetimes[w][0]) / 1000 for w in warps}
    last = max(warps, key=lambda w: lifetimes[w][1])
    return {
        "span": (end - start) / 1000,
        "start": start,
        "end": end,
        "all_warps": len(lifetimes),
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


def summarize(launches, names, locations, kernels=None, sampling=None):
    """Groups of launches by kernel, grid and block. ``sampling`` is the manifests'
    (``sites``): a CTA-sampled kernel's numbers come from its recording CTAs, and a range
    inside a loop of an iteration-sampled kernel holds the sampled iterations only."""
    sampling = sampling or {}
    groups = defaultdict(list)
    for launch in launches:
        if kernels and not re.search(kernels, launch["kernelName"]):
            continue
        key = (launch["kernelName"],
               (launch["gridDimX"], launch["gridDimY"], launch["gridDimZ"]),
               (launch["blockDimX"], launch["blockDimY"], launch["blockDimZ"]))
        sampled = sampling.get(launch["kernelName"], {}).get("ctas", 0) > 0
        numbers = launch_numbers(launch, names, locations, sampled)
        if numbers is not None:
            groups[key].append(numbers)
    out = []
    for (kernel, grid, block), launches in groups.items():
        kernel_sampling = sampling.get(kernel, {})
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
                "sampled_iterations": bool(kernel_sampling.get("iters"))
                and name in kernel_sampling.get("in_loops", ()),
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
            "all_warps": launches[0]["all_warps"],
            "sampled_ctas": kernel_sampling.get("ctas", 0), "sampled_iters": kernel_sampling.get("iters", 0),
            "span_us": median([n["span"] for n in launches]), "warp_us": life,
            "span_total_us": sum(n["span"] for n in launches),
            "ranges": ranges, "phases": dict(phases),
            "last_warp": {"us": last_life, "phases": dict(last), "ranges_total_us": dict(last_ranges)},
        })
    return sorted(out, key=lambda g: -g["span_total_us"])


def find_trace(path):
    """A trace file, or run-iket's output directory: its largest ``*.trace.json`` (the json
    postprocess), else its decoded records (``iket/pid_*/iket.decoded_results.json``, the
    same format: the json postprocess copies it)."""
    path = Path(path)
    if not path.is_dir():
        return path
    found = list(path.glob("*.trace.json")) or list(path.glob("iket/pid_*/iket.decoded_results.json"))
    if not found:
        raise SystemExit(f"{path}: no *.trace.json or iket/pid_*/iket.decoded_results.json")
    return max(found, key=lambda p: p.stat().st_size)


def engine_steps(launches, step_kernel, prefill_kernel):
    """The launches split at each launch of ``step_kernel`` (once an engine step), as
    (kind, launches): "prefill" where the step launches ``prefill_kernel``, else
    "decode". Launches before the first ``step_kernel`` are left out."""
    starts = [i for i, launch in enumerate(launches) if launch["kernelName"] == step_kernel]
    if not starts:
        raise SystemExit(f"{step_kernel}: not launched in the window")
    out = []
    for a, b in zip(starts, starts[1:] + [len(launches)]):
        part = launches[a:b]
        kind = "prefill" if any(launch["kernelName"] == prefill_kernel for launch in part) else "decode"
        out.append((kind, part))
    return out


def occupancy(items):
    """(wall us, busy us) of (start ns, end ns, kernel) items in start order; busy is the
    union of their spans."""
    busy, run_start, run_end = 0, items[0][0], items[0][1]
    for start, end, _ in items[1:]:
        if start > run_end:
            busy += run_end - run_start
            run_start, run_end = start, end
        else:
            run_end = max(run_end, end)
    busy += run_end - run_start
    return (max(end for _, end, _ in items) - items[0][0]) / 1000, busy / 1000


def step_table(parts):
    """Markdown and numbers: each engine step's launches, wall and busy time, and its
    kernels with the most span."""
    lines = ["\n## Engine steps\n", "| step | kind | launches | wall us | busy us | busy | most span |",
             "|---:|---|---:|---:|---:|---:|---|"]
    numbers = []
    for i, (kind, part) in enumerate(parts):
        items = spans(part)
        wall, busy = occupancy(items)
        total = defaultdict(float)
        for start, end, name in items:
            total[name] += (end - start) / 1000
        top = sorted(total, key=lambda n: -total[n])[:3]
        numbers.append({"kind": kind, "launches": len(part), "wall_us": wall, "busy_us": busy,
                        "most_span": {n: total[n] for n in top}})
        lines.append(f"| {i} | {kind} | {len(part)} | {wall:.1f} | {busy:.1f} | "
                     f"{busy / wall if wall else 1:.0%} | "
                     + ", ".join(f"{n} {total[n]:.0f}" for n in top) + " |")
    return lines, numbers


def spans(launches):
    """(start ns, end ns, kernel) of each IKET launch: its first warp start, last warp end."""
    return [(min(w["startTs"] for w in launch["warpLifetimes"]),
             max(w["endTs"] for w in launch["warpLifetimes"]), launch["kernelName"])
            for launch in launches]


def timeline(items, gap_us, top=10):
    """Markdown and numbers: the GPU time of ``items``, (start ns, end ns, kernel) in
    start order. Busy is the union of their spans; idle gaps longer than ``gap_us``
    split it into segments (in an engine run, mostly its steps, between which the host
    schedules)."""
    busy, gaps = 0, []
    run_start, run_end, run_last = items[0]
    segments = [{"launches": 0, "start": run_start, "busy": 0, "first": run_last}]
    for start, end, name in items:
        if start > run_end:
            busy += run_end - run_start
            segments[-1]["busy"] += run_end - run_start
            gaps.append(((start - run_end) / 1000, run_last, name))
            if (start - run_end) / 1000 > gap_us:
                segments[-1]["end"] = run_end
                segments.append({"launches": 0, "start": start, "busy": 0, "first": name})
            run_start, run_end, run_last = start, end, name
        elif end > run_end:
            run_end, run_last = end, name
        segments[-1]["launches"] += 1
    busy += run_end - run_start
    segments[-1]["busy"] += run_end - run_start
    segments[-1]["end"] = run_end
    wall = (run_end - items[0][0]) / 1000
    idle = sum(g for g, _, _ in gaps)
    numbers = {
        "launches": len(items), "wall_us": wall, "busy_us": busy / 1000, "idle_us": idle,
        "gaps": len(gaps), "gap_us": gap_us,
        "segments": [{"launches": s["launches"], "wall_us": (s["end"] - s["start"]) / 1000,
                      "busy_us": s["busy"] / 1000, "first": s["first"]} for s in segments],
        "largest_gaps": [{"us": g, "after": a, "before": b}
                         for g, a, b in sorted(gaps, key=lambda item: -item[0])[:top]],
    }
    lines = [f"\n## Timeline: {len(items)} launches in {wall:.1f} us; the GPU is busy "
             f"{busy / 1000:.1f} us ({busy / 1000 / wall:.0%}) and idle {idle:.1f} us in "
             f"{len(gaps)} gaps; gaps over {gap_us:g} us split it into {len(segments)} segments\n",
             "| segment | launches | wall us | busy us | busy | first kernel |",
             "|---:|---:|---:|---:|---:|---|"]
    for i, s in enumerate(numbers["segments"]):
        lines.append(f"| {i} | {s['launches']} | {s['wall_us']:.1f} | {s['busy_us']:.1f} | "
                     f"{s['busy_us'] / s['wall_us'] if s['wall_us'] else 1:.0%} | {s['first']} |")
    lines += ["\n| idle us | after | before |", "|---:|---|---|"]
    for g in numbers["largest_gaps"]:
        lines.append(f"| {g['us']:.1f} | {g['after']} | {g['before']} |")
    return lines, numbers


def overview(groups, statements, top=20, per_step=None):
    """Markdown: kernels by their total span, and the waits the launches' last warps spend
    most time in (a launch ends with its last warp). With ``per_step`` (the decode steps
    the groups cover), launches and spans are also given a step."""
    total = sum(g["span_total_us"] for g in groups)
    over = f" over {per_step} decode steps" if per_step else ""
    step = per_step or 1
    lines = [f"\n## Overview{over}: {sum(g['launches'] for g in groups)} launches of {len(groups)} "
             f"kernel shapes, spans summing to {total:.1f} us ({total / step:.1f} a step)\n",
             "| kernel | grid | launches a step | span us a step | share | span us a launch | last warp: "
             "wait | load | compute | store | unattributed |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups[:top]:
        last = g["last_warp"]
        shares = [last["phases"].get(p, 0.0) / last["us"] if last["us"] else 0.0
                  for p in ("wait", "load", "compute", "store", "unattributed")]
        lines.append(f"| {g['kernel']} | {g['grid']} | {g['launches'] / step:g} | "
                     f"{g['span_total_us'] / step:.1f} | {g['span_total_us'] / total:.1%} | "
                     f"{g['span_us']:.2f} | " + " | ".join(f"{s:.0%}" for s in shares) + " |")
    waits = []
    for g in groups:
        for name, us in g["last_warp"]["ranges_total_us"].items():
            if phase_of(name) == "wait":
                waits.append((us, g, name))
    lines += ["\n| wait | kernel | grid | last-warp us a step | a launch | first statement |",
              "|---|---|---|---:|---:|---|"]
    for us, g, name in sorted(waits, key=lambda item: -item[0])[:top]:
        statement = statements.get((g["kernel"], name), "")[:80].replace("|", "\\|")
        lines.append(f"| {name} | {g['kernel']} | {g['grid']} | {us / step:.1f} | {us / g['launches']:.2f} | "
                     f"`{statement}` |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--sources")
    parser.add_argument("--nsys", nargs="*", default=[])
    parser.add_argument("--kernels")
    parser.add_argument("--between")
    parser.add_argument("--gap-us", type=float, default=100.0)
    parser.add_argument("--step-kernel",
                        help="with --between: split the window at each launch of this kernel (once "
                             "an engine step); the tables then cover the decode steps")
    parser.add_argument("--prefill-kernel", default="prepare_prefill_inputs_kernel",
                        help="a step that launches it is a prefill step")
    parser.add_argument("--json")
    args = parser.parse_args()
    trace = find_trace(args.trace)
    data = json.loads(trace.read_text())
    graphs = data.get("graphLaunches")
    print(f"trace {trace}: {len(data['launches'])} launches, graphLaunches "
          f"{type(graphs).__name__} of {len(graphs or [])}")
    launches = all_launches(data)
    if args.between:
        launches = window(launches, args.between)
    lines, numbers = timeline(spans(launches), args.gap_us) if args.between else ([], None)
    steps, per_step = None, None
    if args.step_kernel:
        parts = engine_steps(launches, args.step_kernel, args.prefill_kernel)
        step_lines, steps = step_table(parts)
        lines += step_lines
        decode = [part for kind, part in parts if kind == "decode"]
        if not decode:
            raise SystemExit("no decode step in the window")
        launches = [launch for part in decode for launch in part]
        per_step = len(decode)
    statements, sampling = sites(args.sources) if args.sources else ({}, {})
    groups = summarize(launches, data["stringTable"], data["locationTable"], args.kernels, sampling)
    del data
    nsys = {}
    for item in args.nsys:
        label, path = item.split("=", 1)
        nsys[label] = nsys_durations(path)
    for g in groups:
        key = (g["kernel"], tuple(g["grid"]), tuple(g["block"]))
        g["nsys_us"] = {label: median(d.get(key, [])) for label, d in nsys.items()}
        g["nsys_launches"] = {label: len(d.get(key, [])) for label, d in nsys.items()}
    lines += overview(groups, statements, per_step=per_step)
    for g in groups:
        nsys_text = "".join(f", nsys {label} {us:.2f} us" for label, us in g["nsys_us"].items())
        lines.append(f"\n### {g['kernel']} grid {g['grid']} block {g['block']}\n")
        if g["sampled_ctas"]:
            counted = (f"{g['all_warps']} warps; ranges, warp and last warp from the "
                       f"{g['ctas']} recording CTAs' {g['warps']} warps")
        else:
            counted = f"{g['ctas']} CTAs, {g['warps']} warps"
        lines.append(f"{g['launches']} launches, {counted}; span {g['span_us']:.2f} us, "
                     f"warp {g['warp_us']:.2f} us{nsys_text}\n")
        lines.append("| phase | us a warp | share | last warp us | last warp share |")
        lines.append("|---|---:|---:|---:|---:|")
        last = g["last_warp"]
        partial = any(r["sampled_iterations"] for r in g["ranges"].values())
        for phase in sorted(g["phases"], key=lambda p: -g["phases"][p]):
            us = g["phases"][phase]
            lu = last["phases"].get(phase, 0.0)
            label = "unattributed, with loops' unsampled iterations" if partial and phase == "unattributed" else phase
            lines.append(f"| {label} | {us:.2f} | {us / g['warp_us']:.1%} | {lu:.2f} | "
                         f"{lu / last['us'] if last['us'] else 0:.1%} |")
        lines.append("\n| range | us a warp | share | entries | longest warp us | first statement |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, r in sorted(g["ranges"].items(), key=lambda item: -item[1]["us"]):
            statement = statements.get((g["kernel"], name), "")[:90].replace("|", "\\|")
            label = f"{name} (sampled iterations)" if r["sampled_iterations"] else name
            lines.append(f"| {label} | {r['us']:.2f} | {r['share']:.1%} | {r['count']:.1f} | "
                         f"{r['max_us']:.2f} | `{statement}` |")
    print("\n".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps({"timeline": numbers, "steps": steps, "decode_steps": per_step,
                                               "groups": groups}, indent=1) + "\n")


if __name__ == "__main__":
    main()
