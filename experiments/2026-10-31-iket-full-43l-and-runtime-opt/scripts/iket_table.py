#!/usr/bin/env python3
"""A per-phase table from an IKET trace (run-iket's ``*.trace.json``).

usage: iket_table.py TRACE.json|RUN_DIR [--sources DIR [DIR ...]] [--slice TRACE.json ...]
                     [--nsys NAME=FILE.sqlite ...] [--kernels REGEX]
                     [--between MARKER_KERNEL [--gap-us US]] [--json OUT.json]

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
  the warp lifetime, the ranges a warp enters, the longest warp's time in it,
  and when static or runtime payload bytes (``tma_bytes``, ``cp_async_bytes_per_thread``,
  or event ``payload``) are present, the effective bandwidth in GB/s;
- per phase (the range name before the dot): the same summed, plus
  ``unattributed`` (lifetime less every range), and the phase split of the warp
  that ends last, the one the launch waits for.

With ``--slice TRACE.json ...``, family-sliced IKET runs are merged into the base trace's
uninstrumented launches by ``(kernelName, grid, block)`` ordinal, rescaling each slice
launch's ranges into the base launch's span so the table reports real base spans alongside
rescaled phase breakdowns.
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
    return name.split("*")[0].split("#")[0].split(".")[0]


def factor_of(name):
    """The factor a sampled range's name carries to its whole loops (``*N``), else 1."""
    match = re.search(r"\*([0-9.]+)$", name)
    return float(match.group(1)) if match else 1.0


def sites(directories):
    """From one or more IKET site manifest directories:
    - (kernel, range) -> first statement
    - kernel -> its sampling: CTAs, iterations, warps, and ranges inside a loop
    - (kernel, range) -> static payload metadata (tma_bytes, cp_async_bytes_per_thread)."""
    if isinstance(directories, (str, Path)):
        directories = [directories]
    found = defaultdict(set)
    sampling = {}
    payload_meta = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            manifest = json.loads(path.read_text())
            kernel = sampling.setdefault(manifest["kernel"], {
                "ctas": manifest.get("ctas", 0),
                "iters": manifest.get("iters", 0),
                "warps": manifest.get("warps", 0),
                "in_loops": set(),
            })
            for site in manifest["sites"]:
                if "range" in site:
                    key = (site["kernel"], site["range"])
                    found[key].add(site["statement"])
                    if site.get("loops"):
                        kernel["in_loops"].add(site["range"])
                    meta = {}
                    if "tma_bytes" in site:
                        meta["tma_bytes"] = int(site["tma_bytes"])
                    if "cp_async_bytes_per_thread" in site:
                        meta["cp_async_bytes_per_thread"] = int(site["cp_async_bytes_per_thread"])
                    if meta:
                        payload_meta[key] = meta
    return (
        {key: " | ".join(sorted(statements)) for key, statements in found.items()},
        sampling,
        payload_meta,
    )


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


def launch_numbers(launch, names, locations, sampled_ctas=False, sampled_warps=False):
    """One launch: span, warp lifetimes and each warp's time per range. With sampled CTAs
    or sampled warps, lifetimes, ranges and the last warp are those of the warps/CTAs that
    recorded ranges; the span is always the whole launch's."""
    lifetimes = {w["locIdx"]: (w["startTs"], w["endTs"]) for w in launch["warpLifetimes"]}
    if not lifetimes:
        return None
    start = min(s for s, _ in lifetimes.values())
    end = max(e for _, e in lifetimes.values())
    scale = launch.get("_slice_scale", 1.0)
    per_warp = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    payloads = defaultdict(lambda: defaultdict(float))
    for r in launch["ranges"]:
        warp = r["warpLocIdxs"][0]
        name = names[r["rangeNameIdx"]]
        factor = factor_of(name)
        per_warp[warp][name] += (r["endTs"] - r["startTs"]) * scale / 1000 * factor
        counts[warp][name] += 1
        if r.get("payload") is not None:
            payloads[warp][name] += float(r["payload"]) * factor
    warps = list(lifetimes)
    if per_warp:
        if sampled_warps:
            warps = [w for w in warps if w in per_warp]
        elif sampled_ctas:
            recording = {tuple(locations[w]["ctaId"]) for w in per_warp}
            warps = [w for w in warps if tuple(locations[w]["ctaId"]) in recording]
    life = {w: (lifetimes[w][1] - lifetimes[w][0]) * scale / 1000 for w in warps}
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
                "payload": (
                    sum(payloads[w].get(name, 0.0) for w in warps) / len(warps)
                    if any(name in payloads[w] for w in warps) else 0.0
                ),
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


def merge_slice_launches(base_launches, base_strings, base_locations, slice_traces, between=None):
    """Merge instrumented launches from ``slice_traces`` into ``base_launches``.

    For each slice launch that has non-empty ``ranges``, pairs it by
    ``(kernelName, grid, block)`` ordinal with ``base_launches``, remaps ``rangeNameIdx``
    and ``locIdx`` into ``base_strings`` and ``base_locations``, and attaches the rescaled
    slice ranges and ``_slice_scale = base_span / slice_span`` to the base launch while
    preserving the base launch's uninstrumented start/end span."""
    str_to_idx = {s: i for i, s in enumerate(base_strings)}
    by_shape_base = defaultdict(list)
    for bl in base_launches:
        key = (bl["kernelName"],
               (bl["gridDimX"], bl["gridDimY"], bl["gridDimZ"]),
               (bl["blockDimX"], bl["blockDimY"], bl["blockDimZ"]))
        by_shape_base[key].append(bl)

    for s_data in slice_traces:
        s_launches = all_launches(s_data)
        if between:
            s_launches = window(s_launches, between)
        s_strings = s_data["stringTable"]
        s_locs = s_data["locationTable"]
        str_map = {}
        for i, s in enumerate(s_strings):
            if s not in str_to_idx:
                str_to_idx[s] = len(base_strings)
                base_strings.append(s)
            str_map[i] = str_to_idx[s]
        loc_offset = len(base_locations)
        base_locations.extend(s_locs)

        by_shape_slice = defaultdict(list)
        for sl in s_launches:
            key = (sl["kernelName"],
                   (sl["gridDimX"], sl["gridDimY"], sl["gridDimZ"]),
                   (sl["blockDimX"], sl["blockDimY"], sl["blockDimZ"]))
            by_shape_slice[key].append(sl)

        for key, s_items in by_shape_slice.items():
            b_items = by_shape_base.get(key, [])
            if len(b_items) != len(s_items):
                continue
            for bl, sl in zip(b_items, s_items):
                if not sl.get("ranges"):
                    continue
                b_lives = bl["warpLifetimes"]
                s_lives = sl["warpLifetimes"]
                b_span = max(w["endTs"] for w in b_lives) - min(w["startTs"] for w in b_lives)
                s_span = max(w["endTs"] for w in s_lives) - min(w["startTs"] for w in s_lives)
                scale = (b_span / s_span) if s_span > 0 else 1.0
                b0 = min(w["startTs"] for w in b_lives)
                b1 = max(w["endTs"] for w in b_lives)
                s0 = min(w["startTs"] for w in s_lives)
                # Keep base start/end on the first/last warp so span stays exact, while
                # using the slice's warp lifetimes and ranges scaled by `scale`.
                remapped_lives = []
                for w in s_lives:
                    remapped_lives.append({
                        "locIdx": w["locIdx"] + loc_offset,
                        "startTs": int(b0 + (w["startTs"] - s0) * scale),
                        "endTs": int(b0 + (w["endTs"] - s0) * scale),
                    })
                if remapped_lives:
                    remapped_lives[0]["startTs"] = b0
                    remapped_lives[-1]["endTs"] = b1
                bl["warpLifetimes"] = remapped_lives
                bl["ranges"] = [
                    {
                        "warpLocIdxs": [idx + loc_offset for idx in r["warpLocIdxs"]],
                        "startTs": int(b0 + (r["startTs"] - s0) * scale),
                        "endTs": int(b0 + (r["endTs"] - s0) * scale),
                        "rangeNameIdx": str_map[r["rangeNameIdx"]],
                        **({"payload": r["payload"]} if "payload" in r else {}),
                    }
                    for r in sl["ranges"]
                ]
                bl["_slice_scale"] = 1.0


def summarize(launches, names, locations, kernels=None, sampling=None, payload_meta=None):
    """Groups of launches by kernel, grid and block. ``sampling`` is the manifests'
    (``sites``): a CTA-sampled kernel's numbers come from its recording CTAs, and a range
    inside a loop of an iteration-sampled kernel holds the sampled iterations only."""
    sampling = sampling or {}
    payload_meta = payload_meta or {}
    groups = defaultdict(list)
    for launch in launches:
        if kernels and not re.search(kernels, launch["kernelName"]):
            continue
        key = (launch["kernelName"],
               (launch["gridDimX"], launch["gridDimY"], launch["gridDimZ"]),
               (launch["blockDimX"], launch["blockDimY"], launch["blockDimZ"]))
        ksamp = sampling.get(launch["kernelName"], {})
        sampled_ctas = ksamp.get("ctas", 0) > 0
        sampled_warps = ksamp.get("warps", 0) > 0
        numbers = launch_numbers(launch, names, locations, sampled_ctas, sampled_warps)
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
            count = median([n["ranges"].get(name, {}).get("count", 0.0) for n in launches])
            runtime_payload = median([n["ranges"].get(name, {}).get("payload", 0.0) for n in launches])
            factor = factor_of(name)
            meta = payload_meta.get((kernel, name), {})
            r_entry = {
                "us": mean,
                "share": mean / life if life else 0.0,
                "count": count,
                "max_us": median([n["ranges"].get(name, {}).get("max", 0.0) for n in launches]),
                "sampled_iterations": bool(kernel_sampling.get("iters"))
                and name in kernel_sampling.get("in_loops", ()),
                "factor": factor,
            }
            bytes_per_warp = 0.0
            if "tma_bytes" in meta:
                r_entry["tma_bytes"] = meta["tma_bytes"]
                bytes_per_warp = float(meta["tma_bytes"]) * count * factor
            elif "cp_async_bytes_per_thread" in meta:
                r_entry["cp_async_bytes_per_thread"] = meta["cp_async_bytes_per_thread"]
                bytes_per_warp = float(meta["cp_async_bytes_per_thread"]) * 32.0 * count * factor
            elif runtime_payload > 0:
                r_entry["payload"] = runtime_payload
                bytes_per_warp = runtime_payload
            if bytes_per_warp > 0 and mean > 0:
                r_entry["bytes_per_warp"] = bytes_per_warp
                r_entry["gbps_per_warp"] = round(bytes_per_warp / (mean * 1e3), 3)
            ranges[name] = r_entry
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
            "sampled_ctas": kernel_sampling.get("ctas", 0),
            "sampled_iters": kernel_sampling.get("iters", 0),
            "sampled_warps": kernel_sampling.get("warps", 0),
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
    """The launches split at each launch of ``step_kernel`` (once an engine step) after
    the first, as (kind, launches): "prefill" where the step launches
    ``prefill_kernel``, else "decode"."""
    starts = [i for i, launch in enumerate(launches) if launch["kernelName"] == step_kernel]
    if not starts:
        raise SystemExit(f"{step_kernel}: not launched in the window")
    # The window opens on the first step, whose prefill inputs come before its first
    # step kernel (run 29).
    starts[0] = 0
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
    shown = set(sorted(range(len(segments)), key=lambda i: -numbers["segments"][i]["busy_us"])[:top + 5])
    if len(shown) < len(segments):
        lines.insert(1, f"The {len(shown)} of {len(segments)} segments with the most busy time:\n")
    for i, s in enumerate(numbers["segments"]):
        if i not in shown:
            continue
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
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--slice", nargs="*", default=[],
                        help="family-sliced IKET trace JSONs to merge into the base trace")
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
    if args.slice:
        slice_traces = [json.loads(find_trace(p).read_text()) for p in args.slice]
        merge_slice_launches(
            launches, data["stringTable"], data["locationTable"],
            slice_traces, between=args.between,
        )
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
    statements, sampling, payload_meta = (
        sites(args.sources) if args.sources else ({}, {}, {})
    )
    groups = summarize(
        launches, data["stringTable"], data["locationTable"],
        args.kernels, sampling, payload_meta,
    )
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
        if g["sampled_ctas"] or g["sampled_warps"]:
            counted = (f"{g['all_warps']} warps; ranges, warp and last warp from the "
                       f"{g['ctas']} recording CTAs' {g['warps']} warps")
        else:
            counted = f"{g['ctas']} CTAs, {g['warps']} warps"
        lines.append(f"{g['launches']} launches, {counted}; span {g['span_us']:.2f} us, "
                     f"warp {g['warp_us']:.2f} us{nsys_text}\n")
        lines.append("| phase | us a warp | share | last warp us | last warp share |")
        lines.append("|---|---:|---:|---:|---:|")
        last = g["last_warp"]
        partial = any(r["sampled_iterations"] and r["factor"] == 1 for r in g["ranges"].values())
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
            if r["factor"] > 1:
                label = f"{name} (est.: sampled iterations x{r['factor']:g})"
            elif r["sampled_iterations"]:
                label = f"{name} (sampled iterations)"
            else:
                label = name
            if "gbps_per_warp" in r:
                label += f" [{r['gbps_per_warp']} GB/s/warp]"
            lines.append(f"| {label} | {r['us']:.2f} | {r['share']:.1%} | {r['count']:.1f} | "
                         f"{r['max_us']:.2f} | `{statement}` |")
    print("\n".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps({"timeline": numbers, "steps": steps, "decode_steps": per_step,
                                               "groups": groups}, indent=1) + "\n")


if __name__ == "__main__":
    main()
