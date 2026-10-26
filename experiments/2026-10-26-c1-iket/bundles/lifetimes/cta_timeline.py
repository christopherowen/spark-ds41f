#!/usr/bin/env python3
"""What a c1 layer's launches do with the GPU's 48 SMs, from run-iket's warp lifetimes.

usage: cta_timeline.py TRACE.json|RUN_DIR --tp 4|3 [--layers L] [--json OUT]

run-iket records every warp's start, end and SM for every launch, instrumented or not.
c1_layer.py's ``lifetimes`` step runs every kernel uninstrumented, so these lifetimes are
the kernels' own. Launches are taken in issue order (CUPTI's grid id), and the harness's
layers are found by their routed-expert dispatch: in each, the launches issued between
the FFN entry's finalize and the dispatch are the shared expert's (its own stream), the
rest are the main stream.

Per launch:
- ``span``: first warp start to last warp end;
- ``start_gap``: the first warp's start after the previous main-stream launch's end (a
  negative gap is a PDL launch whose CTAs were resident before their predecessor ended;
  their work starts at ``ready``, the later of the two);
- ``busy``: time from ``ready`` to the end; ``exclusive``: the part of ``busy`` after the
  previous main-stream launch ended, which is what the kernel adds to the layer;
- CTAs, the SMs they ran on, waves (CTAs over SMs used), and ``sm_idle``: the share of
  48 SMs x busy time with no CTA of the launch resident.

The routed experts' grouped GEMMs (FC1 and FC2) also get their weight stream: a live
tile (a CTA whose expert is not -1) streams its expert's ``block_N`` rows over every K
block, ``tile_bytes`` (E2M1 values, coded scales and a base exponent a row). Their CTAs
that exit at once (past the last used tile) are counted apart. Within ``busy``:
- ``bytes``: live tiles x tile bytes; ``floor``: bytes at the attainable 266.6 GB/s;
- ``rate``: the stream rate over time, each live CTA streaming its tile evenly over its
  lifetime (from ``ready``), in 1 us bins: ``ramp`` is the time before it first reaches
  90% of the launch's peak bin, ``tail`` the time after it last does, ``steady`` between;
- ``excess``: busy less floor, which is what the launch's fixed costs are.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SMS = 48
RATE = 266.6e9  # bytes/s, FC1's marginal rate at six rows (experiments/2026-10-26-c1-100tps)
HIDDEN = 5120
INTER = {4: 576, 3: 768}
BLOCK_N, BLOCK_K = 128, 128


def tile_bytes(n, k):
    """One live tile of the grouped GEMM: block_N rows over K (tile_expert_weights'
    layout), E2M1 values, a 2-byte scale code per row and 128-wide K block
    (pack_scale_codes), and a 1-byte base exponent per row."""
    blocks = -(-k // BLOCK_K)
    return BLOCK_N * k // 2 + BLOCK_N * 2 * blocks + BLOCK_N


def load(path):
    path = Path(path)
    if path.is_dir():
        path = max(path.glob("*.trace.json"), key=lambda p: p.stat().st_size)
    data = json.loads(path.read_text())
    launches = list(data["launches"])
    graphs = data.get("graphLaunches") or {}
    for value in graphs.values() if isinstance(graphs, dict) else graphs:
        items = value if isinstance(value, list) else value.get("launches", []) if isinstance(value, dict) else []
        launches.extend(i for i in items if isinstance(i, dict) and "kernelName" in i)
    launches = [launch for launch in launches if launch.get("warpLifetimes")]
    launches.sort(key=lambda launch: launch["gridId"])
    return launches, data["locationTable"]


def ctas(launch, locations):
    """CTA -> (start ns, end ns, SM)."""
    out = {}
    for w in launch["warpLifetimes"]:
        loc = locations[w["locIdx"]]
        key = tuple(loc["ctaId"])
        start, end, sm = out.get(key, (w["startTs"], w["endTs"], loc["smId"]))
        out[key] = (min(start, w["startTs"]), max(end, w["endTs"]), sm)
    return out


def segments(launches):
    """Each layer's launches: (main stream launches, shared-expert launches), split at
    the routed dispatch."""
    marks = [i for i, launch in enumerate(launches) if launch["kernelName"].startswith("moe_dispatch")]
    out = []
    previous = None
    for m in marks:
        start = 0 if previous is None else previous
        # A layer ends at its combine (the first moe_combine after the dispatch).
        end = next(i for i in range(m, len(launches)) if launches[i]["kernelName"].startswith("moe_combine")) + 1
        # The shared expert's launches are issued after the FFN entry's finalize.
        f = max(i for i in range(start, m) if launches[i]["kernelName"].startswith("mhc_finalize"))
        main = launches[start:f + 1] + launches[m:end]
        aux = launches[f + 1:m]
        out.append((main, aux))
        previous = end
    return out


def bins(intervals, start, end, width=1000):
    """Sum of rates (bytes/ns) over 1 us bins, from (start, end, bytes) intervals."""
    count = max(1, -(-(end - start) // width))
    out = [0.0] * count
    for s, e, b in intervals:
        if e <= s:
            continue
        r = b / (e - s)
        for i in range(max(0, (s - start) // width), min(count, -(-(e - start) // width))):
            lo, hi = max(s, start + i * width), min(e, start + (i + 1) * width)
            if hi > lo:
                out[i] += r * (hi - lo) / width
    return out


def describe(launch, locations, previous_end, tp):
    c = ctas(launch, locations)
    first = min(s for s, _, _ in c.values())
    last = max(e for _, e, _ in c.values())
    ready = max(first, previous_end) if previous_end is not None else first
    name = launch["kernelName"]
    grid = (launch["gridDimX"], launch["gridDimY"], launch["gridDimZ"])
    busy = last - ready
    sm_busy = defaultdict(int)
    for s, e, sm in c.values():
        s = max(s, ready)
        if e > s:
            sm_busy[sm] += e - s  # CTAs of one launch on one SM run one after another here
    used = len(sm_busy)
    row = {
        "kernel": name, "grid": grid, "ctas": len(c), "sms": used,
        "span_us": (last - first) / 1000, "busy_us": busy / 1000,
        "start_gap_us": None if previous_end is None else (first - previous_end) / 1000,
        "exclusive_us": (last - max(ready, previous_end or ready)) / 1000,
        "waves": round(len(c) / used, 2) if used else 0,
        "sm_idle": round(1 - sum(min(v, busy) for v in sm_busy.values()) / (SMS * busy), 3) if busy > 0 else 0,
        "_end": last, "_ready": ready,
    }
    if name.startswith("moe_grouped"):
        # A CTA past the last used tile exits right after its PDL wait.
        lives = {k: v for k, v in c.items() if v[1] - max(v[0], ready) > 1500}
        n_blocks = grid[0]
        fc1 = n_blocks * BLOCK_N == 2 * INTER[tp]
        n, k = (2 * INTER[tp], HIDDEN) if fc1 else (HIDDEN, INTER[tp])
        per = tile_bytes(n, k)
        total = len(lives) * per
        intervals = [(max(s, ready), e, per) for s, e, _ in lives.values()]
        rate = bins(intervals, ready, last)
        peak = max(rate) if rate else 0.0
        hot = [i for i, r in enumerate(rate) if r >= 0.9 * peak]
        ramp = hot[0] if hot else 0
        tail = len(rate) - 1 - hot[-1] if hot else 0
        floor = total / RATE * 1e9
        lifetimes = sorted((e - max(s, ready)) / 1000 for s, e, _ in lives.values())
        starts = sorted((max(s, ready) - ready) / 1000 for s, _, _ in lives.values())
        row.update({
            "gemm": "FC1" if fc1 else "FC2", "live_tiles": len(lives), "experts": round(len(lives) / n_blocks, 2),
            "exited_ctas": len(c) - len(lives), "tile_bytes": per, "bytes": total,
            "floor_us": floor / 1000, "excess_us": busy / 1000 - floor / 1000,
            "achieved_gbs": total / busy if busy else 0, "peak_bin_gbs": peak,
            "ramp_us": ramp, "steady_us": len(rate) - ramp - tail, "tail_us": tail,
            "cta_life_us": {"min": lifetimes[0], "median": statistics.median(lifetimes), "max": lifetimes[-1]},
            "cta_start_us": {"p50": statistics.median(starts), "max": starts[-1]},
            "waves_live": round(len(lives) / used, 2) if used else 0,
            "rate_bins_gbs": [round(r, 1) for r in rate],
        })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--tp", type=int, choices=(4, 3), required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--routes")
    ap.add_argument("--live", type=int)
    ap.add_argument("--json")
    args = ap.parse_args()
    launches, locations = load(args.trace)
    layers = segments(launches)
    print(f"{len(launches)} launches, {len(layers)} layers")
    rows = []
    for index, (main, aux) in enumerate(layers):
        previous = None
        for launch in main:
            row = describe(launch, locations, previous, args.tp)
            row["layer"], row["stream"] = index, "main"
            rows.append(row)
            previous = row["_end"]
        for launch in aux:
            row = describe(launch, locations, None, args.tp)
            row["layer"], row["stream"] = index, "aux"
            rows.append(row)
    # Steady layers: drop the first layer of each pass (its predecessor is the host).
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stream"], row["kernel"], row["grid"])].append(row)
    summary = []
    for (stream, kernel, grid), items in groups.items():
        keys = ("span_us", "busy_us", "exclusive_us", "start_gap_us", "sm_idle", "waves", "ctas", "sms")
        entry = {"stream": stream, "kernel": kernel, "grid": grid, "launches": len(items)}
        for key in keys:
            values = [r[key] for r in items if r[key] is not None]
            entry[key] = round(statistics.median(values), 3) if values else None
        if kernel.startswith("moe_grouped"):
            for key in ("experts", "live_tiles", "bytes", "floor_us", "excess_us", "achieved_gbs", "peak_bin_gbs",
                        "ramp_us", "steady_us", "tail_us", "waves_live"):
                entry[key] = round(statistics.median(r[key] for r in items), 3)
            entry["gemm"] = items[0]["gemm"]
        summary.append(entry)
    summary.sort(key=lambda e: -(e["exclusive_us"] or 0) * e["launches"])
    print("| stream | kernel | grid | launches | span us | busy us | exclusive us | start gap us | SM idle | waves |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for e in summary:
        print(f"| {e['stream']} | {e['kernel']} | {e['grid']} | {e['launches']} | {e['span_us']:.2f} | "
              f"{e['busy_us']:.2f} | {e['exclusive_us']:.2f} | {e['start_gap_us'] if e['start_gap_us'] is None else round(e['start_gap_us'], 2)} | "
              f"{e['sm_idle']:.0%} | {e['waves']} |")
    print("\n| GEMM | grid | experts | live tiles | MB | floor us | busy us | excess us | GB/s | peak 1-us bin GB/s "
          "| ramp us | steady us | tail us | live waves |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for e in summary:
        if "gemm" in e:
            print(f"| {e['gemm']} | {e['grid']} | {e['experts']} | {e['live_tiles']} | {e['bytes'] / 1e6:.2f} | "
                  f"{e['floor_us']:.2f} | {e['busy_us']:.2f} | {e['excess_us']:.2f} | {e['achieved_gbs']:.1f} | "
                  f"{e['peak_bin_gbs']:.1f} | {e['ramp_us']} | {e['steady_us']} | {e['tail_us']} | {e['waves_live']} |")
    if args.json:
        for row in rows:
            row.pop("_end"), row.pop("_ready")
        Path(args.json).write_text(json.dumps({"summary": summary, "launches": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
