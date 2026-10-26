#!/usr/bin/env python3
"""Each L2 prefetch window's consumers by decoder-layer type, and what a different issue
point could recover, from rank 0's decode traces.

usage: schedule_bounds.py [TRACES] [--scenarios greedy,mixed-c8] [--out JSON]

TRACES is request-inventory-5's ports directory on dgx1 (decode-trace/dgx1: one torch
profiler trace a request type, in traces.DECODE order). Each graph launch's main chain is
split into decoder layers by its anchor kernels (capture order):

  [Engram projection]  mHC entry  fused Q-A/KV  Q-B  | WO fill |  attention core
  WO-A  WO-B  | FFN fill |  all-reduce  mHC entry  router  FC1 ... combine  | NEXT fill |
  all-reduce

A layer's type follows its attention core: "own" holds a compressor projection
(gemm_bf16_splitk; layers 2, 8, 14, 20 own the compressed caches and index), "idx" an
indexer query projection (gemm_mxfp8 with Q-B's grid; layers 24, 28, 32, 36), "swa" the
rest (window-only and compressed layers that reuse a selection, and the drafter's three).
"eng" marks the two Engram layers (1, 14), whose projection runs before the mHC entry.

A consumer's floor is the 25th percentile of its calls that started after their window's
prefetch kernel had ended, in layers whose lines are not lost (not "own" WO, not "eng"
NEXT). Its excess is its time above the floor.

DRAM is taken as idle during latency-bound kernels, gaps and collectives, and during a
consumer's floor share of its time (it reads L2); busy during weight GEMMs in no window
(routed experts, compressor, indexer and Engram projections, the head). "Unused" idle is
idle time while no prefetch kernel runs. An option's bound is the unused idle time it
reaches, times R / BW (a fill's rate against a consumer's DRAM rate), capped by the
excess of the consumers it serves. Collectives count as idle: on the ring a fill beside an
all-reduce slows it (2026-10-06), which these bounds do not charge.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict

import traces as tr

R = 215e9  # fill rate at grid 2, bytes/s: a window's bytes over its kernel's p50 time
BW = 250e9  # a consumer's DRAM read rate for bytes not in L2
# TP4 window plans, in fill order (the release's [l2_prefetch] lines): consumer, bytes.
WO = (("WO-A", 9.0e6), ("WO-B", 11.0e6))
FFN = (("mHC", 2.0e6), ("router", 3.9e6), ("shared gate/up", 5.9e6), ("shared down", 2.9e6))
FFN_DRAFT = (("mHC", 2.0e6), ("router", 1.3e6), ("shared gate/up", 5.9e6), ("shared down", 2.9e6))
NEXT = (("mHC", 2.0e6), ("fused Q-A/KV", 9.2e6), ("Q-B", 8.8e6))
Q_B_TAIL = 1.7e6  # Q-B bytes past the NEXT budget, read from DRAM in every layer
L2_US_PER_MB = 1.0  # a GEMM's L2 read, as Q-B's floor shows
BUSY = (
    "gemm_mxfp8",
    "gemm_bf16_splitk",
    "gemm_bf16_partials",
    "gemm_fp8w_bf16",
    "moe_grouped_fp8_fp4_gemm",
    "vocab_packed_projection",
)


def is_(k, name, grid=None):
    return tr.base(k["name"]) == name and (grid is None or k["grid"][: len(grid)] == grid)


def after(seq, i, test, stop=None):
    if i is None:
        return None
    for j in range(i + 1, len(seq)):
        if stop is not None and stop(seq[j]):
            return None
        if test(seq[j]):
            return j
    return None


def parse(launch_kernels):
    """Decoder layers of one graph launch: anchor indices into its main chain."""
    main = sorted((k for k in launch_kernels if k["cap"] == "main"), key=lambda k: k["node"])
    shared = sorted((k for k in launch_kernels if k["cap"] == "shared"), key=lambda k: k["node"])
    fills = sorted((k for k in launch_kernels if k["cap"] == "prefetch"), key=lambda k: k["node"])
    layers = []
    i = -1
    while True:
        e = after(main, i, lambda k: is_(k, "mhc_project_streams"))
        if e is None:
            break
        qakv = after(main, e, lambda k: is_(k, "gemm_mxfp8", (28,)),
                     stop=lambda k: is_(k, "mhc_project_streams"))  # fmt: skip
        if qakv is None:  # an FFN entry: belongs to the layer before
            i = e
            continue
        qb = after(main, qakv, lambda k: is_(k, "gemm_mxfp8", (64,)))
        woa = after(main, qb, lambda k: is_(k, "gemm_fp8w_bf16_partials"))
        wob = after(main, woa, lambda k: is_(k, "gemm_mxfp8", (40,)))
        ffn = after(main, wob, lambda k: is_(k, "mhc_project_streams"))
        router = after(main, ffn, lambda k: is_(k, "gemm_bf16_partials"))
        combine = after(main, router, lambda k: is_(k, "moe_combine"))
        if None in (qb, woa, wob, ffn, router, combine):
            i = e
            continue
        core = main[qb + 1 : woa]
        kind = (
            "own" if any(is_(k, "gemm_bf16_splitk") for k in core)
            else "idx" if any(is_(k, "gemm_mxfp8", (64,)) for k in core)
            else "swa"
        )  # fmt: skip
        prev_combine = layers[-1]["combine"] if layers else -1
        engram = next(
            (j for j in range(prev_combine + 1, e) if is_(main[j], "gemm_mxfp8", (200,))), None
        )
        idx_qb = next((qb + 1 + j for j, k in enumerate(core) if is_(k, "gemm_mxfp8", (64,))), None)
        wproj = next(
            (qb + 1 + j for j, k in enumerate(core) if is_(k, "gemm_bf16_partials", (1, 40))), None
        )
        sh = [k for k in shared if main[ffn]["node"] < k["node"] < main[combine]["node"]
              and is_(k, "gemm_mxfp8")]  # fmt: skip
        layers.append(dict(kind=kind, eng=engram is not None, entry=e, qakv=qakv, qb=qb,
                           woa=woa, wob=wob, ffn=ffn, router=router, combine=combine,
                           engram=engram, idx_qb=idx_qb, wproj=wproj, shared=sh[:2]))  # fmt: skip
        i = combine
    # each fill belongs to the main kernel captured just before it
    nodes = [k["node"] for k in main]
    by_fork = {}
    for f in fills:
        j = max((n, j) for j, n in enumerate(nodes) if n < f["node"])[1]
        by_fork[j] = f
    for n, layer in enumerate(layers):
        layer["fill_wo"] = by_fork.get(layer["qb"])
        layer["fill_ffn"] = by_fork.get(layer["wob"])
        layer["fill_next"] = by_fork.get(layer["combine"]) if n + 1 < len(layers) else None
    return main, layers


def consumers(main, layers, n, draft):
    """(window, label, consumer kernel, bytes before it in the fill, its bytes) of layer n."""
    L = layers[n]
    out = []
    if L["fill_wo"] is not None:
        out += [("WO", "WO-A", main[L["woa"]], 0.0, WO[0][1]),
                ("WO", "WO-B", main[L["wob"]], WO[0][1], WO[1][1])]  # fmt: skip
    if L["fill_ffn"] is not None:
        plan = FFN_DRAFT if draft else FFN
        ks = [main[L["ffn"]], main[L["router"]], *L["shared"]]
        acc = 0.0
        for (label, size), k in zip(plan, ks):
            out.append(("FFN", label, k, acc, size))
            acc += size
    if L["fill_next"] is not None and n + 1 < len(layers):
        M = layers[n + 1]
        acc = 0.0
        for (label, size), k in zip(NEXT, (main[M["entry"]], main[M["qakv"]], main[M["qb"]])):
            out.append(("NEXT", label, k, acc, size))
            acc += size
    return out


class Idle:
    """DRAM state of a launch's main chain: idle(t0, t1) is the DRAM-idle time in
    [t0, t1], and unused(t0, t1) the part with no prefetch kernel running."""

    def __init__(self, main, floors, fills):
        self.spans = []  # (start, end, idle fraction)
        for a, b in zip(main, main[1:] + [None]):
            frac = 1.0
            if a.get("consumer"):
                frac = min(1.0, floors.get(a["consumer"], a["dur"]) / max(a["dur"], 1e-6))
            elif tr.base(a["name"]).startswith(BUSY):
                frac = 0.0
            self.spans.append((a["ts"], a["end"], frac))
            if b is not None and b["ts"] > a["end"]:
                self.spans.append((a["end"], b["ts"], 1.0))
        self.fills = sorted((f["ts"], f["end"]) for f in fills)

    def idle(self, t0, t1, exclude_fills=False):
        total = 0.0
        for s, e, frac in self.spans:
            lo, hi = max(s, t0), min(e, t1)
            if hi <= lo or frac == 0.0:
                continue
            span = hi - lo
            if exclude_fills:
                span -= sum(max(0.0, min(hi, fe) - max(lo, fs)) for fs, fe in self.fills)
            total += frac * max(0.0, span)
        return total


def analyse(ks, windows):
    tr.label(ks)
    calls = []  # one row a consumer call
    rows = []  # one row a layer instance
    for a, b in windows:
        step = ks[a:b]
        launches = defaultdict(list)
        for k in step:
            if k["launch"][0] == "G":
                launches[k["launch"]].append(k)
        for launch, kernels in launches.items():
            if not any(k["cap"] == "prefetch" for k in kernels):
                continue
            main, layers = parse(kernels)
            if not layers:
                continue
            draft = len(layers) < 10
            for n, L in enumerate(layers):
                for window, label, k, before, size in consumers(main, layers, n, draft):
                    fill = L[{"WO": "fill_wo", "FFN": "fill_ffn", "NEXT": "fill_next"}[window]]
                    target = layers[n + 1] if window == "NEXT" else L
                    lost = (window == "WO" and L["kind"] == "own") or (
                        window == "NEXT" and target["eng"]
                    )
                    key = (window, label, "D" if draft else "T")
                    k["consumer"] = key
                    calls.append(dict(key=key, kind=L["kind"], eng=target["eng"], lost=lost,
                                      dur=k["dur"], lead=fill["end"] - k["ts"], k=k, fill=fill,
                                      before=before, size=size, layer=n))  # fmt: skip
            rows.append(dict(main=main, layers=layers, draft=draft, step=(a, b)))
    floors = {}
    for key in {c["key"] for c in calls}:
        landed = [c["dur"] for c in calls if c["key"] == key and c["lead"] <= 0 and not c["lost"]]
        if len(landed) >= 4:
            floors[key] = statistics.quantiles(landed, n=4)[0]
    for c in calls:
        c["excess"] = max(0.0, c["dur"] - floors.get(c["key"], c["dur"]))
    return calls, rows, floors


def bounds(calls, rows, floors, n_steps):
    """Per option, the time a step could recover (us)."""
    by_kernel = {id(c["k"]): c for c in calls}
    out = defaultdict(float)
    detail = defaultdict(list)
    for row in rows:
        main, layers = row["main"], row["layers"]
        fills = [L[f] for L in layers for f in ("fill_wo", "fill_ffn", "fill_next") if L[f]]
        idle = Idle(main, floors, fills)
        for n, L in enumerate(layers):
            exc = defaultdict(float)
            for window, label, k, *_ in consumers(main, layers, n, row["draft"]):
                c = by_kernel.get(id(k))
                if c is not None:
                    exc[(window, label)] += c["excess"]
            wo_ex = exc[("WO", "WO-A")] + exc[("WO", "WO-B")]
            ffn_ex = sum(v for (w, _), v in exc.items() if w == "FFN")
            wo, ffn = L["fill_wo"], L["fill_ffn"]
            prev_next = layers[n - 1]["fill_next"] if n else None
            shift = 0.0
            if wo is not None and L["kind"] == "swa":
                # S1: WO issued after the fused Q-A/KV, so it starts when NEXT's fill ends
                start = max(main[L["qakv"]]["end"], prev_next["end"] if prev_next else 0.0)
                reach = idle.idle(start, wo["ts"], exclude_fills=True)
                gain = min(wo_ex, reach * R / BW)
                out["S1 WO after Q-A/KV (swa layers)"] += gain
                shift = reach  # the WO fill ends about this much earlier
                detail["S1 reach us"].append(reach)
            if wo is not None and L["kind"] == "own" and L["idx_qb"] is not None:
                # S3: an owner layer loses its WO lines to the compressor's projection.
                # IDX (compressor 10.5 MB, the indexer's Q-B and head weights 5.6 MB)
                # is issued after Q-B and WO right after the compressor, behind IDX.
                # IDX lands the compressor's first bytes over the RoPE and cache
                # writes before its projection, and the indexer's 5.6 MB over the
                # compression after it; the indexer's Q-B then reads L2. WO's fill
                # starts when IDX has landed and runs to WO-A; bytes it has not
                # landed by then (WO-A's L2 time lands more for WO-B) are WO's new
                # excess. Unlike S1 and S2 this is an estimate: the new excess
                # can exceed the lost-line excess it replaces.
                comp = next(i for i in range(L["qb"] + 1, L["woa"])
                            if is_(main[i], "gemm_bf16_splitk"))  # fmt: skip
                qb_end, iq = main[L["qb"]]["end"], main[L["idx_qb"]]
                pre = idle.idle(qb_end, main[comp]["ts"])
                comp_gain = min(main[comp]["dur"], pre * R / BW)
                # IDX's indexer bytes land over the idle time after the projection
                t, need = main[comp]["end"], 5.6e6
                landed_at = t + need / R * 1e6
                after = idle.idle(main[comp]["end"], iq["ts"])
                idx_floor = 5.3 * L2_US_PER_MB + 1.0
                idx_gain = max(0.0, iq["dur"] - idx_floor) * min(1.0, after * R / need)
                # WO's reach: idle time from IDX's landing to WO-A, with the indexer's
                # Q-B now read from L2
                compress = next(i for i in range(comp + 1, L["woa"])
                                if is_(main[i], "compress"))  # fmt: skip
                start = max(landed_at, main[compress]["end"])
                reach = idle.idle(start, main[L["woa"]]["ts"])
                if iq["ts"] >= start:
                    reach += idx_floor
                woa_floor = floors.get(("WO", "WO-A", "T"), 12.0)
                missing = max(0.0, 20.0e6 - (reach + woa_floor) * 1e-6 * R)
                new_excess = missing / BW * 1e6
                out["S3 owner layers: IDX after Q-B, WO after the compressor"] += (
                    wo_ex - new_excess + comp_gain + idx_gain
                )
                detail["S3 WO reach us"].append(reach)
                detail["S3 WO new excess us"].append(new_excess)
            if ffn is not None:
                # S2: FFN issued after WO-A, so it starts when the WO fill ends
                wo_end = (wo["end"] - shift) if wo is not None else 0.0
                start = max(main[L["woa"]]["end"], wo_end)
                reach = idle.idle(start, ffn["ts"], exclude_fills=True)
                if wo is not None and shift > 0:
                    # under S1 the stretch the WO fill vacates becomes unused too
                    reach += idle.idle(max(start, wo["end"] - shift), wo["end"])
                out["S2 FFN after WO-A"] += min(ffn_ex, reach * R / BW)
                detail["S2 reach us"].append(reach)
            if n + 1 < len(layers) and layers[n + 1]["eng"]:
                M = layers[n + 1]
                nx = sum(c["excess"] for c in calls if c["layer"] == n and c["key"][0] == "NEXT"
                         and c["k"]["ts"] >= main[M["entry"]]["ts"] - 1)  # fmt: skip
                gap = idle.idle(main[M["engram"]]["end"], main[M["entry"]]["ts"])
                out["S4a NEXT after the Engram projection"] += min(nx, gap * R / BW)
                proj = main[M["engram"]]
                before = idle.idle(main[L["combine"]]["end"], proj["ts"])
                out["S4b ENGRAM window (projection)"] += min(
                    proj["dur"] * 0.5, before * R / BW
                )
                detail["S4 idle before projection us"].append(before)
            if n + 1 < len(layers) and L["fill_next"] is not None:
                out["S6 Q-B tail in NEXT (shares S1's reach)"] += Q_B_TAIL / BW * 1e6
    return {k: v / n_steps for k, v in out.items()}, {
        k: round(statistics.median(v), 1) for k, v in detail.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?",
                    default="results/private/determinism/request-inventory-5/ports")  # fmt: skip
    ap.add_argument("--scenarios", default="greedy,mixed-c8")
    # The request types the traces hold, in file order (profile_decode.py's --scenarios).
    ap.add_argument("--order", default=",".join(tr.DECODE))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "schedule_bounds.json"))
    args = ap.parse_args()
    files = sorted(os.listdir(os.path.join(args.traces, "decode-trace/dgx1")))
    trace = dict(zip(args.order.split(","),
                     (os.path.join(args.traces, "decode-trace/dgx1", f) for f in files)))
    result = {"source": args.traces, "R": R, "BW": BW, "scenarios": {}}
    for name in args.scenarios.split(","):
        ks = tr.load(trace[name])
        skip, last = tr.WINDOWS.get(name, (1, None))
        windows = tr.steps(ks, skip, last)
        calls, rows, floors = analyse(ks, windows)
        n = len(windows)
        step_us = statistics.median(ks[b]["ts"] - ks[a]["ts"] for a, b in windows)
        table = defaultdict(lambda: defaultdict(list))
        for c in calls:
            group = "lost" if c["lost"] else c["kind"]
            t = table[(c["key"], group)]
            t["dur"].append(c["dur"])
            t["lead"].append(c["lead"])
            t["excess"].append(c["excess"])
        consumers_out = {}
        for (key, group), t in sorted(table.items()):
            consumers_out[f"{key[2]} {key[0]}/{key[1]} [{group}]"] = {
                "per_step": round(len(t["dur"]) / n, 1),
                "floor_us": round(floors.get(key, 0.0), 1),
                "dur_p50_us": round(statistics.median(t["dur"]), 1),
                "starts_before_fill_end": round(sum(x > 0 for x in t["lead"]) / len(t["lead"]), 2),
                "lead_p50_us": round(statistics.median(t["lead"]), 1),
                "excess_us_per_step": round(sum(t["excess"]) / n, 1),
            }
        per_window = defaultdict(float)
        for c in calls:
            state = "lines lost" if c["lost"] else "late" if c["lead"] > 0 else "landed"
            per_window[f"{c['key'][0]} ({state})"] += c["excess"] / n
        timing = sum(v for k, v in per_window.items() if "landed" not in k)
        opts, detail = bounds(calls, rows, floors, n)
        result["scenarios"][name] = {
            "steps": n,
            "step_us_p50": round(step_us, 1),
            "excess_us_per_step": {k: round(v, 1) for k, v in sorted(per_window.items())},
            "excess_total_percent": round(100 * sum(per_window.values()) / step_us, 2),
            "excess_late_or_lost_us": round(timing, 1),
            "excess_late_or_lost_percent": round(100 * timing / step_us, 2),
            "consumers": consumers_out,
            "bounds_us_per_step": {k: round(v, 1) for k, v in sorted(opts.items())},
            "bounds_percent": {k: round(100 * v / step_us, 2) for k, v in sorted(opts.items())},
            "medians": detail,
        }
        s = result["scenarios"][name]
        print(f"{name}: {n} steps, {step_us / 1000:.2f} ms a step, excess {s['excess_total_percent']}%"
              f" (late or lost {s['excess_late_or_lost_us']} us, {s['excess_late_or_lost_percent']}%)")
        for k, v in s["excess_us_per_step"].items():
            print(f"  excess {k:32} {v:7.1f} us")
        for k, v in s["consumers"].items():
            print(f"  {k:44} {json.dumps(v)}")
        for k, v in s["bounds_us_per_step"].items():
            print(f"  bound {k:52} {v:7.1f} us  {s['bounds_percent'][k]:5.2f}%")
        print(f"  medians {json.dumps(detail)}")
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)


if __name__ == "__main__":
    main()
