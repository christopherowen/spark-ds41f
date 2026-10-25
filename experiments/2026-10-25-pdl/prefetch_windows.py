#!/usr/bin/env python3
"""The L2 weight prefetch windows on a decode step's critical path, and what PDL or an
earlier issue could add.

usage: prefetch_windows.py [TRACES] [--scenarios ...] [--window NAME=FIRST:END] [--out JSON]

Same traces and steps as critical_path.py. Every l2_prefetch_kernel is placed by capture
order (the graph node id): it is forked from the main kernel captured just before it, and
its window follows from what comes next (l2_prefetch.py's three a layer):

- WO, after Q-B: this layer's WO-A and WO-B;
- FFN, after WO-B, before the attention all-reduce: the FFN entry's mHC projection, the
  router and the shared expert's gate/up and down;
- NEXT, after the combine, before the MoE all-reduce: the next attention entry's mHC
  projection, fused Q-A/KV and Q-B.

Per window instance: issue (the fork kernel's end), start, end, the side stream's idle
time before the issue (the previous prefetch ended that long before), and each consumer's
start. A consumer that starts before its window's prefetch ends reads part of its weight
from DRAM while the prefetch still fills L2. Its excess is its time above the median of
its calls that started after their prefetch had finished; on the critical path it counts
up to the consumer's share of the path.

Also: main kernels on the path that run beside a prefetch, against calls of the same
kernel and grid that do not (the fill's contention); boundaries where a prefetch is forked
against the others (the fork's cost to the chain); the prefetch join.

Bounds:
- issuing FFN and NEXT earlier inside their windows (never after their all-reduce): a late
  instance can gain at most min(its lateness, the side stream's idle time before it) and at
  most its consumers' excess;
- PDL-coordinated prefetch, a consumer prefetching its own weight in its prologue while
  the kernel before it runs (``pdl.chain()`` launches it when that kernel passes its
  wait): at most the predecessor's share of the path, the consumer's excess, and only
  while no window prefetch already holds DRAM; for weight GEMMs in no window, at most the
  predecessor's share while no prefetch runs.
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

import critical_path as cp

WEIGHT_GEMMS = cp.WEIGHT_GEMMS


def latency_bound(k: dict) -> bool:
    """Not a weight GEMM (its time is weight bytes) nor a collective (peer waits)."""
    name = cp.base(k["name"])
    return not name.startswith(
        (*WEIGHT_GEMMS, "oneshot_", "ncclDev", "engram_rows_wait")
    )


def dram_idle(path_kernels: list[dict], on_path: dict, w: dict) -> float:
    """The latency-bound path time just before this window's issue: back from the fork
    kernel through consecutive latency-bound kernels, stopping at a weight GEMM or where
    the side stream was still busy. An issue moved earlier than that would run beside a
    weight GEMM, which takes the same DRAM bandwidth, so the bytes would only move from
    the consumer to that GEMM."""
    if w["idle_before"] is None:
        return 0.0
    lo = w["issue"] - w["idle_before"]
    total = 0.0
    for k in reversed([k for k in path_kernels if k["end"] <= w["issue"] + 0.05]):
        if not latency_bound(k) or k["end"] <= lo:
            break
        own, pred = on_path[id(k)]
        start = k["end"] - own if pred is None else min(k["ts"], pred["end"])
        total += k["end"] - max(start, lo)
        if start <= lo:
            break
    return total


def windows_of(step: list[dict]) -> list[dict]:
    """Every prefetch of the step's graph launches, with its window and consumers."""
    out = []
    for launch in sorted({k["launch"] for k in step if k["launch"][0] == "G"}):
        seq = sorted(
            (k for k in step if k["launch"] == launch), key=lambda k: k["node"]
        )
        main = [k for k in seq if k["cap"] != "prefetch"]
        order = {id(k): i for i, k in enumerate(main)}
        previous_end = None
        for p in (k for k in seq if k["cap"] == "prefetch"):
            before = [k for k in main if k["node"] < p["node"] and k["cap"] == "main"]
            if not before:
                continue
            fork = before[-1]
            rest = main[order[id(fork)] + 1 :]
            reduce_at = next(
                (
                    i
                    for i, k in enumerate(rest)
                    if cp.base(k["name"]) == "oneshot_allreduce"
                ),
                None,
            )
            ahead = rest if reduce_at is None else rest[:reduce_at]
            after = [] if reduce_at is None else rest[reduce_at + 1 :]
            consumers = {}
            if any(cp.base(k["name"]) == "gemm_fp8w_bf16_partials" for k in ahead):
                kind = "WO"
                i = next(
                    i
                    for i, k in enumerate(ahead)
                    if cp.base(k["name"]) == "gemm_fp8w_bf16_partials"
                )
                consumers["WO-A"] = ahead[i]
                b = next(
                    (k for k in ahead[i + 1 :] if cp.base(k["name"]) == "gemm_mxfp8"),
                    None,
                )
                if b is not None:
                    consumers["WO-B"] = b
            elif cp.base(fork["name"]) == "moe_combine":
                kind = "NEXT"
                entry = next(
                    (k for k in after if cp.base(k["name"]) == "mhc_project_streams"),
                    None,
                )
                if entry is not None:
                    consumers["mHC projection"] = entry
                    gemms = [k for k in after[after.index(entry) :] if cp.base(k["name"]) == "gemm_mxfp8"
                             and k["cap"] == "main"][:2]  # fmt: skip
                    for name, k in zip(("fused Q-A/KV", "Q-B"), gemms):
                        consumers[name] = k
            else:
                kind = "FFN"
                entry = next(
                    (k for k in after if cp.base(k["name"]) == "mhc_project_streams"),
                    None,
                )
                if entry is not None:
                    consumers["mHC projection"] = entry
                    tail = after[after.index(entry) :]
                    router = next(
                        (k for k in tail if cp.base(k["name"]) == "gemm_bf16_partials"),
                        None,
                    )
                    if router is not None:
                        consumers["router"] = router
                    shared = [
                        k
                        for k in tail
                        if k["cap"] == "shared" and cp.base(k["name"]) == "gemm_mxfp8"
                    ][:2]
                    for name, k in zip(("shared gate/up", "shared down"), shared):
                        consumers[name] = k
            out.append({"kind": kind, "prefetch": p, "fork": fork, "issue": fork["end"],
                        "idle_before": None if previous_end is None else max(0.0, fork["end"] - previous_end),
                        "queue": max(0.0, p["ts"] - fork["end"]), "consumers": consumers,
                        "launch": launch})  # fmt: skip
            previous_end = p["end"]
    return out


def analyse(ks, windows_steps):
    cp.label(ks)
    instances, n, span_total = [], 0, 0.0
    on_path = {}  # id(kernel) -> (own time on the path, predecessor)
    fork_gaps = {"fork": [], "other": []}
    contention_rows = []
    join = 0.0
    for a, b in windows_steps:
        step = ks[a:b]
        span_total += ks[b]["ts"] - ks[a]["ts"]
        n += 1
        edges = cp.critical_path(step)
        prefetches = sorted(
            (k for k in step if k["cap"] == "prefetch"), key=lambda k: k["ts"]
        )
        found = windows_of(step)
        forks = {id(w["fork"]) for w in found}
        for pred, k, kind in edges:
            own = k["end"] - (k["ts"] if pred is None else max(k["ts"], pred["end"]))
            on_path[id(k)] = (own, pred)
            if pred is None:
                continue
            if kind == "prefetch-join":
                join += max(0.0, k["ts"] - pred["end"])
            if kind == "chain" and k["launch"][0] == "G":
                gap = max(0.0, k["ts"] - pred["end"])
                fork_gaps["fork" if id(pred) in forks else "other"].append(gap)
        consumers = {id(c) for w in found for c in w["consumers"].values()}
        for pred, k, kind in edges:
            if (
                k["cap"] != "main"
                or k["memcpy"]
                or id(k) in consumers
                or not latency_bound(k)
            ):
                continue
            own = on_path[id(k)][0]
            overlap = sum(
                max(0.0, min(k["end"], p["end"]) - max(k["ts"], p["ts"]))
                for p in prefetches
            )
            contention_rows.append(
                (
                    cp.base(k["name"]),
                    k["grid"],
                    k["dur"],
                    overlap / max(k["dur"], 1e-6),
                    own,
                )
            )
        # the main chain's DRAM-idle time on the path, for the earlier-issue bound
        path_kernels = sorted((k for _, k, _ in edges), key=lambda k: k["ts"])
        for w in found:
            w["dram_idle_before"] = dram_idle(path_kernels, on_path, w)
        for w in found:
            w["step"] = n
        instances.extend(found)

    # consumers: lateness and excess over the median of calls whose prefetch had finished
    finished = defaultdict(list)
    for w in instances:
        for name, c in w["consumers"].items():
            if c["ts"] >= w["prefetch"]["end"]:
                finished[(w["kind"], name)].append(c["dur"])
    typical = {key: statistics.median(v) for key, v in finished.items() if v}
    consumer_rows = defaultdict(
        lambda: {"calls": 0, "late": 0, "excess": 0.0, "path_excess": 0.0}
    )
    for w in instances:
        w["excess"] = 0.0
        w["late"] = max(
            0.0,
            w["prefetch"]["end"]
            - min(
                (c["ts"] for c in w["consumers"].values()), default=w["prefetch"]["end"]
            ),
        )
        for name, c in w["consumers"].items():
            row = consumer_rows[(w["kind"], name)]
            row["calls"] += 1
            if c["ts"] < w["prefetch"]["end"]:
                row["late"] += 1
                excess = max(0.0, c["dur"] - typical.get((w["kind"], name), c["dur"]))
                row["excess"] += excess
                own = on_path.get(id(c), (0.0, None))[0]
                row["path_excess"] += min(excess, own)
                w["excess"] += min(excess, own)

    # bounds
    earlier = defaultdict(float)
    pdl_covered = 0.0
    side_idle = defaultdict(float)
    proportional = defaultdict(float)
    for w in instances:
        if w["late"] > 0:
            # finishing d earlier saves at most d and at most the excess; if the fill and
            # the consumers' reads progress evenly, about the excess times d / lateness
            earlier[w["kind"]] += min(w["dram_idle_before"], w["excess"])
            proportional[w["kind"]] += w["excess"] * min(
                1.0, w["dram_idle_before"] / w["late"]
            )
            side_idle[w["kind"]] += min(w["late"], w["idle_before"] or 0.0, w["excess"])
        for name, c in w["consumers"].items():
            own, pred = on_path.get(id(c), (0.0, None))
            if pred is None or c["ts"] >= w["prefetch"]["end"]:
                continue
            # the window's prefetch still holds DRAM while pred runs: only time it does not
            # overlap counts
            pred_own = on_path.get(id(pred), (0.0, None))[0]
            free = max(
                0.0,
                pred_own
                - max(
                    0.0,
                    min(pred["end"], w["prefetch"]["end"])
                    - max(pred["ts"], w["prefetch"]["ts"]),
                ),
            )
            excess = max(0.0, c["dur"] - typical.get((w["kind"], name), c["dur"]))
            if cp.base(pred["name"]) in cp.CHAINED:
                pdl_covered += min(free, excess, own)
    covered_ids = {id(c) for w in instances for c in w["consumers"].values()}
    uncovered = defaultdict(lambda: {"calls": 0, "path_us": 0.0, "pdl_bound": 0.0})
    for a, b in windows_steps:
        step = ks[a:b]
        prefetches = [k for k in step if k["cap"] == "prefetch"]
        for k in step:
            if (
                id(k) in covered_ids
                or id(k) not in on_path
                or not cp.base(k["name"]).startswith(WEIGHT_GEMMS)
            ):
                continue
            if cp.base(k["name"]) == "moe_grouped_fp8_fp4_gemm":
                continue  # routed experts: the weight is known only once routed
            own, pred = on_path[id(k)]
            key = (cp.base(k["name"]), k["grid"])
            uncovered[key]["calls"] += 1
            uncovered[key]["path_us"] += own
            if pred is not None and cp.base(pred["name"]) in cp.CHAINED:
                pred_own = on_path.get(id(pred), (0.0, None))[0]
                busy = sum(
                    max(0.0, min(pred["end"], p["end"]) - max(pred["ts"], p["ts"]))
                    for p in prefetches
                )
                uncovered[key]["pdl_bound"] += min(max(0.0, pred_own - busy), own)

    # contention: path kernels beside a prefetch against the same kernel and grid alone
    alone = defaultdict(list)
    for name, grid, dur, frac, own in contention_rows:
        if frac < 0.05:
            alone[(name, grid)].append(dur)
    slowed = defaultdict(lambda: [0, 0.0])
    for name, grid, dur, frac, own in contention_rows:
        if frac >= 0.5 and len(alone[(name, grid)]) >= 5:
            extra = min(own, max(0.0, dur - statistics.median(alone[(name, grid)])))
            slowed[(name, grid)][0] += 1
            slowed[(name, grid)][1] += extra

    def pct(us):
        return round(100 * us / span_total, 2)

    kinds = Counter(w["kind"] for w in instances)
    out = {"steps": n, "step_ms": round(span_total / n / 1000, 3), "windows": {}}
    for kind in ("WO", "FFN", "NEXT"):
        ws = [w for w in instances if w["kind"] == kind]
        if not ws:
            continue
        idle = [w["idle_before"] for w in ws if w["idle_before"] is not None]
        out["windows"][kind] = {
            "per_step": round(kinds[kind] / n, 1),
            "prefetch_us_p50": round(
                statistics.median(w["prefetch"]["dur"] for w in ws), 1
            ),
            "queue_us_p50": round(statistics.median(w["queue"] for w in ws), 2),
            "idle_before_us_p50": round(statistics.median(idle), 1) if idle else None,
            "dram_idle_before_us_p50": round(
                statistics.median(w["dram_idle_before"] for w in ws), 1
            ),
            "late_fraction": round(sum(w["late"] > 0 for w in ws) / len(ws), 2),
            "late_us_p50_when_late": round(
                statistics.median([w["late"] for w in ws if w["late"] > 0] or [0]), 1
            ),
            "path_excess_us": round(sum(w["excess"] for w in ws) / n, 1),
            "path_excess_percent": pct(sum(w["excess"] for w in ws)),
        }
    out["consumers"] = {
        f"{kind}/{name}": {"per_step": round(r["calls"] / n, 1), "late_fraction": round(r["late"] / max(r["calls"], 1), 2),
                           "typical_us": round(typical.get((kind, name), 0.0), 1),
                           "excess_us": round(r["excess"] / n, 1), "path_excess_us": round(r["path_excess"] / n, 1)}
        for (kind, name), r in sorted(consumer_rows.items())
    }  # fmt: skip
    total_path_excess = sum(w["excess"] for w in instances)
    out["path_excess"] = {
        "us": round(total_path_excess / n, 1),
        "percent": pct(total_path_excess),
    }
    out["contention"] = {
        "us": round(sum(v[1] for v in slowed.values()) / n, 1),
        "percent": pct(sum(v[1] for v in slowed.values())),
        "top": [{"kernel": name, "grid": list(grid), "per_step": round(v[0] / n, 1), "us": round(v[1] / n, 1)}
                for (name, grid), v in sorted(slowed.items(), key=lambda x: -x[1][1])[:8]],
    }  # fmt: skip
    out["fork_gaps"] = {
        key: {"boundaries_per_step": round(len(v) / n, 1), "median_us": round(statistics.median(v), 2) if v else None,
              "mean_us": round(statistics.mean(v), 2) if v else None}
        for key, v in fork_gaps.items()
    }  # fmt: skip
    out["prefetch_join_us"] = round(join / n, 2)
    out["bounds"] = {
        # FFN and NEXT may move earlier only inside their windows (never after their
        # all-reduce); WO for comparison
        "earlier_issue_us": {k: round(v / n, 1) for k, v in earlier.items()},
        "earlier_ffn_next_percent": pct(earlier["FFN"] + earlier["NEXT"]),
        "earlier_wo_percent": pct(earlier["WO"]),
        "earlier_ffn_next_proportional_percent": pct(
            proportional["FFN"] + proportional["NEXT"]
        ),
        "earlier_if_dram_were_free_percent": pct(side_idle["FFN"] + side_idle["NEXT"]),
        "pdl_prologue_covered_us": round(pdl_covered / n, 1),
        "pdl_prologue_covered_percent": pct(pdl_covered),
        "pdl_prologue_uncovered_us": round(
            sum(v["pdl_bound"] for v in uncovered.values()) / n, 1
        ),
        "pdl_prologue_uncovered_percent": pct(
            sum(v["pdl_bound"] for v in uncovered.values())
        ),
    }
    out["uncovered_weight_gemms_on_path"] = [
        {"kernel": name, "grid": list(grid), "per_step": round(v["calls"] / n, 1), "path_us": round(v["path_us"] / n, 1),
         "pdl_bound_us": round(v["pdl_bound"] / n, 1)}
        for (name, grid), v in sorted(uncovered.items(), key=lambda x: -x[1]["path_us"])[:12]
    ]  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "traces",
        nargs="?",
        default="results/private/determinism/request-inventory-5/ports",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "prefetch_windows.json"),
    )
    ap.add_argument("--scenarios", default=",".join(cp.DECODE))
    ap.add_argument("--window", action="append", default=[])
    args = ap.parse_args()
    for item in args.window:
        name, _, span = item.partition("=")
        first, _, end = span.partition(":")
        cp.WINDOWS[name] = (int(first), int(end) if end else None)
    files = sorted(os.listdir(os.path.join(args.traces, "decode-trace/dgx1")))
    trace = dict(
        zip(
            args.scenarios.split(","),
            (os.path.join(args.traces, "decode-trace/dgx1", f) for f in files),
        )
    )
    out = {"source": args.traces, "decode": {}}
    for name, (skip, last) in ((n, w) for n, w in cp.WINDOWS.items() if n in trace):
        ks = cp.load(trace[name])
        d = out["decode"][name] = analyse(ks, cp.steps(ks, skip, last))
        print(f"{name}: {d['steps']} steps of {d['step_ms']} ms")
        for kind, v in d["windows"].items():
            print(f"  {kind:4} {json.dumps(v)}")
        print(
            f"  path excess {d['path_excess']}, contention {d['contention']['us']} us "
            f"({d['contention']['percent']}%), fork gaps {json.dumps(d['fork_gaps'])}, join {d['prefetch_join_us']} us"
        )
        print(f"  bounds {json.dumps(d['bounds'])}")
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
