#!/usr/bin/env python3
"""Each kernel's share of a profiled decode step or prefill, critical-path aware.

usage: kernel_shares.py RUN_DIR LABEL [--node dgx1] [--decode NAME,...]
                        [--prefill NAME,...] [--top N] [--json OUT]

RUN_DIR/LABEL holds the request inventory's traces (decode-trace/<node>/ and
prefill-trace/<node>/, one profiler window a scenario, in the order decode.json and
prefill.json list them; see experiments/2026-10-25-request-inventory). For each named
scenario:

- **Steps.** A step runs from one ``execute_context_*`` GPU annotation on the eager
  stream to the next (the target's forward, the head, the sampler and the drafter);
  decode steps are those without context tokens, the prefill the step(s) with the
  scenario's most context tokens (the prompt's).
- **Busy time.** The union of the step's kernel, copy and fill intervals on every
  stream, ``l2_prefetch_kernel`` excluded: it runs beside the layers by design, and removing
  it would not shorten a step.
- **Attribution.** While a routed-MoE side-stream kernel runs (a stream that runs
  ``moe_dispatch_kernel`` or ``router_partials_kernel``), the time is charged to the
  side-stream kernels: in every decode layer that branch (router, dispatch, grouped
  GEMMs) is the longer one, so the shared expert on the main stream has slack.
  Otherwise the time is split evenly among the active kernels. A kernel's share is
  its charged time over the busy time: what removing it entirely would save, at most.

Prints, per kernel name (every grid together), the share, launches a step and mean
duration for each scenario; ``--json`` writes the same numbers.
"""
import bisect
import collections
import gzip
import json
import re
import sys
from pathlib import Path

PREFETCH = {"l2_prefetch_kernel"}
SIDE = ("moe_dispatch_kernel", "router_partials_kernel")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def scenario_traces(run: Path, label: str, node: str, workload: str) -> dict[str, Path]:
    lines = [json.loads(line) for line in (run / label / f"{workload}.json").read_text().splitlines()
             if line.startswith("{")]
    names = [line["scenario"] for line in lines if line.get("status", 200) == 200]
    traces = sorted((run / label / f"{workload}-trace" / node).glob("*.json.gz"))
    if len(names) != len(traces):
        raise SystemExit(f"{workload}: {len(names)} scenarios but {len(traces)} traces")
    return dict(zip(names, traces))


def load(trace: Path):
    events = json.load(gzip.open(trace))["traceEvents"]
    # Kernels and device copies and fills alike: each occupies its stream.
    kernels = sorted((e["ts"], e["dur"], e["name"], e["args"].get("stream"))
                     for e in events if e.get("ph") == "X"
                     and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    annotations = [e for e in events if e.get("cat") == "gpu_user_annotation"
                   and e["name"].startswith("execute_context")]
    # The eager stream: the one carrying most step annotations.
    stream = collections.Counter(e["tid"] for e in annotations).most_common(1)[0][0]
    steps = sorted((e["ts"], e["name"]) for e in annotations if e["tid"] == stream)
    end = max(k[0] + k[1] for k in kernels)
    windows = []
    for i, (start, name) in enumerate(steps):
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
        context, generation = int(m.group(1)), int(m.group(3))
        if context == 0 and generation == 0:
            continue  # a dummy step
        stop = steps[i + 1][0] if i + 1 < len(steps) else end
        windows.append((start, stop, int(m.group(2)) if context else 0))
    # The prompt's prefill: the steps with the most context tokens (a request's
    # first decode step can still count a few).
    most = max(tokens for _, _, tokens in windows)
    windows = [(a, b, "prefill" if tokens and tokens == most else "decode" if not tokens
                else "other") for a, b, tokens in windows]
    return kernels, windows


def shares(kernels, windows, phase: str) -> dict:
    spans = [(a, b) for a, b, kind in windows if kind == phase]
    starts = [a for a, _ in spans]
    inside = []
    for k in kernels:
        i = bisect.bisect_right(starts, k[0]) - 1
        if i >= 0 and k[0] <= spans[i][1]:
            inside.append(k)
    side = {k[3] for k in inside if k[2] in SIDE}
    points = []
    for i, k in enumerate(inside):
        if k[2] not in PREFETCH:
            points += [(k[0], 1, i), (k[0] + k[1], -1, i)]
    points.sort(key=lambda p: (p[0], p[1]))
    active, last, busy = set(), None, 0.0
    charged = collections.defaultdict(float)
    for t, kind, i in points:
        if last is not None and active:
            dt = t - last
            busy += dt
            pick = [j for j in active if inside[j][3] in side] or list(active)
            for j in pick:
                charged[j] += dt / len(pick)
        (active.add if kind == 1 else active.discard)(i)
        last = t
    table = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for i, k in enumerate(inside):
        row = table[k[2]]
        row[0] += 1
        row[1] += k[1]
        row[2] += charged.get(i, 0.0)
    return {"steps": len(spans), "busy_us": busy,
            "kernels": {name: {"launches": c, "us": t, "charged_us": e}
                        for name, (c, t, e) in table.items()}}


def main() -> None:
    run, label = Path(sys.argv[1]), sys.argv[2]
    node = arg("--node", "dgx1")
    top = int(arg("--top", "60"))
    results = {}
    for workload, option, phase in (("decode", "--decode", "decode"),
                                    ("prefill", "--prefill", "prefill")):
        wanted = arg(option)
        if not wanted:
            continue
        traces = scenario_traces(run, label, node, workload)
        for name in wanted.split(","):
            kernels, windows = load(traces[name])
            results[f"{name} ({phase})"] = shares(kernels, windows, phase)
    if arg("--json"):
        Path(arg("--json")).write_text(json.dumps(results, indent=1) + "\n")
    names = set().union(*(r["kernels"] for r in results.values()))

    def weight(name):
        return max(r["kernels"].get(name, {"charged_us": 0})["charged_us"] / r["busy_us"]
                   for r in results.values())

    print(f"{'kernel':48}" + "".join(
        f" | {key[:24]:>24} {r['busy_us'] / r['steps'] / 1000:7.2f} ms/step"
        for key, r in results.items()))
    for name in sorted(names, key=weight, reverse=True)[:top]:
        line = f"{name[:48]:48}"
        for r in results.values():
            k = r["kernels"].get(name)
            line += (f" | {100 * k['charged_us'] / r['busy_us']:6.2f}% "
                     f"{k['launches'] / r['steps']:6.1f}/step {k['us'] / k['launches']:8.2f} us"
                     if k else " |" + " " * 39)
        print(line)


if __name__ == "__main__":
    main()
