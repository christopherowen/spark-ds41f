#!/usr/bin/env python3
"""Attribute the TP3 budget sweep's memory by budget, node and start-up phase.

usage: analyze_sweep.py OUT [--arms b6144=6144 b8192=8192 b12288=12288]

OUT is the watcher's output directory (memory/watch.sh): <node>-samples.jsonl from
sampler.py, engine-<node>-<container>.log from the followers, and
traces-<node>/<arm>/{torch,pools}-rank<rank>.{tsv,jsonl} from the overlay (an arm's
SPARK_DS41F_MEMORY_TRACE directory). A trace file holds one section per boot (each
starts with its header line); an arm that runs twice (b8192 and b8192-again) has two.
``--arms`` names each trace directory with its budget (the memory-cuts check:
b8192=8192 cuts=8192).

For every boot and node it prints, at each phase mark of the overlay (weights, profiled:
after the profile run and its release, kv-pool, warmed: graph capture and the tail-shape
warm-up done):
- torch's allocated and reserved GiB;
- MemAvailable, the driver's share (what /proc/meminfo leaves unaccounted: GPU
  allocations live in system memory on GB10 but belong to no process) and anonymous
  memory, from the sampler sample nearest the mark;
then the boot's lowest MemAvailable (what the 5 GiB start-up guard sees), the serving
peaks (torch reserved and allocated) and lowest MemAvailable until the next boot, the
workspace slot lines (VLLM_DEBUG_WORKSPACE) and the allocator pools. Last, per phase,
the change between consecutive arms on each node: per 1,024 budget tokens where their
budgets differ, in GiB where they are the same.
"""
import argparse
import bisect
import csv
import glob
import json
import os
import re
from collections import defaultdict

G = 2**30
KIB_PER_GIB = 2**20
PHASES = ("weights", "profiled", "kv-pool", "warmed")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z ")


def boots(path: str) -> list[dict]:
    """The boots of one trace file: rows (t, allocated, reserved) and phase marks."""
    out, cur = [], None
    with open(path) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if row[0] == "t":
                cur = {"rows": [], "marks": {}}
                out.append(cur)
            elif row[0].startswith("# phase") and cur is not None:
                _, _, name, stamp = row[0].split()
                cur["marks"].setdefault(name, float(stamp))
            elif cur is not None:
                cur["rows"].append((float(row[0]), int(row[2] or 0), int(row[3] or 0)))
    return [b for b in out if b["rows"]]


def samples(path: str) -> dict:
    meminfo, gpu = [], []
    for line in open(path):
        record = json.loads(line)
        if record["k"] == "m":
            mi = record["mi"]
            driver = (mi["MemTotal"] - mi["MemFree"] - mi["Buffers"] - mi["Cached"] - mi["SwapCached"]
                      - mi["AnonPages"] - mi["KReclaimable"] - mi["SUnreclaim"] - mi["KernelStack"]
                      - mi["PageTables"] - mi["SecPageTables"] - mi["Percpu"])
            meminfo.append((record["t"], mi["MemAvailable"] / KIB_PER_GIB, driver / KIB_PER_GIB,
                            mi["AnonPages"] / KIB_PER_GIB))
        elif record["k"] == "g":
            gpu.append((record["t"], sum(app[2] for app in record.get("apps") or []) / 1024))
    return {"m": meminfo, "m_t": [m[0] for m in meminfo], "g": gpu}


def nearest(series, times, t):
    i = min(bisect.bisect_left(times, t), len(series) - 1)
    return series[i] if series else None


def lowest(series, start, stop):
    window = [m for m in series if start <= m[0] < stop]
    return min(window, key=lambda m: m[1]) if window else None


def workspace_lines(out: str, node: str, start: float, stop: float) -> list[str]:
    import datetime as dt

    lines = []
    for path in glob.glob(os.path.join(out, f"engine-{node}-*.log")):
        for line in open(path, errors="replace"):
            if "[WORKSPACE DEBUG]" not in line:
                continue
            match = STAMP.match(line)
            if not match:
                continue
            t = dt.datetime.fromisoformat(match.group(1)[:26]).replace(tzinfo=dt.timezone.utc).timestamp()
            if start <= t < stop:
                lines.append(line.split("[WORKSPACE DEBUG]", 1)[1].strip()[:160])
    return lines


def pools(out: str, node: str, arm: str, start: float, stop: float) -> list[dict]:
    found = []
    for path in glob.glob(os.path.join(out, f"traces-{node}", arm, "pools-rank*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            if start <= record["t"] < stop:
                found.append(record)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--arms", nargs="+", default=["b6144=6144", "b8192=8192", "b12288=12288"])
    args = parser.parse_args()
    arms = [(name, int(budget)) for name, budget in (a.split("=") for a in args.arms)]
    nodes = sorted(os.path.basename(p).split("-samples")[0]
                   for p in glob.glob(os.path.join(args.out, "*-samples.jsonl")))
    table = defaultdict(dict)  # (node, arm) -> phase -> values of the arm's first boot
    for node in nodes:
        sampled = samples(os.path.join(args.out, f"{node}-samples.jsonl"))
        sections = []
        for arm, budget in arms:
            for path in glob.glob(os.path.join(args.out, f"traces-{node}", arm, "torch-rank*.tsv")):
                for index, boot in enumerate(boots(path)):
                    sections.append((boot["rows"][0][0], arm, index, boot))
        sections.sort(key=lambda s: s[0])
        for k, (start, arm, index, boot) in enumerate(sections):
            stop = sections[k + 1][0] if k + 1 < len(sections) else boot["rows"][-1][0] + 1
            label = arm + ("-again" if index else "")
            print(f"== {node} {label} (trace from {start:.0f})")
            print(f"  {'phase':10s} {'alloc':>7s} {'reserv':>7s} | {'avail':>6s} {'driver':>6s} {'anon':>5s}")
            times = [r[0] for r in boot["rows"]]
            for phase in PHASES:
                t = boot["marks"].get(phase)
                if t is None:
                    continue
                row = boot["rows"][max(0, bisect.bisect_right(times, t) - 1)]
                m = nearest(sampled["m"], sampled["m_t"], t)
                values = {"alloc": row[1] / G, "reserved": row[2] / G}
                if m:
                    values.update(avail=m[1], driver=m[2], anon=m[3])
                if not index:
                    table[(node, arm)][phase] = values
                print(f"  {phase:10s} {values['alloc']:7.2f} {values['reserved']:7.2f} | "
                      f"{values.get('avail', float('nan')):6.2f} {values.get('driver', float('nan')):6.2f} "
                      f"{values.get('anon', float('nan')):5.2f}")
            warmed = boot["marks"].get("warmed", start)
            boot_low = lowest(sampled["m"], start - 30, warmed + 5)
            serve = [r for r in boot["rows"] if r[0] > warmed]
            serve_low = lowest(sampled["m"], warmed + 5, stop)
            summary = {
                "boot_low_avail": boot_low[1] if boot_low else None,
                "serve_low_avail": serve_low[1] if serve_low else None,
                "serve_max_reserved": max(r[2] for r in serve) / G if serve else None,
                "serve_max_alloc": max(r[1] for r in serve) / G if serve else None,
            }
            if not index:
                table[(node, arm)]["summary"] = summary
            print("  " + ", ".join(f"{key} {value:.2f}" for key, value in summary.items() if value is not None))
            for line in workspace_lines(args.out, node, start - 60, stop):
                print(f"  workspace: {line}")
            for record in pools(args.out, node, arm, start, stop):
                parts = ", ".join(f"pool {pool}: {v['total_size'] / G:.2f} GiB reserved, "
                                  f"{v['allocated_size'] / G:.2f} allocated, {v['segments']} segments"
                                  for pool, v in record["pools"].items())
                print(f"  pools {record['label']}: {parts}")
    print("== change between consecutive arms (GiB; per 1,024 tokens where budgets differ)")
    for node in nodes:
        present = [(arm, budget) for arm, budget in arms if (node, arm) in table]
        for (low, low_budget), (high, high_budget) in zip(present, present[1:]):
            scale = 1024 / (high_budget - low_budget) if high_budget != low_budget else 1.0
            parts = []
            for phase in (*PHASES, "summary"):
                a, b = table[(node, low)].get(phase, {}), table[(node, high)].get(phase, {})
                for key in ("alloc", "reserved", "avail", "driver", "anon", "boot_low_avail",
                            "serve_low_avail", "serve_max_reserved"):
                    if a.get(key) is not None and b.get(key) is not None:
                        parts.append(f"{phase}.{key} {(b[key] - a[key]) * scale:+.3f}")
            print(f"  {node} {low}->{high}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
