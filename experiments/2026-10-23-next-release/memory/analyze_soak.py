#!/usr/bin/env python3
"""Host-side memory growth over each soak, from sampler.py's records.

usage: analyze_soak.py SAMPLES_DIR SOAK.jsonl [SOAK.jsonl ...]

Each SOAK.jsonl is soak.py's output for one arm (its first and last lines give the
soak's start and end). For every node and every container process other than the GPU
workers that lived through a soak, prints its RSS when the soak started, when it ended
and at its highest, and the change; for the API server and engine core (the processes
sampled by mapping kind), the change by kind: the brk heap, CPython's 1 MiB arenas,
glibc's 64 MiB thread arenas, other anonymous mappings by size, files. Also each node's
lowest MemAvailable during the soak.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

GiB_KB = 2**20


def soak_window(path: str) -> tuple[str, float, float]:
    lines = [json.loads(line) for line in open(path) if line.startswith("{")]
    start = next(r["t"] for r in lines if r.get("event") == "start")
    end = max(r["t"] for r in lines if "t" in r)
    return Path(path).stem.removeprefix("soak-"), start, end


def load(path: Path):
    names, rss, maps, avail = {}, defaultdict(list), defaultdict(list), []
    for line in open(path):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        kind, t = record.get("k"), record.get("t")
        if kind == "new":
            names[record["pid"]] = record["comm"]
        elif kind == "p":
            for pid, kb in record["rss"]:
                if kb is not None:
                    rss[pid].append((t, kb))
        elif kind == "x":
            for pid, kinds in record["maps"].items():
                if kinds:
                    maps[int(pid)].append((t, kinds))
        elif kind == "m":
            value = record["mi"].get("MemAvailable")
            if value is not None:
                avail.append((t, value))
    return names, rss, maps, avail


def nearest(series, t):
    return min(series, key=lambda item: abs(item[0] - t)) if series else None


def main() -> None:
    samples_dir, soaks = Path(sys.argv[1]), [soak_window(p) for p in sys.argv[2:]]
    for path in sorted(samples_dir.glob("*-samples.jsonl")):
        node = path.name.split("-samples")[0]
        names, rss, maps, avail = load(path)
        for label, start, end in soaks:
            low = min((kb for t, kb in avail if start <= t <= end), default=None)
            print(f"== {node} {label}: lowest MemAvailable "
                  f"{low / GiB_KB:.2f} GiB" if low else f"== {node} {label}: no MemAvailable samples")
            for pid, series in sorted(rss.items()):
                comm = names.get(pid, "?")
                if comm.startswith("VLLM::Worker"):
                    continue
                inside = [(t, kb) for t, kb in series if start <= t <= end]
                if len(inside) < 2 or inside[0][0] > start + 120 or inside[-1][0] < end - 120:
                    continue
                first, last, high = inside[0][1], inside[-1][1], max(kb for _, kb in inside)
                if high < 100 * 1024:
                    continue
                print(f"  {comm:<18} pid {pid:<8} start {first / GiB_KB:6.3f}  end {last / GiB_KB:6.3f}  "
                      f"high {high / GiB_KB:6.3f}  change {(last - first) / GiB_KB:+.3f} GiB")
                kinds = maps.get(pid)
                if kinds:
                    a, b = nearest(kinds, start), nearest(kinds, end)
                    for name in sorted(set(a[1]) | set(b[1])):
                        before = a[1].get(name, {"rss": 0, "n": 0})
                        after = b[1].get(name, {"rss": 0, "n": 0})
                        delta = (after["rss"] - before["rss"]) / 1024
                        if abs(delta) >= 1 or after["rss"] >= 64 * 1024:
                            print(f"      {name:<13} {before['rss'] / 1024:9.1f} -> {after['rss'] / 1024:9.1f} MiB "
                                  f"({delta:+.1f}), mappings {before['n']} -> {after['n']}")


if __name__ == "__main__":
    main()
