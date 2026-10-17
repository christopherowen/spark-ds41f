#!/usr/bin/env python3
"""Every JIT compile of a traced boot, by startup phase.

usage: summarize_jittrace.py DIR [DIR...]   (each DIR holds jittrace's <host>-<pid>.jsonl)

Prints, per family and phase, the compiles and seconds on each host, then every distinct
kernel that compiled after load_model (memory profiling, KV init, warm-up and capture,
serving): the ones a prepare stage must move ahead of the startup rendezvous.
"""
import collections
import json
import sys
from pathlib import Path

# Compiles in determine_available_memory before _prepare_kernels (r6u) are counted too.
DISTRIBUTED = ("determine_available_memory", "after_prepare_kernels",
               "after_determine_available_memory",
               "initialize_from_config", "after_initialize_from_config",
               "compile_or_warm_up_model", "serving")


def records(directories):
    for directory in directories:
        for path in sorted(Path(directory).glob("*.jsonl")):
            host = path.stem.rsplit("-", 1)[0]
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    record["host"] = host
                    yield record


def main(directories):
    rows = list(records(directories))
    compiles = [r for r in rows if r["event"] == "compile"]
    hosts = sorted({r["host"] for r in compiles})
    table = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for r in compiles:
        cell = table[(r["family"], r["phase"])][r["host"]]
        cell[0] += 1
        cell[1] += r.get("seconds") or 0.0
    print(f"{'family':9s} {'phase':34s} " + " ".join(f"{h:>16s}" for h in hosts))
    for (family, phase), cells in sorted(table.items()):
        print(f"{family:9s} {phase:34s} " + " ".join(
            f"{cells[h][0]:5d} {cells[h][1]:8.1f}s" if h in cells else f"{'':>16s}" for h in hosts))
    late = collections.defaultdict(lambda: {"hosts": set(), "seconds": 0.0, "phases": set()})
    for r in compiles:
        if r["phase"] in DISTRIBUTED:
            entry = late[(r["family"], r.get("factory") or r["name"], r.get("key"))]
            entry["hosts"].add(r["host"])
            entry["seconds"] = max(entry["seconds"], r.get("seconds") or 0.0)
            entry["phases"].add(r["phase"])
    print(f"\n{len(late)} distinct kernels compiled after load_model:")
    for (family, factory, key), entry in sorted(late.items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"  {family:8s} {entry['seconds']:6.1f}s {','.join(sorted(entry['phases'])):40s} {factory} {key or ''}"[:400])
    errors = [r for r in rows if r["event"] == "trace_error"]
    for r in errors:
        print("trace error:", r["host"], r.get("module"), r.get("error"))


if __name__ == "__main__":
    main(sys.argv[1:])
