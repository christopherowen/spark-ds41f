#!/usr/bin/env python3
"""Boot timeline from boot.sh's saved logs.

usage: timeline.py results/private/boot/LABEL [...]

Times are seconds after the launcher started. One row per boot, phases as
columns, plus NCCL's per-communicator init timings when they were logged.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

MILESTONES = [
    ("engine", r"Initializing a V1 LLM engine"),
    ("workers", r"Registered model loader"),
    ("routing", r"b12x routing checkpoint shards"),
    ("weights", r"Loading weights took"),
    ("loaded", r"Model loading took"),
    ("kv", r"GPU KV cache size"),
    ("warmup", r"JIT kernel warmup starting"),
    ("capture", r"Capturing model for DSpark speculator"),
    ("captured", r"Graph capturing finished"),
    ("api", r"Application startup complete"),
]


def stamp(text: str) -> datetime:
    return datetime.fromisoformat(text.rstrip("Z")[:26])


def boot(directory: Path) -> dict:
    t0 = stamp((directory / "launch_t0").read_text().strip())
    row = {"label": directory.name}
    for node in ("dgx2", "dgx3", "dgx1"):
        started = directory / f"{node}.started"
        if started.exists() and started.read_text().strip():
            row[f"{node}_start"] = (stamp(started.read_text().strip()) - t0).total_seconds()
    seen = set()
    for line in (directory / "dgx1.log").read_text(errors="replace").splitlines():
        m = re.match(r"(\S+Z) (.*)", line)
        if not m:
            continue
        for name, pattern in MILESTONES:
            if name not in seen and re.search(pattern, m.group(2)):
                seen.add(name)
                row[name] = (stamp(m.group(1)) - t0).total_seconds()
    for line in (directory / "launcher.log").read_text().splitlines():
        m = re.match(r"(\S+Z) (.*)", line)
        if m and m.group(2).startswith("API ready at"):
            row["api_seen"] = (stamp(m.group(1)) - t0).total_seconds()
        if m and m.group(2).startswith("cluster ready"):
            row["ready"] = (stamp(m.group(1)) - t0).total_seconds()
    nccl = []
    for node in ("dgx1", "dgx2", "dgx3"):
        log = directory / f"{node}.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                m = re.search(r"Init timings - (\S+): rank (\d+) nranks \d+ total ([\d.]+) \((.*)\)", line)
                if m:
                    nccl.append(f"{node}:{m.group(1)} r{m.group(2)} {m.group(3)}s ({m.group(4)})")
    row["nccl"] = nccl
    return row


rows = [boot(Path(arg)) for arg in sys.argv[1:]]
cols = ["dgx2_start", "dgx3_start", "dgx1_start", "engine", "workers", "routing", "weights",
        "loaded", "kv", "warmup", "capture", "captured", "api", "api_seen", "ready"]
print("label".ljust(22) + "".join(c.replace("_start", "").rjust(9) for c in cols))
for r in rows:
    print(r["label"][:22].ljust(22) + "".join((f"{r[c]:9.1f}" if c in r else "        -") for c in cols))
for r in rows:
    for line in r["nccl"][:12]:
        print(f"  {r['label']}: {line}")
    with open(Path(sys.argv[1]).parent / "summary.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
