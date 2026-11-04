#!/usr/bin/env python3
"""Summarize retained node evidence without loading event captures into memory."""

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            checksum.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": checksum.hexdigest(),
    }


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(root):
    runs = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        identity = read(folder / "identity.json")
        if not identity:
            continue
        memory = defaultdict(list)
        if (folder / "memory.jsonl").exists():
            with (folder / "memory.jsonl").open() as handle:
                for line in handle:
                    sample = json.loads(line)
                    memory[sample.get("job")].append(sample)
        jobs = {}
        results = read(folder / "results.json") or []
        by_label = {row["job"]["label"]: row for row in results}
        for job in identity["matrix"]:
            label = job["label"]
            dest = folder / label
            trace = dest / "stream"
            samples = memory[label]
            row = {
                "result": by_label.get(label),
                "command": read(folder / (label + ".command.json")),
                "collector": read(trace / "identity.json"),
                "summary": read(trace / "summary.json"),
            }
            log = folder / (label + ".log")
            if log.exists():
                row["log"] = digest(log)
                if not row["result"] or row["result"]["exit"]:
                    with log.open(errors="replace") as handle:
                        row["failure_tail"] = list(deque(handle, maxlen=20))
            if samples:
                row["memory"] = {
                    "samples": len(samples),
                    "first": samples[0],
                    "last": samples[-1],
                    "minimum_available_bytes": min(
                        s["available_bytes"] for s in samples
                    ),
                    "maximum_temperature_c": max(s["hottest_c"] for s in samples),
                }
            chunks = sorted(trace.glob("events-*.bin"))
            row["native_chunks"] = {
                "completed": len(chunks),
                "bytes": sum(p.stat().st_size for p in chunks),
                "receipts": [read(p.with_suffix(".bin.json")) for p in chunks],
                "partial_bytes": sum(p.stat().st_size for p in trace.glob("*.partial")),
            }
            jobs[label] = row
        runs[folder.name] = {
            "identity": identity,
            "completed": read(folder / "completed.json"),
            "interruption": read(folder / "interruption.json"),
            "memory_log": digest(folder / "memory.jsonl")
            if (folder / "memory.jsonl").exists()
            else None,
            "jobs": jobs,
        }
    return {"evidence_root": str(root), "runs": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root), indent=2))
