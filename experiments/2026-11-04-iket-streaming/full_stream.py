#!/usr/bin/env python3
"""Unchanged 43-layer dummy-weight workload with the streaming collector."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["full"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--uniform", action="store_true")
args, rest = parser.parse_known_args()
from stream.collector import enable
from stream.export import records

collector = enable(
    args.out / "stream",
    ring_bytes=8 << 20,
    chunk_bytes=64 << 20,
    patched=True,
    uniform_registers=args.uniform,
)
sys.argv = ["/h/iket_step.py", "off", "--out", str(args.out / "model"), *rest]
try:
    runpy.run_path("/h/iket_step.py", run_name="__main__")
finally:
    collector.close()
model = json.loads((args.out / "model/result.json").read_text())
counts = Counter()
for row in records(args.out / "stream"):
    counts["records"] += 1
    counts["kind_" + str(row[9])] += 1
assert counts["kind_0"] == counts["kind_3"]
assert counts["kind_1"] == counts["kind_2"]
summary = json.loads((args.out / "stream/summary.json").read_text())
assert counts["records"] == summary["written_records"]
result = {
    "operation": "full",
    "summary": summary,
    "validation": dict(counts),
    "model": model,
    "tokens_sha256": hashlib.sha256(
        json.dumps(model["token_ids"]).encode()
    ).hexdigest(),
}
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
