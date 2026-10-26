#!/usr/bin/env python3
"""One TP rank's share of the checkpoint loaded by the direct loader, and its weight digest
against an expected one (a kernel-lab script).

usage: rank_digest.py RANK_LOAD CHECKPOINT OUT_DIR --expect SHA256 [--tp 4] [--rank 0]

Runs RANK_LOAD (experiments/2026-10-25-final-names/loader/kernel/rank-load/rank_load.py:
the rank's shard plan read through DirectModelLoader's weights iterator and session,
then weight_digest) with --loader direct, prints its JSON line, then one line comparing
its digest with --expect (the digest the direct and B12X loaders gave on 2026-10-25,
experiments/2026-10-25-final-names/loader/runs/2026-10-25-dgx4/rank-load). Exits 1 when
the load fails its safetensors sample or the digests differ.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank_load")
    parser.add_argument("checkpoint")
    parser.add_argument("out", type=Path)
    parser.add_argument("--expect", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"tensors-direct-tp{args.tp}-rank{args.rank}.json"
    process = subprocess.run(
        [sys.executable, args.rank_load, args.checkpoint, "--loader", "direct",
         "--tp", str(args.tp), "--rank", str(args.rank), "--out", str(out)],
        text=True, stdout=subprocess.PIPE)
    lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
    print(*lines, sep="\n", flush=True)
    digest = json.loads(lines[-1]).get("digest") if lines else None
    equal = process.returncode == 0 and digest == args.expect
    print(json.dumps({"summary": "direct loader digest", "tp": args.tp, "rank": args.rank,
                      "digest": digest, "expected": args.expect, "equal": equal,
                      "exit": process.returncode}), flush=True)
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
