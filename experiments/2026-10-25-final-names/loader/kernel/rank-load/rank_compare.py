#!/usr/bin/env python3
"""The direct loader against B12X's on one rank's shard set (a kernel-lab script).

usage: rank_compare.py CHECKPOINT OUT_DIR [--tp 4] [--rank 0]

Runs rank_load.py with the direct loader, then with B12X's, each in a process of its own
so neither inherits the other's memory, and passes when:
- both loads pass their safetensors sample;
- the weight digests are equal;
- every tensor's SHA-256 is equal;
- the file-source descriptors (DS4.1's Engram tables) are equal.
It prints each load's JSON line, then the comparison's.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out", type=Path)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, ok = {}, True
    for loader in ("direct", "b12x"):
        out = args.out / f"tensors-{loader}-tp{args.tp}-rank{args.rank}.json"
        out.unlink(missing_ok=True)  # never read an earlier run's
        process = subprocess.run(
            [sys.executable, str(HERE / "rank_load.py"), args.checkpoint, "--loader", loader,
             "--tp", str(args.tp), "--rank", str(args.rank), "--out", str(out)],
            text=True, stdout=subprocess.PIPE)
        lines = [line for line in process.stdout.splitlines() if line.startswith("{")]
        print(*(lines or [f'{{"loader": "{loader}", "error": "exit {process.returncode}"}}']), sep="\n",
              flush=True)
        ok &= process.returncode == 0 and out.is_file()
        if out.is_file():
            results[loader] = json.loads(out.read_text())
    differ = []
    if len(results) == 2:
        direct, b12x = results["direct"]["tensors"], results["b12x"]["tensors"]
        differ = sorted(name for name in direct.keys() | b12x.keys() if direct.get(name) != b12x.get(name))
    descriptors_equal = len(results) == 2 and results["direct"]["descriptors"] == results["b12x"]["descriptors"]
    equal = (len(results) == 2 and results["direct"]["digest"] == results["b12x"]["digest"] and not differ
             and descriptors_equal)
    print(json.dumps({"summary": "direct vs b12x", "tp": args.tp, "rank": args.rank, "equal": equal,
                      "tensors_differ": len(differ), "first_differ": differ[:10],
                      "descriptors_equal": descriptors_equal,
                      "descriptors": len(results.get("direct", {}).get("descriptors", {}))}), flush=True)
    return 0 if ok and equal else 1


if __name__ == "__main__":
    sys.exit(main())
