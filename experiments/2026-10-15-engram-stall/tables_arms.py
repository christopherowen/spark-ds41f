#!/usr/bin/env python3
"""The engram-stall run's table: each boot's lone request, hung or not.

usage: tables_arms.py RUN LABEL [LABEL...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    for label in labels:
        path = ROOT / f"results/private/determinism/{run}/probe-{label}.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        try:
            summary = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            print(f"{label}: no probe result")
            continue
        state = "HUNG" if summary["hung"] else "ok"
        print(f"{label}: {state} {json.dumps(summary['request'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
