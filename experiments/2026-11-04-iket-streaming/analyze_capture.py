#!/usr/bin/env python3
"""Validate each warp's lifetime and phase pairing with bounded live state."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from stream.export import records


def analyze(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("capture did not complete")
    names = {}
    for path in (root / "sources").glob("*.json"):
        manifest = json.loads(path.read_text())
        names[manifest["kernel_id"]] = manifest["kernel"]
    counts = defaultdict(Counter)
    live, phases = set(), {}
    maximum_live = 0
    for row in records(root):
        sequence, _, grid, kernel, cta, warp, _, event, _, kind, _ = row
        key = grid, kernel, cta, warp
        counts[names[kernel]][f"kind_{kind}"] += 1
        if kind == 0:
            if key in live:
                raise ValueError(f"duplicate lifetime at record {sequence}: {key}")
            live.add(key)
            maximum_live = max(maximum_live, len(live))
        elif kind == 1:
            if key not in live or key in phases:
                raise ValueError(f"invalid phase start at record {sequence}: {key}")
            phases[key] = event
        elif kind == 2:
            if key not in phases:
                raise ValueError(f"unmatched phase end at record {sequence}: {key}")
            del phases[key]
        elif kind == 3:
            if key not in live or key in phases:
                raise ValueError(f"invalid lifetime end at record {sequence}: {key}")
            live.remove(key)
        else:
            raise ValueError(f"unknown record kind {kind}")
    if live or phases:
        raise ValueError(f"unclosed lifetimes={len(live)}, phases={len(phases)}")
    return {
        "summary": summary,
        "maximum_open_warp_lifetimes": maximum_live,
        "kernels": dict(sorted(counts.items())),
        "per_warp_pairing_valid": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    else:
        print(json.dumps(result, indent=2))
