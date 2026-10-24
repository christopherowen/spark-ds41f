#!/usr/bin/env python3
"""The serving check's table: each arm's determinism and seeded-sampling results, whether
every arm's reference digests agree, and the distinct-prompt decode rates.

usage: tables_arms.py RUN LABEL [LABEL...]

The ports arm passes when its temperature-0 references (the eight prompts alone and the
long prompt) and its seeded temperature-1 reference have the same SHA-256 digests as
the previous candidate's, and each arm is identical within itself. With
controls_digest.py's records, each generation-control scenario's digests (alone and
eight at once) are compared with the first arm's too.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summary(run: str, stem: str, label: str) -> dict | None:
    path = ROOT / f"results/private/determinism/{run}/{stem}-{label}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return None


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    digests = {}
    for label in labels:
        greedy = summary(run, "determinism", label)
        sampled = summary(run, "sampled", label)
        c1 = summary(run, "c1-distinct", label)
        c8 = summary(run, "c8-distinct", label)
        controls = summary(run, "controls", label)
        digests[label] = (
            greedy and (greedy["reference_sha256"], greedy["long_sha256"]),
            sampled and sampled["reference_sha256"],
            controls and controls["digests"],
        )
        print(f"{label}:")
        print(f"  temperature 0: identical {greedy and greedy['identical']}, "
              f"{greedy and greedy['scenarios']}")
        print(f"  seeded temperature 1: identical {sampled and sampled['identical']}, "
              f"{sampled and sampled['scenarios']}")
        print(f"  c1 distinct: {c1 and c1['tps_mean']} tok/s (±{c1 and c1['ci95_pct']}%), "
              f"c8 distinct: {c8 and c8['tps_mean']} tok/s")
    first = digests[labels[0]]
    for label in labels[1:]:
        same_greedy = digests[label][0] == first[0]
        same_sampled = digests[label][1] == first[1]
        print(f"{label} against {labels[0]}: temperature-0 digests "
              f"{'equal' if same_greedy else 'DIFFER'}, seeded temperature-1 digest "
              f"{'equal' if same_sampled else 'DIFFER'}")
        if digests[label][2] is not None and first[2] is not None:
            differ = sorted(k for k in first[2] if digests[label][2].get(k) != first[2][k])
            print(f"  controls: {len(first[2]) - len(differ)}/{len(first[2])} equal"
                  + (f", DIFFER: {', '.join(differ)}" if differ else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
