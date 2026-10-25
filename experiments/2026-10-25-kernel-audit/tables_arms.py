#!/usr/bin/env python3
"""The kernel audit's serving-check table: each arm's temperature-0 determinism, with
and without log probabilities, whether its reference digests equal the first arm's, and
the distinct-prompt decode rates at one, eight and sixteen streams.

usage: tables_arms.py RUN LABEL [LABEL...]

The audit passes when every arm is identical within itself and each audit arm's digests
(the eight prompts alone and the long prompt, as tokens with log probabilities and as
text) equal candidate 6's: the audit changes no output.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = ("reference_sha256", "long_sha256", "reference_text_sha256", "long_text_sha256")


def summary(run: str, stem: str, label: str) -> dict | None:
    path = ROOT / f"results/private/determinism/{run}/{stem}-{label}.jsonl"
    for line in reversed(path.read_text().splitlines() if path.exists() else []):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return None


def rate(record: dict | None) -> str:
    return f"{record['tps_mean']} tok/s (±{record['ci95_pct']}%)" if record else "missing"


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    digests = {}
    for label in labels:
        runs = {stem: summary(run, stem, label) for stem in ("determinism", "greedy")}
        digests[label] = {stem: r and tuple(r.get(key) for key in DIGESTS)
                          for stem, r in runs.items()}
        print(f"{label}:")
        for stem, r in runs.items():
            print(f"  {stem}: identical {r and r['identical']}, {r and r['scenarios']}")
        print(f"  c1 distinct {rate(summary(run, 'c1-distinct', label))}, "
              f"c8 distinct {rate(summary(run, 'c8-distinct', label))}, "
              f"c16 distinct {rate(summary(run, 'c16-distinct', label))}")
    first = digests[labels[0]]
    for label in labels[1:]:
        same = {stem: digests[label][stem] == first[stem] for stem in first}
        print(f"{label} against {labels[0]}: "
              + ", ".join(f"{stem} digests {'equal' if equal else 'DIFFER'}"
                          for stem, equal in same.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
