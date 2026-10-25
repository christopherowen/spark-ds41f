#!/usr/bin/env python3
"""The MoE exchange's serving-check table: each arm's temperature-0 determinism, with and
without log probabilities and with a 16K long prompt, and whether its reference digests
equal candidate 6's.

usage: tables_arms.py RUN LABEL [LABEL...]

RUN is exchange-serving.json's moe-exchange-serving, its arms exchange, candidate6 and
exchange-2. The change passes when every arm is identical within itself and each
exchange arm's digests (the eight prompts alone and the long prompt, as tokens with log
probabilities and as text) equal candidate 6's: the change moves no output. The long
prompts run sequence-parallel: about 10.5K tokens at 700 lines (two exchange slices a
layer) and 16.5K at 1,100 (a 16,288-token chunk, three slices, then its tail). The
prefill rates are servebench's: scripts/servebench_suite_table.py RUN LABEL [LABEL...].

The kernel audit's tables_arms.py, with the 16K prompt in place of the decode rates
(this change runs only in prefill).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = ("reference_sha256", "long_sha256", "reference_text_sha256", "long_text_sha256")
STEMS = ("determinism", "greedy", "determinism-16k")


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


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    digests = {}
    for label in labels:
        runs = {stem: summary(run, stem, label) for stem in STEMS}
        digests[label] = {stem: r and tuple(r.get(key) for key in DIGESTS)
                          for stem, r in runs.items()}
        print(f"{label}:")
        for stem, r in runs.items():
            print(f"  {stem}: identical {r and r['identical']}, {r and r['scenarios']}")
    reference = "candidate6" if "candidate6" in labels else labels[0]
    for label in labels:
        if label == reference:
            continue
        same = {stem: digests[label][stem] == digests[reference][stem] for stem in STEMS}
        print(f"{label} against {reference}: "
              + ", ".join(f"{stem} digests {'equal' if equal else 'DIFFER'}"
                          for stem, equal in same.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
