#!/usr/bin/env python3
"""The lab table for a suite job: servebench comparisons between its arms' suite runs.

usage: scripts/servebench_suite_table.py RUN LABEL [LABEL...]

Arms run candidate first (heat soak), then the reference, then the candidate again. With
two or more arms, the second arm is the reference: it is compared against every other arm,
and the first against the last as the run's own repeat.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from servebench.__main__ import main  # noqa: E402


def pairs(labels: list[str]) -> list[tuple[str, str]]:
    if len(labels) < 2:
        return []
    reference = labels[1]
    out = [(reference, label) for label in labels if label != reference]
    if len(labels) > 2:
        out.append((labels[0], labels[-1]))
    return out


if __name__ == "__main__":
    run, labels = sys.argv[1], sys.argv[2:]
    for base, candidate in pairs(labels):
        print(f"\n######## {base} -> {candidate}", flush=True)
        main(["compare", str(ROOT / "results/private/suites" / run / base),
              str(ROOT / "results/private/suites" / run / candidate)])
