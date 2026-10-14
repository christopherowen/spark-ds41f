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

from servebench import suite  # noqa: E402
from servebench.__main__ import compare_results, main, read_json  # noqa: E402


def pairs(labels: list[str]) -> list[tuple[str, str]]:
    if len(labels) < 2:
        return []
    reference = labels[1]
    out = [(reference, label) for label in labels if label != reference]
    if len(labels) > 2:
        out.append((labels[0], labels[-1]))
    return out


def directory(run: str, label: str) -> str:
    return str(ROOT / "results/private/suites" / run / label)


if __name__ == "__main__":
    run, labels = sys.argv[1], sys.argv[2:]
    if len(labels) > 2:
        # The verdict first: reference -> candidate, judged against the candidate's own repeat.
        judge = lambda b, c: compare_results(b, c, 3.0)
        reference_to_candidate = suite.compare(read_json, directory(run, labels[1]), directory(run, labels[0]), judge)
        repeat = suite.compare(read_json, directory(run, labels[0]), directory(run, labels[-1]), judge)
        print(suite.format_floor(suite.with_floor(reference_to_candidate, repeat), labels[1], labels[0], labels[-1]))
    for base, candidate in pairs(labels):
        print(f"\n######## {base} -> {candidate}", flush=True)
        main(["compare", directory(run, base), directory(run, candidate)])
