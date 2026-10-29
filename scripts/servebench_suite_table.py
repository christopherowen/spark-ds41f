#!/usr/bin/env python3
"""The lab table for a suite job: servebench comparisons between its arms' suite runs.

usage: scripts/servebench_suite_table.py RUN LABEL [LABEL...]

Arms run candidate first (heat soak), then the reference, then the candidate again. With
two or more arms, the second arm is the reference: it is compared against every other arm,
and the first against the last as the run's own repeat. A LABEL with a "/" is a recorded
suite run, repository-relative (a release's manifests/suites/<name>): a released build's
numbers never change, so the reference is read from its record instead of measured again.

The verdict block judges the paired decode row (servebench's per-request pairs) against
AA_FLOOR_PCT, over both candidate boots; the window rows keep the run's own repeat. Rows a
step follow, one line per arm and closed-loop step.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from servebench import suite  # noqa: E402
from servebench.__main__ import compare_results, main, read_json  # noqa: E402

# The 95% bound of one boot against another, in % (1.96 x the sd of A/A changes), by
# suite step: the "X -> X-again" pairs of 2026-10-15 to 10-25 (release, single-stream and
# sampled suites, TP3 and TP4), output tok/s over the whole window. c1: 31 pairs, sd 0.18%;
# c4: 8, 0.14%; c8: 26, 0.20%; c16: 31, 0.22%. experiments/2026-10-26-paired-verdict.
AA_FLOOR_PCT = {"c1": 0.35, "c4": 0.27, "c8": 0.39, "c16": 0.43}


def pairs(labels: list[str]) -> list[tuple[str, str]]:
    if len(labels) < 2:
        return []
    reference = labels[1]
    out = [(reference, label) for label in labels if label != reference]
    if len(labels) > 2:
        out.append((labels[0], labels[-1]))
    return out


def is_recorded(label: str) -> bool:
    return "/" in label


def directory(run: str, label: str) -> str:
    if is_recorded(label):
        return str(ROOT / label)
    return str(ROOT / "results/private/suites" / run / label)


def name(label: str) -> str:
    """A recorded run by its record's name (the release and shape), an arm by its label."""
    return f"{Path(label).name} (recorded)" if is_recorded(label) else label


if __name__ == "__main__":
    run, labels = sys.argv[1], sys.argv[2:]
    if len(labels) > 2:
        # The verdict first: reference -> candidate, judged against the candidate's own repeat.
        judge = lambda b, c: compare_results(b, c, 3.0)
        reference_to_candidate = suite.compare(read_json, directory(run, labels[1]), directory(run, labels[0]), judge)
        repeat = suite.compare(read_json, directory(run, labels[0]), directory(run, labels[-1]), judge)
        reference_to_repeat = suite.compare(read_json, directory(run, labels[1]), directory(run, labels[-1]), judge)
        print(suite.format_floor(suite.with_floor(reference_to_candidate, repeat, floors=AA_FLOOR_PCT,
                                                  reference_to_repeat=reference_to_repeat),
                                 name(labels[1]), labels[0], labels[-1]))
    if labels:
        print("\n" + suite.format_accounting({name(label): suite.step_accounting(read_json, directory(run, label))
                                              for label in labels}))
    for base, candidate in pairs(labels):
        print(f"\n######## {name(base)} -> {name(candidate)}", flush=True)
        main(["compare", directory(run, base), directory(run, candidate)])
