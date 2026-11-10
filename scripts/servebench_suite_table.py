#!/usr/bin/env python3
"""The lab table for a suite job: servebench comparisons between its arms' suite runs.

usage: scripts/servebench_suite_table.py [--suites-dir DIR] [--reference-repeat LABEL | --no-reference-repeat]
                                         RUN LABEL [LABEL...]

Arms run candidate first (heat soak), then the reference, then the candidate again. With
two or more arms, the second arm is the reference: it is compared against every other arm,
and the first against the last as the run's own repeat. A LABEL with a "/" is a recorded
suite run, repository-relative (a release's manifests/suites/<name>): a released build's
numbers never change, so the reference is read from its record instead of measured again.

The verdict block judges the paired decode row (servebench's per-request pairs) and the
engine grid's geometric mean against recorded A/A floors (``floors_for``: by testbed and
suite where enough repeats were recorded, else the defaults), over both candidate boots;
the other headline rows keep the run's own repeat. When the reference has a repeat boot
(``reference_repeat``: an arm or record named ``<reference>-again``, or
``<reference>-again-<suite>`` for a companion's record, or ``--reference-repeat``), every
headline row is the mean over both reference boots and both candidate boots, and the boot
term shrinks to x0.707 of one pair's (servebench.suite.boot_term). Rows a step follow, one
line per arm and closed-loop step.
"""
import argparse
import json
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
# The default for a step no testbed entry below covers.
AA_FLOOR_PCT = {"c1": 0.35, "c4": 0.27, "c8": 0.39, "c16": 0.43}

# The default floor of an engine grid step's decode geometric mean: 1.96 x the root mean
# square of 31 same-build pairs ("X -> X-again", the same image on two boots of one session)
# of candidates 50-58, 2026-11-09 to 11-10, on tp4-ring, tp4-switched and tp3-switched,
# screen and standalone grid suites (rms 0.215%). Before it, a grid verdict took
# interval + |repeat|, one draw of the boot noise (margins 0.13-0.74%).
GRID_FLOOR_PCT = 0.43

# Floors by (testbed, suite name), from the same-build pairs of candidates 50-58
# (2026-11-09 to 11-10; the boot-to-boot variance review). Each floor is 1.96 x the root
# mean square of the pairs' repeat changes, rounded up to 0.01%. Conservative rule: a
# testbed's floor replaces the default only with at least 4 pairs, and is lower than the
# default only with at least 6; otherwise the default stays and "kept" says what the pairs
# showed. The run's own repeat still widens any floor it exceeds.
FLOORS = {
    ("tp4-ring", "screen"): {
        "floors": {"grid": 0.37, "c16": 0.39},
        "pairs": {"grid": 6, "c16": 6},
        "runs": "candidate-50-screen-tp4 (candidate-50-no1009, spark-ds41f-1107), candidate-51-screen-tp4, "
                "candidate-54-screen-tp4, candidate-55-screen-tp4-no978, candidate-58-screen-tp4",
    },
    ("tp4-switched", "screen"): {
        "floors": {"grid": 0.33, "c16": 0.34},
        "pairs": {"grid": 7, "c16": 7},
        "runs": "candidate-50-screen-tp4-switched (candidate-50-no1009, spark-ds41f-1107), "
                "candidate-50-screen-tp4-switched-1009r1, candidate-52-, candidate-54-, candidate-56-, "
                "candidate-58-screen-tp4-switched",
    },
    ("tp4-switched", "grid"): {
        "floors": {"grid": 0.41},
        "pairs": {"grid": 6},
        "runs": "candidate-56-attribute-grid-tp4-switched (candidate-54, candidate-56-no957p1-no956), "
                "candidate-56-attribute-grid-957p1r1-tp4-switched (both arms), "
                "candidate-56-attribute-956r4-tp4-switched (both grid arms)",
    },
    ("tp3-switched", "grid-tp3"): {
        "floors": {"grid": 0.60},
        "pairs": {"grid": 7},
        "runs": "candidate-54-attribute-tp3-grid-tp3-switched, candidate-54-tp3-prefetch-tp3-switched (2), "
                "candidate-54-tp3-prefetch-sweep-tp3-switched (2), candidate-58-screen-tp3-switched grid arms (2)",
    },
    ("tp3-switched", "sampled"): {
        "floors": {"c16": 0.96},
        "pairs": {"c16": 4},
        "kept": {"c8": "3 pairs, 0.85%", "c1": "2 pairs, 1.22%"},
        "runs": "candidate-53-, candidate-54-, candidate-57-, candidate-58-screen-tp3-switched-sampled",
    },
    ("tp3-switched", "screen-tp3"): {
        "floors": {},
        "pairs": {},
        "kept": {"grid": "5 pairs, 0.32%; a lower floor needs 6", "c8": "3 pairs, 0.52%"},
        "runs": "candidate-50-screen-tp3-switched (2), candidate-50-screen-tp3-switched-1009r1, "
                "candidate-53-screen-tp3-switched, candidate-57-screen-tp3-switched",
    },
    ("tp4-ring", "single-stream"): {
        "floors": {},
        "pairs": {},
        "kept": {"c1": "4 pairs, 0.25%; a lower floor needs 6"},
        "runs": "candidate-51-single-stream-tp4, candidate-55-single-stream-tp4-no978, "
                "candidate-58-single-stream-tp4, candidate-58-attribute-c1-tp4",
    },
    ("tp3-switched", "single-stream"): {
        "floors": {},
        "pairs": {},
        "kept": {"c1": "4 pairs, 0.32%; a lower floor needs 6"},
        "runs": "candidate-53-, candidate-54-, candidate-57-, candidate-58-screen-tp3-switched-single-stream",
    },
}
FLOORS_DATE = "2026-11-09 to 2026-11-10"


def testbed_of(index: dict) -> str | None:
    """A suite run's testbed shape from its recorded identity: tp<N>-switched on a switched
    transport, tp<N>-ring on a ring, tp<N>-triangle on direct links; None when unrecorded."""
    identity = (index.get("metadata") or {}).get("identity") or {}
    nodes = (identity.get("topology") or {}).get("nodes") or []
    transport = str(identity.get("transport") or "")
    if not nodes or not transport:
        return None
    kind = ("switched" if "switched" in transport else "ring" if "ring" in transport
            else "triangle" if ("direct" in transport or "triangle" in transport) else None)
    return f"tp{len(nodes)}-{kind}" if kind else None


def floors_for(index: dict) -> tuple[dict, str]:
    """The recorded floors for a candidate's suite run, by step id, and a line naming them:
    the testbed's and suite's own where recorded, else AA_FLOOR_PCT for closed-loop steps and
    GRID_FLOOR_PCT for engine grid steps."""
    definition = index.get("suite") or {}
    floors = {}
    for step in definition.get("steps", []):
        if step.get("command") == "engine":
            floors[step["id"]] = GRID_FLOOR_PCT
        elif step["id"] in AA_FLOOR_PCT:
            floors[step["id"]] = AA_FLOOR_PCT[step["id"]]
    testbed, name = testbed_of(index), definition.get("name")
    entry = FLOORS.get((testbed, name)) or {"floors": {}, "pairs": {}}
    own = {step: value for step, value in entry["floors"].items() if step in floors}
    kept = {step: why for step, why in entry.get("kept", {}).items() if step in floors}
    floors.update(own)
    parts = [f"{step} {value:.2f}% ({entry['pairs'][step]} pairs)" for step, value in own.items()]
    parts += [f"{step} default {floors[step]:.2f}% kept ({why})" for step, why in kept.items()]
    parts += [f"{step} default {value:.2f}%" for step, value in floors.items() if step not in own and step not in kept]
    where = f"floors ({testbed or 'testbed not recorded'}, {name}): "
    if not parts:
        return floors, where + "no step of this suite takes a recorded floor"
    source = (f"; same-build pairs of {FLOORS_DATE}, servebench_suite_table.FLOORS" if own or kept
              else "; the recorded defaults")
    return floors, where + "; ".join(parts) + source


def pairs(labels: list[str], reference_repeat: str | None = None) -> list[tuple[str, str]]:
    if len(labels) < 2:
        return []
    reference = labels[1]
    out = [(reference, label) for label in labels if label != reference]
    if len(labels) > 2:
        out.append((labels[0], labels[-1]))
    if reference_repeat and reference_repeat not in labels:
        out.append((reference, reference_repeat))
    return out


def is_recorded(label: str) -> bool:
    return "/" in label


def directory(run: str, label: str, suites_dir: Path | None = None) -> str:
    if is_recorded(label):
        return str(ROOT / label)
    return str((suites_dir or ROOT / "results/private/suites") / run / label)


def name(label: str) -> str:
    """A recorded run by its record's name (the release and shape), an arm by its label."""
    return f"{Path(label).name} (recorded)" if is_recorded(label) else label


def suite_index(path: str) -> dict | None:
    try:
        return json.loads((Path(path) / "suite.json").read_text())
    except (OSError, ValueError):
        return None


def repeat_names(label: str, suite_name: str | None) -> list[str]:
    """The names a reference's repeat boot takes: ``<label>-again``; for a companion's record
    ``<reference>-<suite>``, ``<reference>-again-<suite>``; and an arm label with a suffix
    (``x-stall``) its ``x-again-stall``."""
    base = label.rstrip("/")
    out = [f"{base}-again"]
    if suite_name and base.endswith(f"-{suite_name}"):
        out.append(f"{base[:-len(suite_name) - 1]}-again-{suite_name}")
    head, _, tail = base.rpartition("-")
    if head and tail and "/" not in tail:
        out.append(f"{head}-again-{tail}")
    return list(dict.fromkeys(out))


def reference_repeat(run: str, labels: list[str], suites_dir: Path | None = None,
                     explicit: str | None = None) -> tuple[str | None, str | None]:
    """The reference's repeat boot, and why one named is not used. An arm of the run named as
    the reference's repeat, or for a recorded reference a record beside it; it must be a run
    of the same suite definition. ``explicit`` names it instead."""
    if len(labels) < 3:
        return None, None
    reference = labels[1]
    index = suite_index(directory(run, reference, suites_dir))
    if index is None:
        return None, None
    names = [explicit] if explicit else repeat_names(reference, index["suite"].get("name"))
    why = None
    for candidate in names:
        if candidate in (labels[0], labels[-1], reference):
            continue
        if not is_recorded(reference) and not explicit and candidate not in labels:
            continue
        other = suite_index(directory(run, candidate, suites_dir))
        if other is None:
            if explicit:
                why = f"reference repeat {candidate} is not a suite run (no suite.json)"
            continue
        if other.get("suite_sha256") != index.get("suite_sha256"):
            why = why or (f"reference repeat {candidate} is a run of suite {other['suite'].get('name')} "
                          f"{str(other.get('suite_sha256'))[:12]}, not the reference's; not averaged")
            continue
        return candidate, None
    return None, why


def verdicts(run: str, labels: list[str], suites_dir: Path | None = None, repeat_of_reference: str | None = None,
             average_reference: bool = True) -> str:
    """The verdict block: reference -> candidate, judged against the candidate's own repeat
    (and the reference's, when it has one) and the recorded floors."""
    judge = lambda b, c: compare_results(b, c, 3.0)
    at = lambda label: directory(run, label, suites_dir)
    candidate, reference, again = labels[0], labels[1], labels[-1]
    reference_to_candidate = suite.compare(read_json, at(reference), at(candidate), judge)
    repeat = suite.compare(read_json, at(candidate), at(again), judge)
    reference_to_repeat = suite.compare(read_json, at(reference), at(again), judge)
    floors, note = floors_for(suite_index(at(candidate)) or {})
    extra, warning = {}, None
    second = None
    if average_reference:
        second, warning = reference_repeat(run, labels, suites_dir, repeat_of_reference)
    if second:
        extra = {"reference_repeat": suite.compare(read_json, at(reference), at(second), judge),
                 "reference_again_to_candidate": suite.compare(read_json, at(second), at(candidate), judge),
                 "reference_again_to_repeat": suite.compare(read_json, at(second), at(again), judge)}
    rows = suite.with_floor(reference_to_candidate, repeat, floors=floors, reference_to_repeat=reference_to_repeat,
                            **extra)
    text = suite.format_floor(rows, name(reference), candidate, again,
                              reference_repeat=name(second) if second else None, floors_note=note)
    if warning:
        text = text.replace("\n", f"\n{warning}\n", 1)
    return text


def parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run")
    parser.add_argument("labels", nargs="*")
    parser.add_argument("--suites-dir", type=Path,
                        help="where RUN/LABEL suite runs live (default results/private/suites); for re-reading receipts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reference-repeat", help="the reference's repeat boot: an arm label or a recorded run's path")
    group.add_argument("--no-reference-repeat", action="store_true",
                       help="judge against one reference boot even when a repeat exists")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse(sys.argv[1:])
    run, labels = args.run, args.labels
    second = None
    if len(labels) > 2:
        # The verdict first: reference -> candidate, judged against the candidate's own repeat.
        print(verdicts(run, labels, args.suites_dir, args.reference_repeat, not args.no_reference_repeat))
        if not args.no_reference_repeat:
            second = reference_repeat(run, labels, args.suites_dir, args.reference_repeat)[0]
    if labels:
        shown = labels + ([second] if second and second not in labels else [])
        print("\n" + suite.format_accounting({name(label): suite.step_accounting(read_json, directory(run, label, args.suites_dir))
                                              for label in shown}))
    for base, candidate in pairs(labels, second):
        print(f"\n######## {name(base)} -> {name(candidate)}", flush=True)
        main(["compare", directory(run, base, args.suites_dir), directory(run, candidate, args.suites_dir)])
