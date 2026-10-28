#!/usr/bin/env python3
"""The paired verdict's rule on the suite runs we kept: A/A floors, false calls, real changes.

usage: validate.py [SUITES_DIR]   (default results/private/suites; dgx1 keeps the lab's)

The kept runs have no request tables (servebench keeps them from this change on), so the
paired row here is servebench's fallback for such runs: output tok/s window k against window
k (report.window_paired), judged by the same rule as the per-request row (suite.with_floor
with recorded floors). Prints, for every closed-loop step id:

1. the A/A floor: every run with arms X and X-again, the whole-window output tok/s change,
   its sd and 1.96 sd (one boot against another);
2. each A/A pair judged as a candidate (one boot) with a leave-one-out floor (the other pairs'
   1.96 sd): every pair should read "same";
3. what judging against the run's own repeat alone does once the interval is small: an A/A
   change against another A/A pair's |change|, over every ordered pair of pairs;
4. reference -> candidate (and candidate again) in every run that has all three arms: the
   suite table's window verdict beside the window-paired verdict over both boots.
"""
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from servebench import report, suite  # noqa: E402

STEPS = ("c1", "c4", "c8", "c16")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def closed(run: dict | None) -> bool:
    return bool(run) and run["workload"].get("mode") == "closed" and "batches" in run["summary"].get("output_tps", {})


def as_paired(result: dict) -> dict:
    """A comparison whose window-paired row stands in for the per-request row."""
    rows = [dict(row, metric=suite.PAIRED) if row["metric"] == "output tok/s, window-paired" else row
            for row in result["rows"]]
    return {"steps": {"step": ("servebench/1", {"rows": rows})}}


def main() -> None:
    top = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/private/suites"
    runs = sorted(p for p in top.iterdir() if p.is_dir())
    aa = {step: [] for step in STEPS}       # (run, arm, whole-window change %, window-paired row)
    triples = {step: [] for step in STEPS}  # (run, reference, candidate, three runs)
    for run in runs:
        arms = sorted(p.name for p in run.iterdir() if p.is_dir())
        for step in STEPS:
            data = {arm: load(run / arm / f"{step}.json") for arm in arms}
            data = {arm: value for arm, value in data.items() if closed(value)}
            for arm in data:
                again = f"{arm}-again"
                if again not in data:
                    continue
                x, y = data[arm], data[again]
                whole = 100 * (y["summary"]["output_tps"]["mean"] / x["summary"]["output_tps"]["mean"] - 1)
                row = next(r for r in report.compare(x, y)["rows"] if r["metric"] == "output tok/s, window-paired")
                aa[step].append((run.name, arm, whole, row))
                for reference in data:
                    if reference not in (arm, again) and not reference.endswith("-again"):
                        triples[step].append((run.name, reference, arm, (data[reference], x, y)))

    print("1. A/A floor: one boot against another, whole-window output tok/s")
    floors = {}
    for step in STEPS:
        changes = [c for _, _, c, _ in aa[step]]
        if len(changes) < 3:
            continue
        sd = statistics.stdev(changes)
        floors[step] = 1.96 * sd
        print(f"  {step:4s} {len(changes):3d} pairs  mean {statistics.fmean(changes):+.3f}%  sd {sd:.3f}%  "
              f"largest |A/A| {max(abs(c) for c in changes):.3f}%  1.96 sd {1.96 * sd:.3f}%")

    print("\n2. each A/A pair as a candidate (one boot), leave-one-out floor, window-paired interval")
    for step in floors:
        verdicts = {}
        for i, (run, arm, whole, row) in enumerate(aa[step]):
            others = [c for j, (_, _, c, _) in enumerate(aa[step]) if j != i]
            floor = {"step": 1.96 * statistics.stdev(others)}
            judged = suite.with_floor(as_paired({"rows": [row]}), as_paired({"rows": [dict(row, change_pct=0.0)]}),
                                      floors=floor)
            verdict = judged[0]["verdict"]
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if verdict != "same":
                print(f"  {step:4s} {run}/{arm}: {row['change_pct']:+.3f}% [{row['low_pct']:+.3f}, {row['high_pct']:+.3f}] "
                      f"margin {judged[0]['margin_pct']:.3f}%  {verdict}")
        print(f"  {step:4s} {', '.join(f'{v} {n}' for v, n in sorted(verdicts.items()))} of {len(aa[step])}")

    print("\n3. an A/A change judged against another pair's |change| as its only floor (interval ~0)")
    for step in floors:
        changes = [c for _, _, c, _ in aa[step]]
        calls = sum(abs(a) > abs(b) for i, a in enumerate(changes) for j, b in enumerate(changes) if i != j)
        total = len(changes) * (len(changes) - 1)
        print(f"  {step:4s} {calls} of {total} ordered pairs call a build different from itself ({100 * calls / total:.0f}%)")

    print("\n4. reference -> candidate: the window verdict (suite table) and the window-paired verdict, both boots")
    for step in floors:
        for run, reference, arm, (ref, cand, again) in triples[step]:
            first, second, repeat = report.compare(ref, cand), report.compare(ref, again), report.compare(cand, again)
            window = suite.with_floor({"steps": {"step": ("servebench/1", first)}},
                                      {"steps": {"step": ("servebench/1", repeat)}})
            window = next(r for r in window if r["metric"] == "output tok/s")
            judged = suite.with_floor(as_paired(first), as_paired(repeat), floors={"step": floors[step]},
                                      reference_to_repeat=as_paired(second))
            paired = next(r for r in judged if r["metric"] == suite.PAIRED)
            print(f"  {step:4s} {run}: {reference} -> {arm}\n"
                  f"       window  {window['change_pct']:+6.2f}% margin {window['margin_pct']:5.2f}%  {window['verdict']:6s}"
                  f"   window-paired {paired['change_pct']:+6.2f}% ±{paired['ci95_pct']:.2f}% floor {paired['floor_pct']:.2f}% "
                  f"margin {paired['margin_pct']:5.2f}%  {paired['verdict']}")


if __name__ == "__main__":
    main()
