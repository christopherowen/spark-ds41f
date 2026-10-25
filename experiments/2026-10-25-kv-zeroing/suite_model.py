#!/usr/bin/env python3
"""The poison arm's release-suite model behaviour against candidate 6's own release run.

usage: suite_model.py [RUN LABEL [REFERENCE_RUN REFERENCE_LABEL]]

Defaults: kv-zeroing-tp4-suite / poison-tp4 against candidate-6-release-tp4 /
candidate-6 (the triangle: kv-zeroing-tp3-suite poison-tp3 candidate-6-release-tp3
candidate-6).
Reads each run's results/private/suites/<run>/<label>/model.json.gz: 256 prompts at 16
streams (TP3: 8), greedy. The poison arm passes when every response is token-identical,
draft acceptance by position is equal (the drafter reads poisoned blocks in no step that
counts) and the teacher-forced scores are identical distributions; servebench's own
comparison prints the numbers. Exits 1 when a run is missing or a check fails.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from servebench import behaviour  # noqa: E402
from servebench.__main__ import read_json  # noqa: E402


def model(run: str, label: str) -> dict | None:
    for name in ("model.json.gz", "model.json"):
        path = ROOT / "results/private/suites" / run / label / name
        if path.exists():
            return read_json(str(path))
    return None


def main() -> int:
    args = sys.argv[1:] or ["kv-zeroing-tp4-suite", "poison-tp4"]
    run, label = args[0], args[1]
    shape = "tp3" if label.endswith("tp3") else "tp4"
    reference = args[2:4] or [f"candidate-6-release-{shape}", "candidate-6"]
    base, candidate = model(*reference), model(run, label)
    if base is None or candidate is None:
        print(f"missing: {'/'.join(reference) if base is None else ''} "
              f"{run + '/' + label if candidate is None else ''}")
        return 1
    result = behaviour.compare(base, candidate)
    print(behaviour.format_comparison(result, "/".join(reference), f"{run}/{label}"))
    generation = result["generation"]
    equal_rates = all(row["base"] == row["candidate"] for row in generation["per_position"])
    scores = result.get("scores") or {}
    passed = (generation["compared"] > 0 and generation["identical"] == generation["compared"]
              and equal_rates and scores.get("identical_fraction", 1.0) == 1.0)
    print(f"verdict: {'PASS' if passed else 'FAIL'} (responses identical "
          f"{generation['identical']}/{generation['compared']}, acceptance by position "
          f"{'equal' if equal_rates else 'DIFFERS'})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
