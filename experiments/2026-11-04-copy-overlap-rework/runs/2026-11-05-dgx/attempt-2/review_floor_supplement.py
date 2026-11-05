#!/usr/bin/env python3
"""Apply the unchanged repeat-floor method to both C41 boots, read-only inputs."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from servebench import suite  # noqa: E402
from scripts.servebench_suite_table import AA_FLOOR_PCT  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def supplement(comparison):
    pin = "0da35cc9668ff491065446b2a842fe996a73e80f"
    method_inputs = {}
    for relative in ("servebench/suite.py", "scripts/servebench_suite_table.py"):
        current = (ROOT / relative).read_bytes()
        frozen = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{pin}:{relative}"])
        if current != frozen:
            raise ValueError(f"comparison method differs from tested execution: {relative}")
        method_inputs[relative] = sha(current)
    data = comparison.read_bytes()
    original = json.loads(data)["candidate-40"]["comparisons"]["release-tp3"]
    views = []
    for ref in (1, 2):
        for candidate in (1, 2):
            pair = original["pairs"][f"reference_boot{ref}_to_candidate_boot{candidate}"]
            other = original["pairs"][f"reference_boot{ref}_to_candidate_boot{3-candidate}"]
            for repeat in ("candidate_repeat", "saved_reference_repeat"):
                rows = suite.with_floor(pair, original[repeat], floors=AA_FLOOR_PCT,
                                        reference_to_repeat=other)
                rows = [r for r in rows if r["metric"] != suite.PAIRED]
                views.append({"reference_boot": ref, "candidate_boot": candidate,
                              "repeat_floor": repeat, "rows": rows})
    counts = {}
    for label, step, metric in (("grid", "grid", "decode geometric mean (15)"),
                                ("cold_524K", "prefill", "prefill 524,288 tok/s")):
        verdicts = [row["verdict"] for view in views for row in view["rows"]
                    if row["step"] == step and row["metric"] == metric]
        counts[label] = {verdict: verdicts.count(verdict) for verdict in sorted(set(verdicts))}
    return {"purpose": "Non-paired boot2 supplement; existing comparisons unchanged",
            "execution_commit": pin, "comparison": str(comparison.relative_to(ROOT)),
            "comparison_sha256": sha(data), "method_files": method_inputs,
            "with_floor_source_sha256": sha(inspect.getsource(suite.with_floor).encode()),
            "AA_FLOOR_PCT": AA_FLOOR_PCT, "new_tolerance": False,
            "counts_of_eight_pair_and_floor_views": counts, "views": views,
            "limits": ["A floor view is not an independent experiment.",
                       "Non-paired margins remain CI95 plus absolute repeat change.",
                       "No new measurement or original-receipt modification."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path(__file__).with_name("recipe-comparison.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = supplement(args.comparison.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["counts_of_eight_pair_and_floor_views"]))


if __name__ == "__main__":
    main()
