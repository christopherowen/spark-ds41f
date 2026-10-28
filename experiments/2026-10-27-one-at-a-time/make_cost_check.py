#!/usr/bin/env python3
"""Are the leave-one-out costs of 0506, 0526 and 0528 real? Boots, not replays, as samples.

usage: make_cost_check.py VLLM_WORKTREE

Each leave-one-out run booted candidate 9, the arm, then candidate 9 again, once each. At
131K prefill, every arm on candidate 9 came out faster than its run's candidate 9 boots
(0502 +0.24%, 0505 +0.25%, 0528 +0.35%, 0506 +0.44%, 0526 +0.47%), while 18 candidate 9
boots spread 0.22% (sd). And the engine grid that charged 0528 1.2% never runs the
rejection sampler (ModelRunner._dummy_run: target and drafter only).

This writes a null arm, "null-overlay": candidate 9 with the files those three arms
replace mounted from candidate 9 itself (VLLM_WORKTREE at e2b2b0b59), so it differs from
candidate 9 only in being an overlay arm. cost-check.json boots candidate 9 three times
and each other arm twice, interleaved so no arm holds one position, on cost-suite.json
(the lean suite's heat-soak, grid and prefill). cost_check.py compares the sets.
"""

import json
import subprocess
import sys
from pathlib import Path

import make_arms

HERE = Path(__file__).resolve().parent
LABEL = "null-overlay"
ARMS = ["no-0526", "no-0506", "no-0528"]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    files = sorted({str(path.relative_to(HERE / "overlay" / arm))
                    for arm in ARMS for path in (HERE / "overlay" / arm).rglob("*") if path.is_file()})
    for path in files:
        target = HERE / "overlay" / LABEL / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.run(["git", "-C", repo, "show", f"{make_arms.HEAD}:{path}"],
                                          check=True, capture_output=True).stdout)
    make_arms.arms(LABEL, files, f"candidate 9's own {', '.join(ARMS)} files, mounted as those arms "
                                 "mount theirs: the overlay arm control")
    (HERE / f"{LABEL}.json").unlink()
    c9 = {"config": f"{make_arms.CANDIDATE}/rc-tp4-dev.json"}

    def arm(label: str, boot: str) -> dict:
        if label == "candidate-9":
            return {**c9, "label": f"candidate-9-{boot}"}
        return {"config": f"{label}-tp4-dev.json", "label": f"{label}-{boot}"}

    order = [arm("candidate-9", "a"), arm(LABEL, "a"), *[arm(a, "a") for a in ARMS],
             arm("candidate-9", "b"), *[arm(a, "b") for a in reversed(ARMS)], arm(LABEL, "b"),
             arm("candidate-9", "c")]
    make_arms.write_json(HERE / "cost-check.json", {
        "experiment": make_arms.FOLDER,
        "run": "one-at-a-time-cost-check",
        "jobs": [{"kind": "sync"},
                 {"kind": "measure", "suite": f"{make_arms.FOLDER}/cost-suite.json", "arms": order}],
    })
    lean = json.loads((HERE / "lean-suite.json").read_text(encoding="utf-8"))
    make_arms.write_json(HERE / "cost-suite.json", {
        "name": "program-cost",
        "version": 1,
        "description": "The lean suite's model step (the heat-soak), engine grid and cold real-text "
                       "prefill, without the closed loops: one boot's sample for cost_check.py.",
        "steps": [step for step in lean["steps"] if step["id"] in ("model", "grid", "prefill")],
    })
    print(f"{LABEL}: {len(files)} files; cost-check.json: {len(order)} boots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
