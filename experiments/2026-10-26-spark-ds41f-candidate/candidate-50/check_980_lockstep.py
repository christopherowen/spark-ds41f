#!/usr/bin/env python3
"""#980 check: the PR's 1,000-step TP lockstep screen shows 0 rank mismatches on every node
of the served shape. Reads the outputs of the lab's fabric jobs that ran the screen; a lab
extra, so it takes the served URL first and ignores it.

usage: check_980_lockstep.py URL RUN_DIR --labels LABEL[,LABEL...] [--steps 1000] [--seed 1234]

The screen is screen_zero_broadcast_sync_rowsfix_valid_seeded.py beside this file:
candidate-48's #980 round 1 harness (the PR's screen_zero_broadcast_sync.py at 11b169b0dd07
with the TP2 row fix, the written-rows comparison and --sampling-seed), byte for byte as it
ran on sp2 (SHA-256 99a61824f3c1). A fabric job runs it on every node at once in the arm's
image and environment, rank = the node's rank, with serving stopped; each node's output is
RUN_DIR/fabric-<label>-<node>.txt, and rank 0 prints the JSON_RECORD lines.

For every label, required:
- every node's output is there and none holds a Python traceback;
- rank 0 reports the four determinism modes over --steps steps with --seed, at the tensor
  parallel size of the node count;
- each candidate mode (greedy, sampled unseeded, DFlash estimator) makes no broadcast, has the
  same seeds on every rank and 0 mismatching steps in every compared quantity;
- the control (unseeded global NumPy RNG, the code before the fix) does mismatch, so the screen
  can see a divergence;
- the latency part ran to its end (three estimator records).

Prints one JSON line per label, then a summary line ("passed"); exits 1 when a requirement
fails, 0 otherwise. Stdlib only.
"""
import argparse
import json
import sys
from pathlib import Path

MODES = ("control_unseeded_global_rng", "candidate_greedy", "candidate_sampled_unseeded",
         "candidate_dflash_estimator")
MISMATCHES = ("mismatch_draft_steps", "mismatch_conf_steps", "mismatch_budget_steps", "mismatch_capacities_steps",
              "mismatch_dead_cut_steps", "mismatch_sampled_steps", "mismatch_estimator_steps")


def check(run_dir: Path, label: str, steps: int, seed: int) -> dict:
    result = {"label": label, "passed": False, "problems": []}
    outputs = sorted(run_dir.glob(f"fabric-{label}-*.txt"))
    result["nodes"] = [path.name[len(f"fabric-{label}-"):-4] for path in outputs]
    if not outputs:
        result["problems"].append(f"no fabric-{label}-*.txt in {run_dir}")
        return result
    records = []
    for path in outputs:
        text = path.read_text(errors="replace")
        if "Traceback (most recent call last)" in text:
            result["problems"].append(f"{path.name} holds a traceback")
        records += [json.loads(line.split("JSON_RECORD:", 1)[1]) for line in text.splitlines()
                    if line.startswith("JSON_RECORD:")]
    modes = {r["mode"]: r for r in records if r.get("kind") == "determinism_1000_steps"}
    result["tp_size"] = sorted({r.get("tp_size") for r in records})
    if result["tp_size"] != [len(outputs)]:
        result["problems"].append(f"tp_size {result['tp_size']} for {len(outputs)} nodes")
    for mode in MODES:
        record = modes.get(mode)
        if record is None:
            result["problems"].append(f"{mode} missing")
            continue
        counts = {key: record.get(key) for key in MISMATCHES}
        result[mode] = {**counts, "broadcast_calls": record.get("broadcast_calls"),
                        "seeds_equal_across_ranks": record.get("seeds_equal_across_ranks")}
        if record.get("steps") != steps or record.get("sampling_seed") != seed:
            result["problems"].append(f"{mode}: steps {record.get('steps')}, seed {record.get('sampling_seed')}")
        if mode.startswith("candidate_"):
            if record.get("broadcast_calls") != 0:
                result["problems"].append(f"{mode}: {record.get('broadcast_calls')} broadcasts")
            if record.get("seeds_equal_across_ranks") is not True:
                result["problems"].append(f"{mode}: seeds differ across ranks")
            result["problems"] += [f"{mode}: {key} {value}" for key, value in counts.items() if value != 0]
        elif not record.get("mismatch_sampled_steps"):
            result["problems"].append("control shows no mismatch: the screen cannot see a divergence")
    if sum(r.get("kind") == "estimator_predict_latency" for r in records) < 3:
        result["problems"].append("the screen did not run to its end")
    result["passed"] = not result["problems"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the served endpoint (unused: the screen ran with serving stopped)")
    parser.add_argument("run_dir")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    results = [check(Path(args.run_dir), label, args.steps, args.seed) for label in args.labels.split(",")]
    for result in results:
        print(json.dumps(result), flush=True)
    passed = all(r["passed"] for r in results)
    print(json.dumps({"summary": "#980 TP lockstep screen", "passed": passed,
                      "labels": {r["label"]: r["passed"] for r in results},
                      "failing": [f"{r['label']}: {p}" for r in results for p in r["problems"]]}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
