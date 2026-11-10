#!/usr/bin/env python3
"""The first step where two boots' decisions differ, and the input that differs first.

usage: diff_step_traces.py BOOT_A_DIR BOOT_B_DIR [--rank 0] [--context 2] [--json OUT]

Each directory holds one boot's collected traces (collect_traces.py: <node>/steps-*.jsonl
from every rank, and the head's sched-*.jsonl). Request IDs differ between boots, so
requests are matched by the order they first appear (the c1 workload sends the same
prompts in the same order), and each request's steps by their order within it.

For every aligned step the fields are compared in causal order, so the first differing
field is the earliest decision input that differs:
  1. scheduler   the schedule() record: scheduled tokens, draft placeholders, skip_drafts,
                 num_spec_tokens_to_schedule, the running request's copy allocation
  2. budget-in   the worker's draft-budget inputs: slot, scheduled drafts, the stale
                 confidence snapshot (bits), tau, the idle counter, computed tokens
  3. budget-out  the budget's result: verification rows, admitted drafts per request
  4. verify      sampled tokens, sampled and rejected counts, verified drafts, dead-row cut
  5. estimator   refits, steps since refit, slope, intercepts, accumulators (bits)
  6. drafter     raw and calibrated confidences, planned confidences, next drafts,
                 copy-drafter choice, estimates, allocation and state (bits)
It also checks, within each boot, that every rank recorded the same decisions (rank
disagreement would itself be a defect). Prints the verdict and both boots' values of the
first differing field with the steps around it.
"""
import argparse
import glob
import json
import os
import sys

ORDER = [
    ("scheduler", ["sched.num_scheduled_tokens", "sched.spec_tokens", "sched.skip_drafts",
                   "sched.num_spec_tokens_to_schedule", "sched.running", "sched.new", "sched.finished"]),
    ("budget-in", ["slots", "budget.slots", "budget.scheduled_tokens", "budget.scheduled_drafts",
                   "budget.stale_idx", "budget.stale_confidences", "budget.tau", "budget.idle_before",
                   "budget.num_computed_np", "budget.prefill_len_np"]),
    ("budget-out", ["budget.num_tokens", "budget.batch_budget", "budget.idle_after", "num_tokens",
                    "num_tokens_after_padding", "num_scheduled_tokens", "num_draft_tokens_per_req",
                    "num_spec_tokens_to_schedule"]),
    ("verify", ["device.sampled_token_ids", "device.num_sampled", "device.num_rejected",
                "device.num_verified_draft_tokens", "device.dead_row_cut"]),
    ("estimator", ["est_refits", "est_steps_since_refit", "tau", "idle", "device.est_slope",
                   "device.est_intercepts", "device.est_counts", "device.est_info", "device.est_grad"]),
    ("drafter", ["device.raw_confidences", "device.confidences", "device.planned_confidences",
                 "device.draft_tokens_next", "device.copy_chosen", "device.copy_estimates",
                 "device.copy_alloc", "device.copy_int_state", "device.copy_float_state"]),
]


def load_jsonl(paths):
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def boot(directory, rank):
    steps = sorted(glob.glob(os.path.join(directory, "**", f"steps-*-tp{rank}-*.jsonl"), recursive=True))
    sched = sorted(glob.glob(os.path.join(directory, "**", "sched-*.jsonl"), recursive=True))
    if not steps:
        raise SystemExit(f"{directory}: no steps-*-tp{rank}-*.jsonl")
    return load_jsonl(steps), load_jsonl(sched)


def canonical(value, names):
    """Replace request IDs by their first-seen index so two boots compare."""
    if isinstance(value, dict):
        if "hex" in value and "dtype" in value:
            return {"dtype": value["dtype"], "shape": value["shape"], "hex": value["hex"]}
        return {names.get(k, k): canonical(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v, names) for v in value]
    if isinstance(value, str):
        return names.get(value, value)
    return value


def field(record, dotted):
    value = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def align(steps, sched):
    """(request index, step within request) -> worker record and its schedule record."""
    names = {}
    for record in steps:
        for r in record.get("req_ids", []):
            names.setdefault(r, f"req{len(names)}")
    schedules = [s for s in sched if s.get("kind") == "schedule" and s.get("num_scheduled_tokens")]
    by_set = {}
    for s in schedules:
        by_set.setdefault(tuple(sorted(s["num_scheduled_tokens"])), []).append(s)
    keyed, seen, used = {}, {}, {}
    for record in steps:
        key_reqs = tuple(sorted(record.get("req_ids", [])))
        reqs = tuple(names[r] for r in record.get("req_ids", []))
        seen[reqs] = seen.get(reqs, 0) + 1
        sched_list = by_set.get(key_reqs, [])
        i = used.get(key_reqs, 0)
        used[key_reqs] = i + 1
        record = dict(record)
        if i < len(sched_list):
            record["sched"] = sched_list[i]
        keyed[(reqs, seen[reqs])] = record
    return keyed, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--json")
    args = parser.parse_args()
    result = {"a": args.a, "b": args.b, "rank": args.rank}

    # Within each boot: every rank must have recorded the same decisions.
    for label, directory in (("a", args.a), ("b", args.b)):
        per_rank = {}
        for rank in range(16):
            files = sorted(glob.glob(os.path.join(directory, "**", f"steps-*-tp{rank}-*.jsonl"), recursive=True))
            if files:
                per_rank[rank] = load_jsonl(files)
        disagreements = []
        if per_rank:
            ref_rank = min(per_rank)
            fields = ["num_tokens", "num_scheduled_tokens", "device.num_sampled", "device.dead_row_cut",
                      "device.draft_tokens_next", "device.confidences", "budget.num_tokens"]
            for rank, recs in per_rank.items():
                for i, (x, y) in enumerate(zip(per_rank[ref_rank], recs)):
                    for f in fields:
                        if json.dumps(field(x, f)) != json.dumps(field(y, f)):
                            disagreements.append({"step": i, "rank": rank, "field": f})
                            break
                    if len(disagreements) >= 5:
                        break
        result[f"{label}_ranks"] = sorted(per_rank)
        result[f"{label}_rank_disagreements"] = disagreements

    steps_a, sched_a = boot(args.a, args.rank)
    steps_b, sched_b = boot(args.b, args.rank)
    keyed_a, names_a = align(steps_a, sched_a)
    keyed_b, names_b = align(steps_b, sched_b)
    order = list(keyed_a)
    result["steps"] = {"a": len(keyed_a), "b": len(keyed_b), "requests": {"a": len(names_a), "b": len(names_b)}}
    first = None
    for position, key in enumerate(order):
        if key not in keyed_b:
            first = {"index": position, "key": list(key), "group": "schedule shape",
                     "field": "this request step exists in A only"}
            break
        ra, rb = keyed_a[key], keyed_b[key]
        for group, fields in ORDER:
            for f in fields:
                va = canonical(field(ra, f), names_a)
                vb = canonical(field(rb, f), names_b)
                if json.dumps(va, sort_keys=True) != json.dumps(vb, sort_keys=True):
                    first = {"index": position, "key": list(key), "group": group, "field": f,
                             "a": field(ra, f), "b": field(rb, f)}
                    break
            if first:
                break
        if first:
            break
    result["first_difference"] = first
    if first:
        lo = max(0, first["index"] - args.context)
        window = []
        for key in order[lo:first["index"] + 1]:
            for label, keyed in (("a", keyed_a), ("b", keyed_b)):
                r = keyed.get(key)
                if r is None:
                    continue
                window.append({"boot": label, "key": list(key), "num_tokens": r.get("num_tokens"),
                               "num_draft_tokens_per_req": r.get("num_draft_tokens_per_req"),
                               "budget": {k: r.get("budget", {}).get(k) for k in ("scheduled_drafts", "tau",
                                          "idle_before", "num_tokens", "batch_budget")} if r.get("budget") else None,
                               "num_sampled": field(r, "device.num_sampled.values"),
                               "est_refits": r.get("est_refits"), "est_steps_since_refit": r.get("est_steps_since_refit"),
                               "slope": field(r, "device.est_slope.values"),
                               "draft_tokens_next": field(r, "device.draft_tokens_next.values"),
                               "sched": r.get("sched")})
        result["window"] = window
        verdicts = {
            "scheduler": "the engine core scheduled the step differently (an engine-side input: arrival "
                         "timing, the copy allocation it read, or the finished/new request bookkeeping)",
            "budget-in": "the worker's budget read different inputs although the schedule matched",
            "budget-out": "the budget decided differently from equal inputs (host arithmetic)",
            "verify": "the target's verification differed on equal rows and drafts (target numerics)",
            "estimator": "the estimator's state differs with equal observations so far",
            "drafter": "the drafter produced other confidences or drafts from equal inputs (drafter numerics "
                       "or state)",
        }
        result["verdict"] = verdicts.get(first["group"], first["field"])
    else:
        result["verdict"] = "no difference: every aligned step's decisions and inputs are equal"
    text = json.dumps(result, indent=1, default=str)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text[:20000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
