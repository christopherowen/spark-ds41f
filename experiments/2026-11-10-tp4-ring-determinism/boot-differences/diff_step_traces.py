#!/usr/bin/env python3
"""The first step where two boots' decisions differ, and the input that differs first.

usage: diff_step_traces.py BOOT_A_DIR BOOT_B_DIR [--rank 0] [--context 2] [--rank-check] [--json OUT]

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

Only the valid part of a partly written buffer is compared: sampled_token_ids is
[requests, drafts + 1] and only each row's first num_sampled entries are written (the
tail is whatever the allocator left there, different on every boot). Differences found
only past the valid prefix are counted per field under "ignored_tail_differences" and
never reported as the first difference. Fields stream in compactly (bit strings, no
value lists), so a 120 MB trace needs a few hundred MB. --rank-check also verifies,
within each boot, that every rank recorded the same decisions (a disagreement between
ranks would itself be a defect).
"""
import argparse
import glob
import json
import os
import struct
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
# A partly written [rows, width] buffer -> the [rows] field holding each row's valid width.
VALID_WIDTH = {"device.sampled_token_ids": "device.num_sampled"}
HEX_PER_ELEMENT = {"int64": 16, "uint64": 16, "float64": 16, "int32": 8, "uint32": 8, "float32": 8,
                   "int16": 4, "uint16": 4, "float16": 4, "bfloat16": 4, "int8": 2, "uint8": 2, "bool": 2}
FORMAT = {"int64": "q", "int32": "i", "float32": "f", "int16": "h", "float64": "d"}
KEEP = {"step", "t", "req_ids", "slots", "num_scheduled_tokens", "num_draft_tokens_per_req", "num_tokens",
        "num_tokens_after_padding", "is_prefilling", "num_spec_tokens_to_schedule", "finished_req_ids",
        "est_refits", "est_steps_since_refit", "tau", "idle", "budget", "device"}


def compact(value):
    """Arrays keep only dtype, shape and bits (hex); containers recurse."""
    if isinstance(value, dict):
        if "hex" in value and "dtype" in value:
            return {"dtype": value["dtype"], "shape": value["shape"], "hex": value["hex"]}
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def decode(array):
    """An array's values from its bits (little-endian), flat."""
    if not isinstance(array, dict) or "hex" not in array:
        return None
    fmt = FORMAT.get(array["dtype"])
    if fmt is None:
        return None
    raw = bytes.fromhex(array["hex"])
    return list(struct.unpack(f"<{len(raw) // struct.calcsize(fmt)}{fmt}", raw))


def load_jsonl(paths, keep=None):
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if keep is not None:
                    record = {k: v for k, v in record.items() if k in keep}
                records.append(compact(record))
    return records


def boot(directory, rank):
    steps = sorted(glob.glob(os.path.join(directory, "**", f"steps-*-tp{rank}-*.jsonl"), recursive=True))
    sched = sorted(glob.glob(os.path.join(directory, "**", "sched-*.jsonl"), recursive=True))
    if not steps:
        raise SystemExit(f"{directory}: no steps-*-tp{rank}-*.jsonl")
    return load_jsonl(steps, KEEP), [s for s in load_jsonl(sched) if s.get("kind") == "schedule"]


def canonical(value, names):
    """Replace request IDs by their first-seen index so two boots compare."""
    if isinstance(value, dict):
        if "hex" in value and "dtype" in value:
            return value
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


def valid_part(record, dotted):
    """The field's comparable bits: each row's valid prefix for a partly written buffer."""
    value = field(record, dotted)
    width_field = VALID_WIDTH.get(dotted)
    if width_field is None or not isinstance(value, dict) or "hex" not in value:
        return value
    widths = decode(field(record, width_field))
    shape = value["shape"]
    if widths is None or len(shape) != 2 or len(widths) != shape[0]:
        return value
    per = HEX_PER_ELEMENT[value["dtype"]]
    rows, width = shape
    return {"dtype": value["dtype"], "valid_widths": widths,
            "rows": [value["hex"][r * width * per:(r * width + max(0, min(w, width))) * per]
                     for r, w in enumerate(widths)]}


def align(steps, sched):
    """(request index, step within request) -> worker record with its schedule record."""
    names = {}
    for record in steps:
        for r in record.get("req_ids", []):
            names.setdefault(r, f"req{len(names)}")
    schedules = [s for s in sched if s.get("num_scheduled_tokens")]
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
        if i < len(sched_list):
            record["sched"] = sched_list[i]
        keyed[(reqs, seen[reqs])] = record
    return keyed, names


def summary(record):
    if record is None:
        return None
    budget = record.get("budget") or {}
    return {"num_tokens": record.get("num_tokens"), "num_draft_tokens_per_req": record.get("num_draft_tokens_per_req"),
            "budget": {k: budget.get(k) for k in ("scheduled_drafts", "tau", "idle_before", "num_tokens")},
            "num_sampled": decode(field(record, "device.num_sampled")),
            "sampled_valid": [decode({"dtype": "int64", "hex": h}) for h in
                              (valid_part(record, "device.sampled_token_ids") or {}).get("rows", [])],
            "draft_tokens_next": decode(field(record, "device.draft_tokens_next")),
            "confidences": decode(field(record, "device.confidences")),
            "est_refits": record.get("est_refits"), "est_steps_since_refit": record.get("est_steps_since_refit"),
            "slope": decode(field(record, "device.est_slope")),
            "intercepts": decode(field(record, "device.est_intercepts")),
            "spec_tokens": field(record, "sched.spec_tokens"), "skip_drafts": field(record, "sched.skip_drafts")}


def rank_disagreements(directory, rank0_keyed, limit=5):
    out = []
    for rank in range(1, 16):
        files = sorted(glob.glob(os.path.join(directory, "**", f"steps-*-tp{rank}-*.jsonl"), recursive=True))
        if not files:
            continue
        keyed, _ = align(load_jsonl(files, KEEP), [])
        fields = ["num_tokens", "num_scheduled_tokens", "device.num_sampled", "device.dead_row_cut",
                  "device.draft_tokens_next", "device.confidences", "budget.num_tokens", "device.sampled_token_ids"]
        for key, ref in rank0_keyed.items():
            other = keyed.get(key)
            if other is None:
                out.append({"rank": rank, "key": list(key), "field": "step missing"})
                break
            bad = [f for f in fields if json.dumps(valid_part(ref, f), sort_keys=True)
                   != json.dumps(valid_part(other, f), sort_keys=True)]
            if bad:
                out.append({"rank": rank, "key": list(key), "fields": bad})
                if len(out) >= limit:
                    return out
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--rank-check", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    result = {"a": args.a, "b": args.b, "rank": args.rank}
    steps_a, sched_a = boot(args.a, args.rank)
    steps_b, sched_b = boot(args.b, args.rank)
    keyed_a, names_a = align(steps_a, sched_a)
    keyed_b, names_b = align(steps_b, sched_b)
    del steps_a, steps_b
    if args.rank_check:
        result["a_rank_disagreements"] = rank_disagreements(args.a, keyed_a)
        result["b_rank_disagreements"] = rank_disagreements(args.b, keyed_b)
    order = list(keyed_a)
    result["steps"] = {"a": len(keyed_a), "b": len(keyed_b), "requests": {"a": len(names_a), "b": len(names_b)}}
    ignored = {}
    first = None
    compared = 0
    # Every field's first differing step too (the scan goes on past the first difference
    # until the boots' steps stop aligning), so a bit-level difference that leaves the
    # decisions alone is told apart from the first decision that differs.
    first_by_field = {}
    differing_steps = {}
    shape_break = None
    for position, key in enumerate(order):
        if key not in keyed_b:
            shape_break = {"index": position, "key": list(key), "group": "schedule shape",
                           "field": "this request step exists in A only"}
            break
        ra, rb = keyed_a[key], keyed_b[key]
        compared += 1
        for group, fields in ORDER:
            for f in fields:
                va = canonical(valid_part(ra, f), names_a)
                vb = canonical(valid_part(rb, f), names_b)
                if json.dumps(va, sort_keys=True) != json.dumps(vb, sort_keys=True):
                    differing_steps[f] = differing_steps.get(f, 0) + 1
                    if f not in first_by_field:
                        first_by_field[f] = {"index": position, "key": list(key), "group": group}
                    if first is None:
                        first = {"index": position, "key": list(key), "group": group, "field": f, "a": va, "b": vb}
                elif f in VALID_WIDTH and json.dumps(field(ra, f)) != json.dumps(field(rb, f)):
                    ignored[f] = ignored.get(f, 0) + 1
    if first is None and shape_break is not None:
        first = shape_break
    result["aligned_steps_compared"] = compared
    result["steps_align_until"] = shape_break
    result["first_difference_by_field"] = dict(sorted(first_by_field.items(), key=lambda kv: kv[1]["index"]))
    result["differing_steps_by_field"] = differing_steps
    result["ignored_tail_differences"] = ignored
    result["first_difference"] = first
    if first:
        lo = max(0, first["index"] - args.context)
        result["window"] = [{"boot": label, "key": list(key), **summary(keyed.get(key))}
                            for key in order[lo:first["index"] + 1]
                            for label, keyed in (("a", keyed_a), ("b", keyed_b)) if keyed.get(key) is not None]
        verdicts = {
            "schedule shape": "the boots ran a different number of steps for a request (a decision differed earlier "
                              "in a field this tool does not record)",
            "scheduler": "the engine core scheduled the step differently (an engine-side input: arrival timing, the "
                         "copy allocation it read, or the finished/new request bookkeeping)",
            "budget-in": "the worker's budget read different inputs although the schedule matched",
            "budget-out": "the budget decided differently from equal inputs (host arithmetic)",
            "verify": "the target's verification differed on equal rows and drafts (target numerics)",
            "estimator": "the estimator's state differs with equal observations so far",
            "drafter": "the drafter produced other confidences or drafts from equal inputs (drafter numerics or state)",
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
