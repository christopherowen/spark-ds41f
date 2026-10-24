#!/usr/bin/env python3
"""The logprob-token-ids check's table: each arm's own checks, and whether each fix arm's
temperature-0 replies are those of the arm it fixes.

usage: tables_arms.py RUN LABEL [LABEL...]

Reads logprob_ids_check.py's summaries
(results/private/determinism/RUN/logprob-ids-LABEL.jsonl). The arm ``X-0267`` is compared
with ``X-0266``: it passes when its own checks pass and its digests of the seven other
prompts (alone and beside the probe), of the probe's reply and of its chosen tokens'
logprobs (alone and beside them) equal those of ``X-0266``. Every arm is also compared
with the first, for the record. Exits 1 when an arm's checks or a fix arm's comparison
fail.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARED = ("others_sha256", "probe_sha256", "probe_logprobs_sha256",
            "together_others_sha256", "together_probe_sha256",
            "together_probe_logprobs_sha256")


def summary(run: str, label: str) -> dict | None:
    path = ROOT / f"results/private/determinism/{run}/logprob-ids-{label}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return None


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    records = {label: summary(run, label) for label in labels}
    ok = True
    for label in labels:
        record = records[label]
        if record is None:
            print(f"{label}: no summary")
            ok = False
            continue
        print(f"{label}: own checks {'pass' if record['pass'] else 'FAIL'}"
              + "".join(f"\n  {failure}" for failure in record["failures"]))
        if "bug_status" in record:
            print(f"  the request with ids, without the fix: HTTP {record['bug_status']}; "
                  "the engine served the probe again")
        if not record["reference"] and "entries" in record:
            print(f"  ids: {record['entries']} entries for {record['tokens']} tokens, "
                  f"problems {record['entry_problems']}; alone and beside the others "
                  f"{'equal' if record.get('together_ids_logprobs_equal') else 'differ'}")
            print(f"  completion: {record.get('completion_entries')} entries, problems "
                  f"{record.get('completion_problems')}, tokens "
                  f"{'equal' if record.get('completion_tokens_equal') else 'differ'}")
            print(f"  teacher: {record.get('teacher')}")
        ok &= record["pass"]
        for other, gating in ((label.replace("-0267", "-0266"), True), (labels[0], False)):
            if other == label or other not in records or records[other] is None:
                continue
            if not gating and other == label.replace("-0267", "-0266"):
                continue
            differ = [key for key in COMPARED if record.get(key) != records[other].get(key)]
            print(f"  against {other}: "
                  + (f"DIFFER in {', '.join(differ)}" if differ else "temperature-0 replies equal")
                  + ("" if gating else " (record)"))
            ok &= not (gating and differ)
    print("pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
