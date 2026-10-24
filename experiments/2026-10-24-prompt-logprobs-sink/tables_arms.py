#!/usr/bin/env python3
"""The prompt-logprobs check's table: each arm's own checks, and whether each fix arm's
temperature-0 replies are those of the arm it fixes.

usage: tables_arms.py RUN LABEL [LABEL...]

Reads prompt_logprobs_check.py's summaries
(results/private/determinism/RUN/prompt-logprobs-LABEL.jsonl). The arm ``X-fix`` is
compared with ``X``: it passes when its own checks pass and its digests of the seven other
prompts (alone and beside the probe) and of the probe's reply (alone and beside them)
equal those of ``X``. Every arm is also compared with the first, for the record. Exits 1
when an arm's checks or a fix arm's comparison fail.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARED = ("others_sha256", "probe_sha256", "together_others_sha256", "together_probe_sha256")


def summary(run: str, label: str) -> dict | None:
    path = ROOT / f"results/private/determinism/{run}/prompt-logprobs-{label}.jsonl"
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
        if not record["reference"] and "prompt_tokens" in record:
            print(f"  prompt logprobs: {record.get('prompt_tokens')} prompt tokens, "
                  f"problems {record.get('prompt_logprob_problems')}, alone and beside the "
                  f"others {'equal' if record.get('prompt_logprobs_sha256') == record.get('together_prompt_logprobs_sha256') else 'differ'}")
            print(f"  teacher: {record.get('teacher')}")
        ok &= record["pass"]
        for other, gating in ((label.removesuffix("-fix"), True), (labels[0], False)):
            if other == label or other not in records or records[other] is None:
                continue
            if not gating and other == label.removesuffix("-fix"):
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
