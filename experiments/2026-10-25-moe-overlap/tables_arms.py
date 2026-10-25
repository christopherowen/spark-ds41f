#!/usr/bin/env python3
"""The A/B's output check and distinct-prompt rates. Both changes are bit-identical by
design, so an arm passes only with candidate 6's outputs exactly.

usage: tables_arms.py RUN LABEL [LABEL...]   (from the deployment checkout)

For each arm: experiments/2026-10-12-r6g/token_determinism_text.py's summaries (with
log probabilities, "determinism", and greedy without, "greedy"), whether each is
identical within itself, and scripts/distinct_streams.py's rates at 16, 8 and 4 streams.
Each arm after the first is compared with the first: every reference digest (log
probabilities and texts, the eight prompts and the long prompt) must be equal. The rates
are shown for reference only; the screen suite's engine grid and single-stream suite
judge speed.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = ("reference_sha256", "long_sha256", "reference_text_sha256", "long_text_sha256")
STREAMS = (16, 8, 4)


def summary(run: str, stem: str, label: str) -> dict | None:
    path = ROOT / f"results/private/determinism/{run}/{stem}-{label}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return None


def digests(record: dict | None) -> tuple | None:
    return record and tuple(record.get(key) for key in DIGESTS)


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    outputs, rates = {}, {}
    for label in labels:
        print(f"{label}:")
        outputs[label] = {}
        for stem in ("determinism", "greedy"):
            record = summary(run, stem, label)
            outputs[label][stem] = digests(record)
            print(f"  {stem}: identical {record and record.get('identical')}, "
                  f"references {record and record.get('reference_sha256', '')[:8]} / "
                  f"{record and record.get('long_sha256', '')[:8]}")
        rates[label] = {}
        for streams in STREAMS:
            record = summary(run, f"c{streams}-distinct", label)
            rates[label][streams] = record and record.get("tps_mean")
        print("  distinct prompts: " + ", ".join(
            f"c{streams} {rates[label][streams]} tok/s" for streams in STREAMS))
    first = labels[0]
    for label in labels[1:]:
        for stem in ("determinism", "greedy"):
            same = outputs[label][stem] is not None and outputs[label][stem] == outputs[first][stem]
            print(f"{label} against {first}, {stem}: digests {'equal' if same else 'DIFFER'}")
        changes = []
        for streams in STREAMS:
            ours, theirs = rates[first][streams], rates[label][streams]
            if ours and theirs:
                changes.append(f"c{streams} {100 * (ours - theirs) / theirs:+.2f}%")
        print(f"  {first} against {label}, distinct prompts: {', '.join(changes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
