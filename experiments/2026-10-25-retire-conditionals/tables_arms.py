#!/usr/bin/env python3
"""The serving check's table: whether the retire-conditionals arm serves candidate 6's
outputs bit for bit, and starts without the removed work.

usage: tables_arms.py RUN LABEL [LABEL...]

The first label is the arm under test, the others are the references it is compared
with (retire-check.json: retire, then candidate6). Every arm must be identical within
itself at temperature 0, with log probabilities (determinism) and without (greedy). The
arm under test passes when, against each reference, the temperature-0 digests (the eight
prompts alone and the long prompt, with log probabilities and as text), the seeded
temperature-1 reference digest, every generation-control digest, the prompt-logprobs
digests and the image digests are equal, and its startup log has no B12X preparation,
temporary KV block or JIT monitor line and the reference's TileLang call count. The
seeded penalties at eight streams differ between boots of one build (candidate 6's
README), so they are recorded and not compared. The digests of candidate 5's window
(candidate 6's package files) are printed for the record. Exits 1 when a check fails.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE_0 = ("reference_sha256", "long_sha256", "reference_text_sha256", "long_text_sha256")
# Differs between boots of one build: experiments/2026-10-25-spark-ds41f-candidate/README.md.
VARIES = {"penalties-sampled/c8"}
REMOVED = ("b12x_preparation", "b12x_temporary_kv_blocks", "b12x_progress",
           "jit_monitor_activated", "jit_compile_reports")
# Candidate 5's window, candidate 6's package files
# (experiments/2026-10-25-spark-ds41f-candidate/README.md).
WINDOW = {"determinism": ("3eab5112", "db25fde3", "c9cf9dbe", "c6bb4f4e"),
              "sampled": "80b3b3f3", "images": "92679390"}


def summary(run: str, stem: str, label: str) -> dict:
    path = ROOT / f"results/private/determinism/{run}/{stem}-{label}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "summary" in record:
            return record
    return {}


def short(value) -> str:
    return value[:8] if isinstance(value, str) else str(value)


def main(argv: list[str]) -> int:
    run, labels = argv[0], argv[1:]
    stems = ("determinism", "greedy", "sampled", "controls", "prompt-logprobs", "images", "startup")
    records = {label: {stem: summary(run, stem, label) for stem in stems} for label in labels}
    ok = True
    for label in labels:
        record = records[label]
        print(f"{label}:")
        for stem in ("determinism", "greedy"):
            got = record[stem]
            print(f"  {stem}: identical {got.get('identical')}, "
                  + " / ".join(short(got.get(key)) for key in TEMPERATURE_0))
            ok &= got.get("identical") is True
        print(f"  seeded temperature 1: reference {short(record['sampled'].get('reference_sha256'))}")
        print(f"  images: {', '.join(f'{k} {short(v)}' for k, v in record['images'].get('digests', {}).items())}")
        startup = record["startup"]
        print("  startup: " + ", ".join(f"{key} {startup.get(key)}"
                                        for key in (*REMOVED, "tilelang_calls", "init_engine_s")))
    print("candidate 5's window: determinism " + " / ".join(WINDOW["determinism"])
          + f", seeded temperature 1 {WINDOW['sampled']}, images {WINDOW['images']}")
    tested = records[labels[0]]
    ok &= all(tested["startup"].get(key) == 0 for key in REMOVED)
    for other in labels[1:]:
        reference = records[other]
        differ = [f"{stem} {key}" for stem in ("determinism", "greedy") for key in TEMPERATURE_0
                  if tested[stem].get(key) is None
                  or tested[stem].get(key) != reference[stem].get(key)]
        if (tested["sampled"].get("reference_sha256") is None
                or tested["sampled"].get("reference_sha256") != reference["sampled"].get("reference_sha256")):
            differ.append("seeded temperature 1")
        for stem in ("controls", "prompt-logprobs", "images"):
            ours, theirs = tested[stem].get("digests"), reference[stem].get("digests")
            if not ours or not theirs:
                differ.append(f"{stem} missing")
                continue
            differ += [f"{stem} {key}" for key in sorted(set(ours) | set(theirs))
                       if key not in VARIES and ours.get(key) != theirs.get(key)]
            varies = [key for key in sorted(VARIES & set(ours)) if ours.get(key) != theirs.get(key)]
            if varies:
                print(f"  {stem} {', '.join(varies)} differ (varies between boots; record)")
        if tested["startup"].get("tilelang_calls") != reference["startup"].get("tilelang_calls"):
            differ.append("TileLang call count")
        print(f"{labels[0]} against {other}: " + (f"DIFFER: {', '.join(differ)}" if differ else "equal"))
        ok &= not differ
    print("pass" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
