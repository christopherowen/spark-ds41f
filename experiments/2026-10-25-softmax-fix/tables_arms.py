#!/usr/bin/env python3
"""The softmax fix's serving table: each arm against the one labelled "release" (else
the first).

usage: tables_arms.py RUN LABEL [LABEL...]

From each arm's records under results/private/determinism/RUN:

- determinism (token_determinism_text.py with logprobs) and greedy (without): every
  arm identical within itself; text digests equal across arms (and, without
  logprobs, the release protocol's references); the logprob digests, which the fix
  changes by design;
- logprobs (logprob_margins.py): for each mode (greedy, prompt, sampled), how many
  chosen-token log probabilities each arm reports as exactly 0 or in (-1e-5, 0) and
  on the 2^-23 grid; whether the arms' tokens are identical; and, where they are, the
  size of every difference: the largest |delta| of the chosen tokens' and of the top
  alternatives' log probabilities, and the values the later arm reports where the
  first reported exactly 0.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The release protocol's temperature-0 text references at tp4-ring
# (docs/release-benchmark.md).
TEXT_REFERENCES = ("c9cf9dbe", "c6bb4f4e")


def records(run: str, stem: str, label: str) -> list[dict]:
    path = ROOT / f"results/private/determinism/{run}/{stem}-{label}.jsonl"
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summary(run: str, stem: str, label: str) -> dict | None:
    found = [r for r in records(run, stem, label) if "summary" in r]
    return found[-1] if found else None


def short(digest: str | None) -> str:
    return (digest or "-")[:8]


def determinism(run: str, labels: list[str]) -> None:
    for stem, what in (("determinism", "with logprobs"), ("greedy", "without logprobs")):
        print(f"temperature 0, {what}:")
        first = None
        for label in labels:
            s = summary(run, stem, label)
            if s is None:
                print(f"  {label}: no record")
                continue
            texts = (short(s.get("reference_text_sha256")), short(s.get("long_text_sha256")))
            values = (short(s.get("reference_sha256")), short(s.get("long_sha256")))
            first = first or texts
            print(f"  {label}: identical {s['identical']}, text {texts[0]} / {texts[1]} "
                  f"({'equal' if texts == first else 'DIFFER'}"
                  + (", the release references" if texts == TEXT_REFERENCES else "")
                  + f"), digests {values[0]} / {values[1]}")


def pairs(mine: list[dict], theirs: list[dict]):
    """Requests of one mode, paired by index."""
    by_index = {r["index"]: r for r in theirs}
    for record in mine:
        other = by_index.get(record["index"])
        if other is not None:
            yield record, other


def top_deltas(a: list, b: list) -> list[float]:
    """|delta| of the alternatives both report at a position (by token)."""
    theirs = {str(t): v for t, v in b}
    return [abs(v - theirs[str(t)]) for t, v in a if str(t) in theirs]


def compare(run: str, base: str, label: str) -> None:
    print(f"{label} against {base}, logprobs near 0:")
    mine, theirs = records(run, "logprobs", label), records(run, "logprobs", base)
    for mode in ("greedy", "prompt", "sampled"):
        a = [r for r in mine if r.get("mode") == mode]
        b = [r for r in theirs if r.get("mode") == mode]
        if not a or not b:
            print(f"  {mode}: no records")
            continue
        sa = summary(run, "logprobs", label)[mode]["chosen"]
        sb = summary(run, "logprobs", base)[mode]["chosen"]
        print(f"  {mode}: {base} exactly 0 {sb['zero']}/{sb['n']}, in (-1e-5, 0) "
              f"{sb['tiny']} ({sb['on_grid']} on the 2^-23 grid); {label} exactly 0 "
              f"{sa['zero']}/{sa['n']}, in (-1e-5, 0) {sa['tiny']} ({sa['on_grid']} on "
              f"the grid)")
        same, chosen, alternatives, at_zero = 0, [], [], []
        for x, y in pairs(a, b):
            if x["tokens"] != y["tokens"]:
                first = next((i for i, (s, t) in enumerate(zip(x["tokens"], y["tokens"]))
                              if s != t), min(len(x["tokens"]), len(y["tokens"])))
                print(f"    request {x['index']}: tokens differ from position {first}")
                continue
            same += 1
            for i, (u, v) in enumerate(zip(x["logprobs"], y["logprobs"])):
                chosen.append(abs(u - v))
                if v == 0.0:
                    at_zero.append(u)
                if x.get("top") and y.get("top"):
                    alternatives += top_deltas(x["top"][i], y["top"][i])
        print(f"    tokens identical in {same}/{len(a)} requests")
        if chosen:
            print(f"    chosen |delta|: max {max(chosen):.3e}, mean {statistics.fmean(chosen):.3e}"
                  f", nonzero {sum(d > 0 for d in chosen)}/{len(chosen)}")
        if alternatives:
            print(f"    top alternatives |delta|: max {max(alternatives):.3e}, mean "
                  f"{statistics.fmean(alternatives):.3e}")
        if at_zero:
            ordered = sorted(at_zero)
            print(f"    where {base} reported 0, {label} reports {len(at_zero)} values: "
                  f"min {ordered[0]:.3e}, median {ordered[len(ordered) // 2]:.3e}, max "
                  f"{ordered[-1]:.3e}, still 0: {sum(u == 0.0 for u in at_zero)}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run, labels = argv[0], argv[1:]
    determinism(run, labels)
    base = "release" if "release" in labels else labels[0]
    for label in labels:
        if label != base:
            compare(run, base, label)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
