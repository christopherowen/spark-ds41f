#!/usr/bin/env python3
"""The poison check's verdicts: each arm's records against the release's references and
against the zero arm of its job (zero-*: new blocks zeroed, the reference; nofill-*: not
zeroed, the end state after vLLM 0502; poison-*: filled with NaN bytes, the gate).

usage: tables_arms.py RUN LABEL [LABEL...]

From each arm's records under results/private/determinism/RUN (labels ending in
"tp3" are read against the triangle's references):

- determinism (with logprobs) and greedy (without), token_determinism_text.py: every
  scenario identical within the arm, and its four digests the release's;
- prefix-extension: the short-first-chunk hit identical to the prompt computed from
  scratch, the hit's length and first chunk;
- controls-digest and sampled: equal to the zero arm's, and to candidate 6's own gate
  (results/private/determinism/candidate-6-tp4, candidate-6) where it is recorded;
- pool-pressure: every request's tokens with the pool exhausted equal its tokens alone,
  whether the pass preempted, and the outputs equal across the arms;
- counters: requests with NaN logits, each node's device asserts and NaN reports, and
  whether the server still answered.

An arm passes when every record it has passes; a poison-unfixed arm (the positive
control) is expected to fail, by NaN reports or an engine that died. Release-suite runs
are compared with candidate 6's by suite_model.py.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The release protocol's temperature-0 references since the 1017 checkpoint
# (docs/release-benchmark.md; the triangle's from spark-ds41f-1022 at TP3): with
# logprobs (reference, long prompt) and as text.
REFERENCES = {
    "tp4": {"digests": ("3eab5112", "db25fde3"), "texts": ("c9cf9dbe", "c6bb4f4e")},
    "tp3": {"digests": ("62313cb5", "9742ac2c"), "texts": ("4d143977", "12fd8257")},
}
CANDIDATE_GATE = ("candidate-6-tp4", "candidate-6")


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


def short(digest) -> str:
    return (digest or "-")[:8]


def shape(label: str) -> str:
    return "tp3" if label.endswith("tp3") else "tp4"


def determinism(s: dict, label: str, stem: str) -> tuple[bool, str]:
    refs = REFERENCES[shape(label)]
    texts = (short(s.get("reference_text_sha256")), short(s.get("long_text_sha256")))
    digests = (short(s.get("reference_sha256")), short(s.get("long_sha256")))
    ok = bool(s.get("identical")) and texts == refs["texts"]
    line = f"identical {s.get('identical')}, text {texts[0]} / {texts[1]}"
    if stem == "determinism":
        ok = ok and digests == refs["digests"]
        line += f", logprob digests {digests[0]} / {digests[1]}"
    return ok, line + (" (the release's)" if ok else " (NOT the release's)")


def digest_of(s: dict | None, stem: str):
    if s is None:
        return None
    return s.get("digests") if stem == "controls-digest" else s.get("reference_sha256")


def main() -> int:
    run, labels = sys.argv[1], sys.argv[2:]
    zero = next((label for label in labels if label.startswith("zero")), None)
    verdicts = {}
    for label in labels:
        print(f"== {label}")
        checks = []
        for stem in ("determinism", "greedy"):
            s = summary(run, stem, label)
            if s is not None:
                ok, line = determinism(s, label, stem)
                checks.append(ok)
                print(f"  {stem}: {line}")
        s = summary(run, "prefix-extension", label)
        if s is not None:
            checks.append(bool(s["identical"]) and not s["failed"])
            print(f"  prefix extension: identical {s['identical']}, hit {s['hit_tokens']:.0f} of "
                  f"{s['hit_point']} tokens, first chunk {s['first_chunk']}"
                  f"{' (short)' if s['short_first_chunk'] else ' (NOT short: the hit missed the point)'}"
                  f", failed {s['failed']}")
        for stem in ("controls-digest", "sampled"):
            s = summary(run, stem, label)
            if s is None:
                continue
            mine = digest_of(s, stem)
            others = {"zero arm": digest_of(summary(run, stem, zero), stem) if zero and zero != label else None,
                      "candidate 6": digest_of(summary(CANDIDATE_GATE[0], stem, CANDIDATE_GATE[1]), stem)}
            compared = {name: other == mine for name, other in others.items() if other is not None}
            checks.append(all(compared.values()) and not s.get("failed"))
            print(f"  {stem}: " + (", ".join(f"{'equal to' if same else 'DIFFERS from'} the {name}'s"
                                              for name, same in compared.items()) or "no reference recorded"))
        s = summary(run, "pool-pressure", label)
        if s is not None:
            checks.append(bool(s["identical"]) and not s["failed"] and not s["nan_requests"])
            other = summary(run, "pool-pressure", zero) if zero and zero != label else None
            across = "" if other is None else (
                f", alone {'equal to' if other['alone_sha256'] == s['alone_sha256'] else 'DIFFERS from'}"
                f" the zero arm's, together {'equal' if other['together_sha256'] == s['together_sha256'] else 'differs (preemption points differ between runs)'}")
            print(f"  pool pressure: identical {s['identical']} (differ {s['differ']}), preempted "
                  f"{s['preemptions']:.0f} times{'' if s['pressure'] else ' (NO PRESSURE: inconclusive)'}"
                  f", prefix-cache hits {s['prefix_cache_hits']:.0f} tokens, NaN requests "
                  f"{s['nan_requests']:.0f}, failed {s['failed']}{across}")
        s = summary(run, "counters", label)
        if s is not None:
            nan = s.get("nan_requests", 0) + s.get("nan_reports", 0)
            checks.append(s.get("answered", False) and not nan)
            nodes = ", ".join(f"{node} {sum(v for k, v in c.items() if isinstance(v, int))}"
                              for node, c in s.get("nodes", {}).items())
            print(f"  counters: server answered {s.get('answered')}, NaN-logit requests "
                  f"{s.get('nan_requests', '-')} (counting {s.get('counting')}), node reports: "
                  f"{nodes or '-'}")
            for node, c in s.get("nodes", {}).items():
                if c.get("first"):
                    print(f"    {node}: {c['first']}")
        passed = bool(checks) and all(checks)
        expected_failure = "unfixed" in label
        verdicts[label] = passed
        if expected_failure:
            print(f"  verdict: {'positive control FAILED as expected' if not passed else 'positive control PASSED: the check did not see the drafter hazard'}")
        else:
            print(f"  verdict: {'PASS' if passed else 'FAIL' if checks else 'no records'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
