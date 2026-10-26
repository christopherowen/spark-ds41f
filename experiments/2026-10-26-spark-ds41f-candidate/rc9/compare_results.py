#!/usr/bin/env python3
"""The step harness's measured outputs, compared between runs.

usage: compare_results.py RESULT.json RESULT.json...

Each RESULT.json is iket_step.py's: the measured request's token ids and, per token, its
top-5 log probabilities. Prints each run's SHA-256 digests of the token ids and of the
token ids with the log probabilities; exits 1 unless every run has the same.
"""

import hashlib
import json
import sys


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    rows = []
    for path in sys.argv[1:]:
        result = json.load(open(path))
        rows.append((path, digest(result["token_ids"]),
                     digest([result["token_ids"], result["logprobs"]]), len(result["token_ids"])))
    for path, tokens, logprobs, count in rows:
        print(f"{tokens} {logprobs} {count:>4} tokens  {path}")
    same = len({(tokens, logprobs) for _, tokens, logprobs, _ in rows}) == 1
    print("identical" if same else "DIFFER")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
