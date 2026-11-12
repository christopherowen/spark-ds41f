"""dual_probe.py's replies, per served name, compared across runs.

Usage: python3 dual_replies.py DUAL_JSONL [DUAL_JSONL ...]

Each file is one boot's dual_probe.py output (results/private/determinism/<run>/dual-<label>.jsonl).
For each file, prints the probe's verdict, the prompts steering changed, and per served name a
SHA-256 over its "alone" replies (prompt index, token IDs and log probabilities, in order). With
more than one file, says whether every file gives the same replies per name, and for each name
that differs, the first prompt and token where the first file and the others part.
"""
import hashlib
import json
import sys


def replies(path):
    rows = [json.loads(line) for line in open(path) if line.strip().startswith("{")]
    summary = [r for r in rows if "summary" in r][-1]["summary"]
    alone = {}
    for r in rows:
        if r.get("phase") == "alone":
            alone.setdefault(r["model"], []).append((r["prompt"], r["tokens"], r["logprobs"]))
    return summary, alone


def digest(items):
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


runs = [(path, *replies(path)) for path in sys.argv[1:]]
for path, summary, alone in runs:
    names = ", ".join(f"{name} {digest(items)[:12]}" for name, items in sorted(alone.items()))
    print(f"{path}: pass {summary['pass']}, moved {summary['detail'].get('moved_prompts')}; {names}")
if len(runs) > 1:
    first = runs[0][2]
    same = True
    for path, _, alone in runs[1:]:
        for name in sorted(set(first) | set(alone)):
            a, b = first.get(name, []), alone.get(name, [])
            if a == b:
                continue
            same = False
            for (pa, ta, la), (pb, tb, lb) in zip(a, b):
                if (ta, la) != (tb, lb):
                    k = next((i for i in range(min(len(ta), len(tb))) if (ta[i], la[i]) != (tb[i], lb[i])),
                             min(len(ta), len(tb)))
                    print(f"{path}: {name} differs from {runs[0][0]} at prompt {pa}, token {k}")
                    break
            else:
                print(f"{path}: {name} has {len(b)} replies, {runs[0][0]} {len(a)}")
    print("same replies per name in every file" if same else "replies differ")
