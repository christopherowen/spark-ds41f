#!/usr/bin/env python3
"""The copy probe's categories walked through every copy-drafting variant.

usage: copy_cases.py SNAPSHOT OUT.json [--prompts 12] [--tokens 1500]

SNAPSHOT needs the checkpoint's tokenizer.json and encoding/ (nothing else is read). Each
category is ``--prompts`` conversations built like copy_probe.py's (cases.py), with the ideal
copy as the reply. Every variant of variants.py walks every reply; the histograms go to OUT.json
by category, with the suffix drafter that copy_probe.py simulated (exact, m = 8, up to 16) as
the check against the 2026-10-23 measurements (15.9 / 14.9 / 7.1 tokens a step).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import cases  # noqa: E402
import variants  # noqa: E402
from servebench import prompts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    snap = Path(args.snapshot)
    sys.path.insert(0, str(snap / "encoding"))
    from encoding import encode_messages
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731
    out = {"tokens": args.tokens, "prompts": args.prompts, "categories": {}}
    for name in cases.CATEGORIES:
        hists, reply_tokens = {}, 0
        for i in range(args.prompts):
            seed = args.seed * 1000 + i
            while True:  # a document with no function cannot make a tool case: the next seed
                code = cases.document(prompts, args.tokens, seed)
                try:
                    messages, reply = cases.conversation(name, code)
                    break
                except ValueError:
                    seed += args.prompts
            ids, start = cases.render(encode_messages, encode, messages, reply)
            reply_tokens += len(ids) - start
            stream = variants.Stream(np.asarray(ids, dtype=np.int64), [(start, len(ids))])
            for fam, m, k, pol in variants.variants(families=("exact", "request")):
                h = hists.setdefault(f"{fam}_m{m}_k{k}_{pol}", Counter())
                variants.walk(stream, fam, m, k, pol, h)
        check = hists["exact_m8_k16_full"]
        tokens = sum(key[3] * c for key, c in check.items())
        suffix_steps = sum(c for key, c in check.items() if key[0] == "suffix")
        print(f"{name}: {args.prompts} replies, {reply_tokens} tokens; exact m8 k16: "
              f"{tokens / sum(check.values()):.2f} tokens a step, {suffix_steps} suffix steps", flush=True)
        out["categories"][name] = {
            "reply_tokens": reply_tokens,
            "histograms": {key: [[*k, c] for k, c in sorted(h.items())] for key, h in hists.items()},
        }
    with open(args.out, "w") as handle:
        json.dump(out, handle)
        handle.write("\n")


if __name__ == "__main__":
    main()
