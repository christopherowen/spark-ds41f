#!/usr/bin/env python3
"""Long-context retrieval and decode at the full context length.

Each request is one ~1M-token prompt of real text (the bench's standard-library
source text) with a unique code word planted at a given depth, ending with a
question about it. Completions are greedy and streamed, so each request also
gives its time to first token (the prefill) and its decode rate with the whole
context resident. A request passes when the answer contains the code word.
Writes one JSON line per request and exits non-zero on any miss or error.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3_bench", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)

WORDS = ("amber", "basalt", "cobalt", "dahlia", "ember", "fennel", "garnet", "harbor", "indigo", "juniper")


def request(url, prompt, max_tokens):
    payload = {"model": "deepseek-v4.1-flash", "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    stamps, text, usage = [], [], None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("text"):
                    now = time.monotonic()
                    first = now if first is None else first
                    stamps.append(now)
                    text.append(choice["text"])
    decode_s = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    completion = (usage or {}).get("completion_tokens") or len(stamps)
    return {"ttft_s": round(first - started, 1) if first else None,
            "decode_tok_s": round((completion - 1) / decode_s, 1) if decode_s else None,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": completion,
            "answer": "".join(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--tokens", type=int, default=960_000, help="approximate prompt length")
    parser.add_argument("--depths", default="0.1,0.9", help="needle depths as fractions of the prompt")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    failures = 0
    for index, depth in enumerate(float(d) for d in args.depths.split(",")):
        rng = random.Random(4100 + index)
        word = f"{rng.choice(WORDS)}-{rng.randrange(1000, 9999)}"
        body = spark3.source_text(args.tokens, 7000 + index)
        cut = int(len(body) * depth)
        cut = body.rfind("\n", 0, cut) + 1 or cut
        needle = f"\n# The secret code for this document is {word}. Remember it.\n"
        prompt = (body[:cut] + needle + body[cut:]
                  + "\n\n# Question: what is the secret code for this document, stated in a comment above?\n# Answer: The secret code is")
        try:
            result = request(args.url, prompt, args.max_tokens)
            result.update(depth=depth, code=word, found=word in result["answer"])
        except Exception as error:  # noqa: BLE001
            result = {"depth": depth, "code": word, "found": False, "error": f"{type(error).__name__}: {error}"}
        failures += not result["found"]
        print(json.dumps(result), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
