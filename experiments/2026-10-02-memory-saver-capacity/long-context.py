#!/usr/bin/env python3
"""Validate retrieval near the configured limit using real source text.

Run on dgx1 inside an exclusive cluster window. Tokenize the actual chat request
before sending it. Three depths each use a distinct phrase; assert that every
served prompt exceeds 500,000 tokens and leaves room for the answer.
"""
import json
from pathlib import Path
import time
import urllib.request

BASE = "http://10.0.1.71:8000"
MODEL = "deepseek-v4.1-flash"
TARGET = 520000


def post(path, body):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def main():
    root = Path.home() / "projects/spark3-vllm-ds41f/.work/upstreams/vllm/vllm"
    pieces, length = [], 0
    for path in sorted(root.rglob("*.py")):
        piece = f"# file: {path.relative_to(root)}\n{path.read_text(errors='replace')}\n"
        pieces.append(piece)
        length += len(piece)
        if length > TARGET * 5:
            break
    corpus = "".join(pieces)
    assert len(corpus) > TARGET * 4, "Prepared source corpus is missing or too short"
    for depth in (0.1, 0.5, 0.9):
        code = f"COPPER-OTTER-{int(depth * 1000):04d}-CEDAR"
        lo, hi, messages = 0, len(corpus), None
        for _ in range(24):
            size = (lo + hi) // 2
            text = corpus[:size]
            at = text.rfind("\n", 0, int(len(text) * depth)) + 1
            text = text[:at] + f"\n# The vault access phrase is {code}. Remember it.\n" + text[at:]
            candidate = [{"role": "user", "content": text +
                          "\nWhat is the vault access phrase stated above? Reply with the phrase only."}]
            count = post("/tokenize", {"model": MODEL, "messages": candidate,
                                       "chat_template_kwargs": {"thinking": False}})["count"]
            if TARGET - 1024 <= count <= TARGET:
                messages = candidate
                break
            if count > TARGET:
                hi = size
            else:
                lo = size + 1
        assert messages is not None, "Could not fit the near-limit prompt"
        print(json.dumps({"event": "start", "depth": depth, "tokenized_prompt": count}), flush=True)
        start = time.monotonic()
        reply = post("/v1/chat/completions", {"model": MODEL, "messages": messages,
                     "temperature": 0, "max_tokens": 32,
                     "chat_template_kwargs": {"thinking": False}})
        answer = (reply["choices"][0]["message"]["content"] or "").strip()
        served = reply["usage"]["prompt_tokens"]
        ok = code in answer and 500000 < served <= TARGET
        print(json.dumps({"event": "result", "depth": depth, "prompt_tokens": served,
                          "seconds": time.monotonic() - start, "answer": answer,
                          "passed": ok}), flush=True)
        assert ok, "Near-limit retrieval failed"


if __name__ == "__main__":
    main()
