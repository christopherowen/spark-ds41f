#!/usr/bin/env python3
"""Long-prompt checks for prefill sequence parallelism.

usage: sp_check.py BASE_URL OUT.json

- needle: about 6K tokens of real text with a code sentence at 40% depth,
  then a question; the answer must contain the code.
- prompt logprobs: two cold prefills of the same ~5K-token real-text prompt
  (distinct cache salts) with prompt_logprobs=0, so arms can be compared
  token by token, and each arm's own run-to-run spread is recorded.
"""
import importlib.machinery
import importlib.util
import json
import sys
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
BASE = sys.argv[1].rstrip("/")
OUT = Path(sys.argv[2])


def post(path, body):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]

text = spark3.source_text(6000, 21)
cut = int(len(text) * 0.4)
needle_prompt = (text[:cut] + "\nThe secret deployment code is QUARTZ-4417.\n" + text[cut:]
                 + "\n\nWhat is the secret deployment code in the text above? Answer with the code only.")
answer = post("/v1/chat/completions", {
    "model": model, "messages": [{"role": "user", "content": needle_prompt}], "max_tokens": 24,
    "temperature": 0, "chat_template_kwargs": {"thinking": False}})
needle_text = answer["choices"][0]["message"]["content"]
needle = {"prompt_tokens": answer["usage"]["prompt_tokens"], "answer": needle_text,
          "pass": "QUARTZ-4417" in needle_text}
print(f"needle: {needle['prompt_tokens']} tokens, pass={needle['pass']}, answer={needle_text!r}", flush=True)

logprob_prompt = spark3.source_text(5000, 22)
runs = []
for _ in range(2):
    result = post("/v1/completions", {
        "model": model, "prompt": logprob_prompt, "max_tokens": 1, "temperature": 0,
        "prompt_logprobs": 0, "cache_salt": uuid.uuid4().hex})
    entries = result["choices"][0].get("prompt_logprobs") or []
    values = []
    for entry in entries:
        if not entry:
            values.append(None)
            continue
        values.append(max(item["logprob"] for item in entry.values()))
    runs.append(values)
spread = [abs(a - b) for a, b in zip(*runs) if a is not None and b is not None]
print(f"prompt logprobs: {len(runs[0])} tokens, run-to-run mean |diff| "
      f"{sum(spread) / max(len(spread), 1):.5f}, max {max(spread or [0]):.4f}", flush=True)
OUT.write_text(json.dumps({"needle": needle, "prompt_logprobs": runs}) + "\n")
