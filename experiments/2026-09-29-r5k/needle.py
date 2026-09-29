#!/usr/bin/env python3
"""Long-context retrieval check: hide one fact at several depths in real text.

usage: needle.py BASE_URL TOKENS [DEPTH ...]
Builds a prompt of about TOKENS tokens from the prepared vLLM sources, puts a
unique code phrase at each depth (fraction of the prompt), asks for it back at
temperature 0 with reasoning off, and reports the server's prompt token count,
time to answer, and whether the answer contains the phrase.
"""
import json
import pathlib
import sys
import time
import urllib.request

base, target = sys.argv[1].rstrip("/"), int(sys.argv[2])
depths = [float(d) for d in sys.argv[3:]] or [0.1, 0.5, 0.9]
root = pathlib.Path.home() / "projects/spark3-vllm-ds41f/.work/upstreams/vllm/vllm"
text, chars = [], 0
for path in sorted(root.rglob("*.py")):
    body = path.read_text(errors="replace")
    text.append(f"# file: {path.relative_to(root)}\n{body}\n")
    chars += len(text[-1])
    if chars > target * 3.2:  # about 3.2 characters per token for this source
        break
corpus = "".join(text)
model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]
failures = 0
for depth in depths:
    code = f"BLUE-HERON-{int(depth * 1000):04d}-QUARTZ"
    at = corpus.rfind("\n", 0, int(len(corpus) * depth)) + 1
    haystack = corpus[:at] + f"# The vault access phrase is {code}. Remember it.\n" + corpus[at:]
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": haystack + "\n\nWhat is the vault access phrase stated in the text above? "
            "Reply with the phrase only.",
        }],
        "max_tokens": 32,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    reply = json.load(urllib.request.urlopen(request, timeout=1800))
    answer = (reply["choices"][0]["message"]["content"] or "").strip()
    ok = code in answer
    failures += not ok
    verdict = "PASS" if ok else "FAIL"
    print(f"depth {depth:.2f}: prompt {reply['usage']['prompt_tokens']} tokens, "
          f"{time.time() - started:.1f} s, {verdict} ({answer[:60]!r})", flush=True)
sys.exit(1 if failures else 0)
