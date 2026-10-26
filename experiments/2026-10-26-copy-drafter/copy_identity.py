#!/usr/bin/env python3
"""Temperature-0 text and tokens a step on copy-heavy prompts (a lab extra).

usage: copy_identity.py HEAD_URL [--prompts 12] [--seed 0] [--thinking] [--streams 8]

The copy probe's categories (experiments/2026-10-23-next-release/acceptance/
copy_probe.py: prose, reproduce a file, the file with comments added, an edit tool
call), at temperature 0, each prompt served alone and then the category's prompts all
at once on ``--streams`` streams, thinking off or (``--thinking``) on. One JSON line a
category: whether every output in company equals its output alone, the SHA-256 digest
of the outputs alone (token ids; equal digests between two arms are identical text),
and the tokens a step from the server's spec-decode counters over the alone pass.
Exits 0 either way; the run records. The copy-drafts gate compares the digests of
DSpark alone and DSpark with copy drafts on the same build.
"""
import argparse
import hashlib
import json
import sys
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "2026-10-23-next-release" / "acceptance"))
sys.path.insert(0, str(HERE.parents[2]))
import copy_probe  # noqa: E402

CATEGORIES = ("prose", "copy", "edit", "tool")


def chat(url: str, messages: list, max_tokens: int, tools, thinking: bool) -> list[int]:
    body = {"model": copy_probe.MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "return_token_ids": True,
            "chat_template_kwargs": {"thinking": thinking}}
    if tools:
        body.update(tools=tools, tool_choice="required")
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())["choices"][0]["token_ids"]


def together(url: str, requests: list, thinking: bool, streams: int) -> list:
    outputs = [None] * len(requests)

    def run(start: int) -> None:
        for i in range(start, len(requests), streams):
            messages, max_tokens, tools = requests[i]
            try:
                outputs[i] = chat(url, messages, max_tokens, tools, thinking)
            except (OSError, ValueError, KeyError) as error:
                outputs[i] = f"error: {str(error)[:200]}"

    threads = [threading.Thread(target=run, args=(s,)) for s in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--streams", type=int, default=8)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    summary = {"thinking": args.thinking, "identical": True, "digests": {}}
    for name in CATEGORIES:
        requests = list(copy_probe.cases(name, args.prompts, args.seed))
        before = copy_probe.counters(url)
        alone = []
        for messages, max_tokens, tools in requests:
            try:
                alone.append(chat(url, messages, max_tokens, tools, args.thinking))
            except (OSError, ValueError, KeyError) as error:
                alone.append(f"error: {str(error)[:200]}")
        after = copy_probe.counters(url)
        drafts = after[copy_probe.COUNTERS[0]] - before[copy_probe.COUNTERS[0]]
        accepted = after[copy_probe.COUNTERS[2]] - before[copy_probe.COUNTERS[2]]
        company = together(url, requests, args.thinking, args.streams)
        same = all(a == b and not isinstance(a, str) for a, b in zip(alone, company))
        digest = hashlib.sha256(json.dumps(alone).encode()).hexdigest()[:16]
        summary["identical"] &= same
        summary["digests"][name] = digest
        print(json.dumps({
            "category": name, "requests": len(requests), "thinking": args.thinking,
            "identical_in_company": same, "digest": digest,
            "output_tokens": sum(len(o) for o in alone if not isinstance(o, str)),
            "tokens_per_step": round((accepted + drafts) / drafts, 3) if drafts else None,
        }), flush=True)
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
