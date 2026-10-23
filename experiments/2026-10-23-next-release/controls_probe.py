#!/usr/bin/env python3
"""The generation controls (vLLM 0256) and the DeepSeek API thinking switch (0257), live.

usage: controls_probe.py URL [--tokenizer PATH]

Each check sends chat requests with max_tokens 1, temperature 0 and return_token_ids, and reads
the prompt token IDs the engine used:
- after_assistant: a conversation ending in an assistant message, with add_generation_prompt,
  ends on the opened assistant turn (<｜Assistant｜> then the thinking token), not on the
  message's end-of-sentence token;
- continue_final: the same conversation with continue_final_message and
  add_generation_prompt false ends on the assistant's own text;
- thinking_disabled: top-level thinking {"type": "disabled"} gives the prompt of
  chat_template_kwargs {"thinking": false}, also beside reasoning_effort "high", and differs
  from the default (thinking on);
- thinking_unknown: thinking {"type": "maybe"} is refused with HTTP 400.
Prints one JSON line with a "summary"; exits 1 when any check fails. Stdlib only.
"""
import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request

TOKENIZER_GLOB = ("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
                  "snapshots/*/tokenizer.json")
BAR = "｜"


def post(base, body):
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    path = args.tokenizer or sorted(glob.glob(os.path.expanduser(TOKENIZER_GLOB)))[-1]
    added = {t["content"]: t["id"] for t in json.load(open(path))["added_tokens"]}
    assistant, think, end_think = added[f"<{BAR}Assistant{BAR}>"], added["<think>"], added["</think>"]
    eos = added[f"<{BAR}end▁of▁sentence{BAR}>"]
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.loads(response.read())["data"][0]["id"]

    def prompt(messages, **extra):
        status, data = post(base, {"model": model, "messages": messages, "max_tokens": 1,
                                   "temperature": 0, "return_token_ids": True, **extra})
        return status, (data or {}).get("prompt_token_ids")

    conversation = [{"role": "user", "content": "Name a colour."},
                    {"role": "assistant", "content": "Blue"}]
    results = {}
    _, ids = prompt(conversation, add_generation_prompt=True)
    results["after_assistant"] = {"pass": bool(ids) and ids[-2:] == [assistant, think] and eos in ids,
                                  "tail": (ids or [])[-3:]}
    _, ids = prompt(conversation, add_generation_prompt=False, continue_final_message=True)
    results["continue_final"] = {"pass": bool(ids) and ids[-1] not in (eos, assistant, think, end_think),
                                 "tail": (ids or [])[-3:]}
    hola = [{"role": "user", "content": "Hola"}]
    _, chat = prompt(hola, chat_template_kwargs={"thinking": False})
    _, disabled = prompt(hola, thinking={"type": "disabled"})
    _, beside = prompt(hola, thinking={"type": "disabled"}, reasoning_effort="high")
    _, default = prompt(hola)
    results["thinking_disabled"] = {"pass": bool(chat) and disabled == chat and beside == chat and default != chat,
                                    "lengths": [len(chat or []), len(disabled or []), len(beside or []),
                                                len(default or [])]}
    status, _ = prompt(hola, thinking={"type": "maybe"})
    results["thinking_unknown"] = {"pass": status == 400, "status": status}
    ok = all(r["pass"] for r in results.values())
    print(json.dumps({"summary": {"pass": ok, "checks": results, "model": model}}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
