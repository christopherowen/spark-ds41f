#!/usr/bin/env python3
"""Per-request draft acceptance with token ids: the engine side of the reference drafter probe.

usage: per_request_acceptance.py BASE_URL [--suite N] [--gsm8k N] [--shots K] [--max-tokens T]
                                 [--suite-tokens T] [--full-prompt] [--data DIR]

One request at a time on an otherwise idle server, temperature 0, with return_token_ids. The
per-position speculative-decoding counters are read around every request, so each request
carries its own drafts (verification steps), accepted and verified drafts by position and its
token ids. One JSON line a request, in reference/drafter_probe.py's case format (the counters
under "engine"), then one summary line.

- --suite N: the first N of servebench's default prompts (the first 16 are one per template),
  chat with thinking off, --suite-tokens tokens with ignore_eos, as the release suite's single
  stream writes them.
- --gsm8k N: the first N GSM8K test questions behind the first K training examples, exactly
  ../2026-10-26-cluster-program/gsm8k_acceptance.py's prompts and stop strings, raw
  /v1/completions, at most --max-tokens. With --chat, the same text is one user turn of the
  chat format with thinking off (SGLang PR 45450's GSM8K-200), with no stop strings.
- --full-prompt: every request asks for prompt_logprobs=1. The DS4.1 model state runs such a
  request keep_full: every prompt row goes through the decoder layers and its replay floor is
  the prefix start (vllm/models/deepseek_v4_1/ced.py, replay_rows), which is CED off for that
  request alone. Without it the same prompts run under CED, so the two are an A/B on one boot.
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "2026-10-26-cluster-program"))
from servebench import prom  # noqa: E402
from servebench.prompts import default_prompts  # noqa: E402

import gsm8k_acceptance as gsm8k  # noqa: E402

ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
VERIFIED_POS = "vllm:spec_decode_num_draft_tokens_per_pos_total"


def counters(base: str) -> dict:
    text = prom.fetch(base, timeout=60)
    if text is None:
        raise SystemExit(f"{base}/metrics unreachable")
    totals = prom.parse(text)
    return {"drafts": totals.get(prom.VLLM["drafts"], 0.0),
            "verified": totals.get(prom.VLLM["draft_tokens"], 0.0),
            "accepted": totals.get(prom.VLLM["accepted_tokens"], 0.0),
            "generation_tokens": totals.get(prom.VLLM["generation_tokens"], 0.0),
            "finished": totals.get(prom.VLLM["requests_finished"], 0.0),
            "accepted_pos": prom.by_label(text, ACCEPTED_POS, "position"),
            "verified_pos": prom.by_label(text, VERIFIED_POS, "position")}


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def delta(before: dict, after: dict) -> dict:
    positions = sorted(set(after["accepted_pos"]) | set(after["verified_pos"]), key=int)
    return {
        "drafts": after["drafts"] - before["drafts"],
        "verified": after["verified"] - before["verified"],
        "accepted": after["accepted"] - before["accepted"],
        "generation_tokens": after["generation_tokens"] - before["generation_tokens"],
        "finished": after["finished"] - before["finished"],
        "accepted_by_position": [after["accepted_pos"].get(p, 0.0) - before["accepted_pos"].get(p, 0.0)
                                 for p in positions],
        "verified_by_position": [after["verified_pos"].get(p, 0.0) - before["verified_pos"].get(p, 0.0)
                                 for p in positions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--suite", type=int, default=0)
    parser.add_argument("--suite-tokens", type=int, default=256)
    parser.add_argument("--gsm8k", type=int, default=0)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chat", action="store_true", help="GSM8K as one chat turn, thinking off")
    parser.add_argument("--full-prompt", action="store_true")
    parser.add_argument("--data", type=Path, default=Path.home() / "spark-lab" / "inputs" / "gsm8k")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    extra = {"prompt_logprobs": 1} if args.full_prompt else {}

    requests = []
    labels = {}
    for prompt in default_prompts()[: args.suite]:
        requests.append((f"suite-{prompt['id']}", "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt["text"]}],
            "max_tokens": args.suite_tokens, "min_tokens": args.suite_tokens, "ignore_eos": True,
            "temperature": 0, "return_token_ids": True, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"thinking": False}, **extra}))
    if args.gsm8k:
        texts, answers = gsm8k.prompts(args.data, args.gsm8k, args.shots)
        for index, text in enumerate(texts):
            labels[f"gsm8k-chat-{index:03d}" if args.chat else f"gsm8k-{index:03d}"] = answers[index]
            if args.chat:
                # SGLang PR 45450's form: the same few-shot text as one user turn in the
                # DeepSeek-V4.1 chat format, thinking off.
                requests.append((f"gsm8k-chat-{index:03d}", "/v1/chat/completions", {
                    "model": model, "messages": [{"role": "user", "content": text}],
                    "max_tokens": args.max_tokens, "temperature": 0, "return_token_ids": True,
                    "cache_salt": uuid.uuid4().hex, "chat_template_kwargs": {"thinking": False},
                    **extra}))
                continue
            requests.append((f"gsm8k-{index:03d}", "/v1/completions", {
                "model": model, "prompt": text, "max_tokens": args.max_tokens, "temperature": 0,
                "stop": gsm8k.STOP, "return_token_ids": True, "cache_salt": uuid.uuid4().hex, **extra}))

    prom.require_idle(base, "per-request acceptance")
    totals = {"drafts": 0.0, "accepted": 0.0, "verified": 0.0}
    by_position = None
    correct = 0
    began = time.time()
    for case_id, path, body in requests:
        before = counters(base)
        answer = post(base + path, body)
        after = counters(base)
        engine = delta(before, after)
        choice = answer["choices"][0]
        prompt_ids = answer.get("prompt_token_ids") or choice.get("prompt_token_ids")
        record = {"id": case_id, "prompt_ids": prompt_ids, "output_ids": choice.get("token_ids"),
                  "finish_reason": choice.get("finish_reason"), "full_prompt": args.full_prompt,
                  "engine": engine}
        if case_id in labels:
            reply = choice.get("text") or (choice.get("message") or {}).get("content") or ""
            record["correct"] = gsm8k.answer_value(reply) == labels[case_id]
        if engine["finished"] != 1:
            record["error"] = f"{engine['finished']:g} requests finished in this request's window"
        print(json.dumps(record), flush=True)
        correct += bool(record.get("correct"))
        if "error" not in record:
            for key in totals:
                totals[key] += engine[key]
            rows = engine["accepted_by_position"]
            by_position = rows if by_position is None else [a + b for a, b in zip(by_position, rows)]
    drafts = totals["drafts"]
    print(json.dumps({
        "summary": "per-request acceptance", "requests": len(requests), "full_prompt": args.full_prompt,
        "seconds": round(time.time() - began, 1), **totals,
        "gsm8k_correct": [correct, len(labels)] if labels else None,
        "tokens_per_step": round(1 + totals["accepted"] / drafts, 4) if drafts else None,
        "survival_by_position": [round(a / drafts, 4) for a in by_position] if drafts and by_position else None,
    }), flush=True)


if __name__ == "__main__":
    main()
