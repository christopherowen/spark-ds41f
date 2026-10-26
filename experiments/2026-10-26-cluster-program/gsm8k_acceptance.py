#!/usr/bin/env python3
"""Draft acceptance on GSM8K few-shot: the external acceptance figures' workload on our stack.

usage: gsm8k_acceptance.py BASE_URL [--data DIR] [--questions N] [--shots K]
                           [--max-tokens T] [--concurrency C]

vLLM's GSM8K evaluator's prompts (tests/evals/gsm8k/gsm8k_eval.py): the first K training
examples as "Question: ...\\nAnswer: ..." shots, then each of the first N test questions,
sent raw to /v1/completions (no chat template, so no reasoning block) at temperature 0
with its stop strings, C requests in flight. The server's speculative-decoding counters
are read before and after:

- tokens a step: 1 + accepted / drafts, a request's output tokens per verification step
  (SGLang's "accept length", generation tokens / verify calls);
- by position k: accepted and verified drafts. With every draft verified (adaptive
  verification off), accepted_k / drafts is the drafter's survival P(a > k) on this text;
  with adaptive verification on, accepted_k / verified_k is the acceptance of the rows it
  chose to verify. accepted_k / accepted_(k-1) is position k's hit rate given k - 1.

Requests that finished beyond the questions ("foreign_requests", someone else's load
reaching the arm) make the counters not this workload's: the script exits 3 after printing.
Accuracy uses the evaluator's rule (the last integer in the answer) and doubles as a
parity check. One JSON line a question (index, output tokens, finish reason, correct),
then the summary line. The data directory holds GSM8K's train.jsonl and test.jsonl
(the evaluator's own download; default ~/spark-lab/inputs/gsm8k).
"""
import argparse
import ast
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench import prom  # noqa: E402

INVALID = -9999999
STOP = ["Question", "Assistant:", "<|separator|>"]
ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
VERIFIED_POS = "vllm:spec_decode_num_draft_tokens_per_pos_total"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line and not line.startswith("#")]


def answer_value(text: str) -> int:
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def prompts(data: Path, questions: int, shots: int) -> tuple[list[str], list[int]]:
    train, test = read_jsonl(data / "train.jsonl"), read_jsonl(data / "test.jsonl")
    prefix = "".join(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n" for row in train[:shots])
    rows = test[:questions]
    return ([prefix + f"Question: {row['question']}\nAnswer:" for row in rows],
            [answer_value(row["answer"]) for row in rows])


def counters(base: str) -> dict:
    text = prom.fetch(base, timeout=60)
    if text is None:
        raise SystemExit(f"{base}/metrics unreachable")
    totals = prom.parse(text)
    return {"drafts": totals.get(prom.VLLM["drafts"], 0.0),
            "verified": totals.get(prom.VLLM["draft_tokens"], 0.0),
            "accepted": totals.get(prom.VLLM["accepted_tokens"], 0.0),
            "steps": totals.get(prom.VLLM["steps"], 0.0),
            "generation_tokens": totals.get(prom.VLLM["generation_tokens"], 0.0),
            "finished": totals.get(prom.VLLM["requests_finished"], 0.0),
            "accepted_pos": prom.by_label(text, ACCEPTED_POS, "position"),
            "verified_pos": prom.by_label(text, VERIFIED_POS, "position")}


def model_id(base: str) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0]["id"]


def complete(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stop": STOP}
    request = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--data", type=Path, default=Path.home() / "spark-lab" / "inputs" / "gsm8k")
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    model = model_id(base)
    texts, labels = prompts(args.data, args.questions, args.shots)
    prom.require_idle(base, "the acceptance window")
    before, began = counters(base), time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        replies = list(pool.map(lambda text: complete(base, model, text, args.max_tokens), texts))
    seconds, after = time.time() - began, counters(base)
    correct = 0
    for index, (reply, label) in enumerate(zip(replies, labels)):
        choice = reply["choices"][0]
        right = answer_value(choice["text"]) == label
        correct += right
        print(json.dumps({"question": index, "tokens": reply["usage"]["completion_tokens"],
                          "finish": choice.get("finish_reason"), "correct": right}), flush=True)
    delta = {key: after[key] - before[key] for key in ("drafts", "verified", "accepted", "steps",
                                                         "generation_tokens", "finished")}
    foreign = int(delta["finished"]) - len(labels)
    positions = sorted(set(after["accepted_pos"]) | set(after["verified_pos"]), key=int)
    accepted_pos = [after["accepted_pos"].get(p, 0.0) - before["accepted_pos"].get(p, 0.0) for p in positions]
    verified_pos = [after["verified_pos"].get(p, 0.0) - before["verified_pos"].get(p, 0.0) for p in positions]
    drafts = delta["drafts"]
    if not drafts:
        raise SystemExit("no drafts in the window")
    print(json.dumps({
        "summary": "gsm8k acceptance", "questions": len(labels), "shots": args.shots,
        "max_tokens": args.max_tokens, "concurrency": args.concurrency,
        "correct": correct, "accuracy": round(correct / len(labels), 4),
        "output_tokens": sum(r["usage"]["completion_tokens"] for r in replies),
        "seconds": round(seconds, 1), "foreign_requests": foreign,
        "drafts": drafts, "verified": delta["verified"], "accepted": delta["accepted"],
        "tokens_per_step": round(1 + delta["accepted"] / drafts, 4),
        "verified_per_draft": round(delta["verified"] / drafts, 4),
        "accepted_per_verified": round(delta["accepted"] / delta["verified"], 4) if delta["verified"] else None,
        "engine_steps": delta["steps"], "output_tokens_per_engine_step":
            round(delta["generation_tokens"] / delta["steps"], 4) if delta["steps"] else None,
        "accepted_by_position": accepted_pos, "verified_by_position": verified_pos,
        "survival_by_position": [round(a / drafts, 4) for a in accepted_pos],
        "verified_share_by_position": [round(v / drafts, 4) for v in verified_pos],
        "accepted_per_verified_by_position": [round(a / v, 4) if v else None
                                              for a, v in zip(accepted_pos, verified_pos)],
        # P(position k accepted | k - 1 accepted): the drafter's per-position hit rate.
        "step_acceptance_by_position": [round(a / previous, 4) if previous else None
                                        for a, previous in zip(accepted_pos, [drafts, *accepted_pos])],
    }), flush=True)
    if foreign:
        raise SystemExit(3)  # someone else's requests shared the window: the counters are not ours


if __name__ == "__main__":
    main()
