#!/usr/bin/env python3
"""The stock and steered model names served side by side, live.

usage: dual_probe.py URL [--steered NAME] [--tokens 128] [--speed]

Greedy chat requests (temperature 0, ignore_eos, log probabilities and token IDs), compared
exactly (token IDs and log probabilities):
- models: /v1/models lists the stock name first and the steered name;
- echo: a response names the model it was asked for;
- repeat: the steered replies are the same twice;
- moves: steered replies differ from stock on at least one prompt (the file is applied);
- mixed: stock and steered requests in one batch (every prompt to both names at once) give
  each name's replies alone: one row's steering never reaches another's;
- cache: a long prompt served stock, then steered, then stock without a cache salt gives each
  name its cold reply (steered rows write other KV; neither name reads the other's blocks);
- completions: /v1/completions serves the steered name and names it;
- tokenize: /tokenize accepts the steered name;
- beam: beam search refuses the steered name (HTTP 400).
With --speed, it also times 1 and 16 concurrent streams of distinct prompts on each name and
reads the engine's speculative-decoding counters, so a step's time is compared with draft
acceptance (which follows the text) taken out. Prints one JSON line with a "summary"; exits 1
when a check fails. Stdlib only.
"""
import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

PROMPTS = [
    "Explain how a binary search works, step by step.",
    "Write a short poem about autumn rain in a city.",
    "List five practical tips for learning a new language.",
    "What causes the seasons on Earth?",
    "Describe the plot of a heist film in one paragraph.",
    "Summarize the main ideas of supply and demand.",
    "Write a Python function that reverses a linked list.",
    "Give three arguments for and against remote work.",
]
SPEED_PROMPTS = [f"Write a detailed essay about topic number {i}: " + topic for i, topic in enumerate([
    "the history of bridges", "deep sea creatures", "the printing press", "volcanoes",
    "chess openings", "coffee cultivation", "the Roman roads", "honeybees", "glaciers",
    "jazz improvisation", "the steam engine", "coral reefs", "medieval castles",
    "solar sails", "origami", "desert ecosystems"])]


def request(base: str, path: str, body: dict | None = None, timeout: float = 900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None


def chat(base: str, model: str, prompt: str, tokens: int, salt: str | None = None) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens, "temperature": 0, "ignore_eos": True, "logprobs": True,
            "return_token_ids": True, "chat_template_kwargs": {"thinking": False}}
    if salt is not None:
        body["cache_salt"] = salt
    status, data = request(base, "/v1/chat/completions", body)
    if status != 200:
        return {"status": status}
    choice = data["choices"][0]
    return {"status": status, "model": data["model"], "text": choice["message"]["content"],
            "tokens": choice.get("token_ids"),
            "logprobs": [entry["logprob"] for entry in choice["logprobs"]["content"]]}


def same(a: dict, b: dict) -> bool:
    return a.get("status") == b.get("status") == 200 and a["tokens"] == b["tokens"] \
        and a["logprobs"] == b["logprobs"]


def at_once(base: str, jobs: list[tuple[str, str]], tokens: int, salt: str | None = None) -> list:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        return list(pool.map(lambda job: chat(base, job[0], job[1], tokens, salt), jobs))


def counters(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        text = response.read().decode()
    totals: dict[str, float] = {}
    for name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                 "vllm:generation_tokens_total"):
        totals[name] = sum(float(m.group(1)) for m in
                           re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M))
    return totals


def speed(base: str, model: str, streams: int, tokens: int, samples: int) -> dict:
    """Decode at ``streams`` concurrent distinct prompts: tokens a second, and the time of
    an engine step (a request's drafts are its steps; at a fixed concurrency every request
    is in every step)."""
    rates, step_ms = [], []
    for sample in range(samples):
        prompts = [f"{SPEED_PROMPTS[(i + sample) % len(SPEED_PROMPTS)]} (variant {sample})"
                   for i in range(streams)]
        before = counters(base)
        start = time.perf_counter()
        replies = at_once(base, [(model, p) for p in prompts], tokens, salt=uuid.uuid4().hex)
        elapsed = time.perf_counter() - start
        after = counters(base)
        if any(r.get("status") != 200 for r in replies):
            return {"error": "a request failed"}
        produced = sum(len(r["tokens"]) for r in replies)
        drafts = after["vllm:spec_decode_num_drafts_total"] - before["vllm:spec_decode_num_drafts_total"]
        rates.append(produced / elapsed)
        if drafts:
            step_ms.append(1000 * elapsed / (drafts / streams))
    return {"streams": streams, "tok_s": rates, "step_ms": step_ms,
            "tok_s_mean": sum(rates) / len(rates),
            "step_ms_mean": sum(step_ms) / len(step_ms) if step_ms else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--steered", default="deepseek-v4.1-flash-weightless")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--speed-tokens", type=int, default=256)
    parser.add_argument("--speed-samples", type=int, default=3)
    args = parser.parse_args()
    base, steered, tokens = args.url.rstrip("/"), args.steered, args.tokens
    checks: dict[str, bool] = {}
    detail: dict = {}

    _, models = request(base, "/v1/models")
    names = [entry["id"] for entry in (models or {}).get("data", [])]
    stock = names[0] if names else None
    checks["models"] = len(names) == 2 and names[1] == steered
    detail["models"] = names
    if not checks["models"]:
        print(json.dumps({"summary": {"pass": False, "checks": checks, "detail": detail}}))
        return 1

    def alone(model: str) -> list[dict]:
        return [chat(base, model, p, tokens, salt=uuid.uuid4().hex) for p in PROMPTS]

    stock_alone, steered_alone = alone(stock), alone(steered)
    steered_again = alone(steered)
    for name, replies in (("stock", stock_alone), ("steered", steered_alone)):
        for i, reply in enumerate(replies):
            print(json.dumps({"phase": "alone", "model": name, "prompt": i, **reply}), flush=True)
    checks["echo"] = all(r.get("model") == stock for r in stock_alone) and \
        all(r.get("model") == steered for r in steered_alone)
    checks["repeat"] = all(same(a, b) for a, b in zip(steered_alone, steered_again))
    moved = [i for i, (a, b) in enumerate(zip(stock_alone, steered_alone)) if a["tokens"] != b["tokens"]]
    first_diff = [next((j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
                  for a, b in zip(stock_alone, steered_alone)]
    checks["moves"] = bool(moved)
    detail["moved_prompts"], detail["first_difference"] = moved, first_diff

    mixed = at_once(base, [(m, p) for p in PROMPTS for m in (stock, steered)], tokens,
                    salt=uuid.uuid4().hex)
    mixed_stock, mixed_steered = mixed[0::2], mixed[1::2]
    checks["mixed"] = all(same(a, b) for a, b in zip(mixed_stock, stock_alone)) and \
        all(same(a, b) for a, b in zip(mixed_steered, steered_alone))
    detail["mixed_differs"] = {
        "stock": [i for i, (a, b) in enumerate(zip(mixed_stock, stock_alone)) if not same(a, b)],
        "steered": [i for i, (a, b) in enumerate(zip(mixed_steered, steered_alone)) if not same(a, b)]}

    long_prompt = ("Read this log and say which service failed first.\n" + "\n".join(
        f"2026-10-28T03:{i // 60:02d}:{i % 60:02d}Z service-{i % 7} status ok latency {(i * 37) % 900}ms"
        for i in range(400)) + f"\nRun {uuid.uuid4().hex}.")
    steered_cold = chat(base, steered, long_prompt, tokens, salt=uuid.uuid4().hex)
    stock_cold = chat(base, stock, long_prompt, tokens, salt=uuid.uuid4().hex)
    stock_warm = chat(base, stock, long_prompt, tokens)
    steered_after_stock = chat(base, steered, long_prompt, tokens)
    stock_after_steered = chat(base, stock, long_prompt, tokens)
    checks["cache"] = same(stock_warm, stock_cold) and same(steered_after_stock, steered_cold) \
        and same(stock_after_steered, stock_cold)
    detail["cache"] = {"stock_warm": same(stock_warm, stock_cold),
                       "steered_after_stock": same(steered_after_stock, steered_cold),
                       "stock_after_steered": same(stock_after_steered, stock_cold),
                       "steered_differs_from_stock": steered_cold.get("tokens") != stock_cold.get("tokens")}

    status, data = request(base, "/v1/completions", {"model": steered, "prompt": "The capital of France is",
                                                       "max_tokens": 8, "temperature": 0})
    checks["completions"] = status == 200 and data["model"] == steered
    status, _ = request(base, "/tokenize", {"model": steered, "prompt": "hello"})
    checks["tokenize"] = status == 200
    status, _ = request(base, "/v1/chat/completions", {
        "model": steered, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
        "use_beam_search": True, "n": 2})
    checks["beam"] = status == 400
    detail["beam_status"] = status

    if args.speed:
        detail["speed"] = {}
        for streams in (1, 16):
            for name, model in (("stock", stock), ("steered", steered)):
                result = speed(base, model, streams, args.speed_tokens, args.speed_samples)
                detail["speed"][f"{name}-c{streams}"] = result
                print(json.dumps({"phase": "speed", "model": name, **result}), flush=True)
        for streams in (1, 16):
            a, b = detail["speed"][f"stock-c{streams}"], detail["speed"][f"steered-c{streams}"]
            if a.get("step_ms_mean") and b.get("step_ms_mean"):
                detail["speed"][f"step_ratio-c{streams}"] = b["step_ms_mean"] / a["step_ms_mean"]

    passed = all(checks.values())
    print(json.dumps({"summary": {"pass": passed, "checks": checks, "detail": detail}}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
