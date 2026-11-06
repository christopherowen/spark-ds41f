#!/usr/bin/env python3
"""Opt-in serving gate: native multi-turn prompts, exact outputs, reuse and TTFT.

Run only inside an authorized lab window; lab.py supplies node memory/thermal
telemetry. Each JSONL receipt is retained, including failures and cancellation.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from window_lease import require_window

WINDOW_STATE = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def post(url, body, request_id=None):
    require_window(WINDOW_STATE)
    headers = {"Content-Type": "application/json"}
    if request_id is not None:
        headers["X-Request-Id"] = request_id
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    return urllib.request.urlopen(request, timeout=900)


def tokenize(url, model, messages, tools):
    with post(url + "/tokenize", {"model": model, "messages": messages, "tools": tools}) as r:
        return json.load(r)["tokens"]


def request(url, model, prompt, salt, label, cancel=False):
    started = time.monotonic()
    receipt = {"label": label, "prompt_tokens": len(prompt), "prompt_sha256": digest(prompt),
               "cache_salt": salt, "client_request_id": f"prefix-{salt}-{label}", "token_ids": [], "logprobs": [], "ttft_s": None}
    try:
        with post(url + "/v1/completions", {
            "model": model, "prompt": prompt, "cache_salt": salt,
            "temperature": 0, "max_tokens": 64, "ignore_eos": True,
            "logprobs": 1, "return_token_ids": True, "stream": True,
            "stream_options": {"include_usage": True},
        }, request_id=receipt["client_request_id"]) as response:
            receipt["response_request_id"] = getattr(response, "headers", {}).get("X-Request-Id")
            if cancel:
                time.sleep(0.2)
                receipt["cancelled_by_client"] = True
            else:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    if data.get("id"):
                        receipt["response_id"] = data["id"]
                    if data.get("usage"):
                        receipt["usage"] = data["usage"]
                    for choice in data.get("choices", []):
                        ids = choice.get("token_ids") or []
                        if receipt["ttft_s"] is None and (ids or choice.get("text")):
                            receipt["ttft_s"] = time.monotonic() - started
                        receipt["token_ids"].extend(ids)
                        receipt["logprobs"].extend(
                            (choice.get("logprobs") or {}).get("token_logprobs") or []
                        )
        if not cancel:
            usage = receipt["usage"]
            receipt["cached_tokens"] = usage["prompt_tokens_details"]["cached_tokens"]
            receipt["uncached_tokens"] = usage["prompt_tokens"] - receipt["cached_tokens"]
            assert usage["prompt_tokens"] == len(prompt)
            assert 0 <= receipt["cached_tokens"] <= len(prompt)
            assert len(receipt["token_ids"]) == len(receipt["logprobs"]) == 64
            assert receipt["ttft_s"] is not None
    except Exception as error:
        receipt["error"] = repr(error)
    receipt["wall_s"] = time.monotonic() - started
    if not cancel and "error" not in receipt:
        receipt["output_tokens_per_s"] = len(receipt["token_ids"]) / receipt["wall_s"]
    return receipt


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    global WINDOW_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint-policy", choices=["candidate", "control"], default="candidate")
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--window-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-prompt-tokens", type=int, default=65536)
    parser.add_argument("--pressure-prompts", type=int, default=32)
    args = parser.parse_args()
    WINDOW_STATE = args.window_state
    require_window(WINDOW_STATE)
    args.url = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=False)
    stream = (args.out / "requests.jsonl").open("x")
    failures = []

    def save(receipt):
        stream.write(json.dumps(receipt) + "\n")
        stream.flush()
        if "error" in receipt:
            failures.append(receipt["label"] + ": " + receipt["error"])
        elif "cold" in receipt["label"] and receipt.get("cached_tokens") != 0:
            failures.append(receipt["label"] + " was not cold")
        return receipt

    def compare(a, b):
        if "error" in a or "error" in b or any(a[k] != b[k] for k in ("token_ids", "logprobs")):
            failures.append(a["label"] + " differs from " + b["label"])

    paths = ["spark_ds41f/doctor.py", "scripts/recipe.py", "spark_ds41f/build.py"]
    source = "\n".join(f"File: {p}\n{(args.repo / p).read_text()}" for p in paths)
    tools = [{"type": "function", "function": {"name": "read_files", "description": "Read repository files",
              "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}}}}]
    messages = [
        {"role": "system", "content": "Review repository correctness. Base findings on the supplied files and tests."},
        {"role": "user", "content": "Review source identity validation and cancellation cleanup. Read the implementation first."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "read_1", "type": "function", "function": {
            "name": "read_files", "arguments": json.dumps({"paths": paths})}}]},
        {"role": "tool", "tool_call_id": "read_1", "content": source},
    ]
    ids = tokenize(args.url, args.model, messages, tools)
    while len(ids) < args.minimum_prompt_tokens:
        messages[-1]["content"] += "\nAdditional source listing:\n" + source
        ids = tokenize(args.url, args.model, messages, tools)
    # Keep the native chat encoder and a complete transcript. Position the
    # prompt 64 tokens beyond the page whose endpoint is under test.
    for _ in range(8):
        padding = (320 - len(ids)) % 512
        if not padding:
            break
        messages[-1]["content"] += " item" * padding
        ids = tokenize(args.url, args.model, messages, tools)
    assert len(ids) % 512 == 320, "could not obtain the native-encoded boundary fixture"
    transcripts = [list(messages)]
    for turn in range(1, 4):
        messages = messages + [
            {"role": "assistant", "content": "The next check is the test output and whether cancelled requests release ownership.",
             "tool_calls": [{"id": f"read_{turn + 1}", "type": "function", "function": {
                 "name": "read_files", "arguments": json.dumps({"paths": ["tests/output.txt"]})}}]},
            {"role": "tool", "tool_call_id": f"read_{turn + 1}",
             "content": ("test_cancelled_request_releases_blocks PASSED\n"
                         "test_changed_prefix_cannot_reuse_suffix PASSED\n") * 64},
        ]
        transcripts.append(messages)
    prompts = [tokenize(args.url, args.model, m, tools) for m in transcripts]
    (args.out / "workload.json").write_text(json.dumps({"tools": tools, "transcripts": transcripts,
        "prompt_hashes": [digest(p) for p in prompts], "lengths": list(map(len, prompts)),
        "common_prefixes": [common_prefix(a, b) for a, b in zip(prompts, prompts[1:])]}, indent=2))
    nonce = str(time.time_ns())
    cold, warm = [], []
    for turn, prompt in enumerate(prompts):
        cold.append(save(request(args.url, args.model, prompt, f"{nonce}-cold-{turn}", f"cold-{turn}")))
        warm.append(save(request(args.url, args.model, prompt, nonce + "-warm", f"warm-{turn}")))
        compare(cold[-1], warm[-1])
        if turn == 0:
            resend = save(request(args.url, args.model, prompt, nonce + "-warm", "resend-first"))
            compare(cold[0], resend)
            if resend.get("cached_tokens") != (len(prompt) - 128) // 256 * 256:
                failures.append("identical resend did not preserve the CED replay suffix")
    expected_endpoint = min(common_prefix(prompts[0], prompts[1]), len(prompts[1]) - 128) // 256 * 256
    if args.endpoint_policy == "control":
        expected_endpoint = min(expected_endpoint, (len(prompts[0]) - 128) // 256 * 256)
    if warm[1].get("cached_tokens") != expected_endpoint:
        failures.append(f"extension reused {warm[1].get('cached_tokens')} tokens; expected {expected_endpoint}")
    sibling_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(request, args.url, args.model, prompt, nonce + "-warm", f"concurrent-{i}")
                for i, prompt in enumerate(prompts[1:])]
        for i, job in enumerate(jobs, 1):
            compare(cold[i], save(job.result()))
    sibling_wall = time.monotonic() - sibling_started
    agent_workloads = []
    agent_cold = []
    agent_prompts = []
    topics = ["cancellation ownership", "source identity and build locking", "memory admission and eviction"]
    for i, topic in enumerate(topics):
        instruction = {"role": "user", "content": f"Review {topic}. Read the supplied implementation and test results."}
        initial = [dict(m) for m in transcripts[0]]
        final = [dict(m) for m in transcripts[-1]]
        initial[1] = final[1] = instruction
        base_ids = tokenize(args.url, args.model, initial, tools)
        final_ids = tokenize(args.url, args.model, final, tools)
        agent_workloads.append({"topic": topic, "initial": initial, "final": final,
                                "initial_hash": digest(base_ids), "final_hash": digest(final_ids),
                                "initial_tokens": len(base_ids), "final_tokens": len(final_ids)})
        agent_prompts.append(final_ids)
        agent_cold.append(save(request(args.url, args.model, final_ids, f"{nonce}-agent-{i}-cold", f"agent-cold-{i}")))
        save(request(args.url, args.model, base_ids, f"{nonce}-agent-{i}", f"agent-prime-{i}"))
    (args.out / "distinct-agent-workloads.json").write_text(json.dumps(agent_workloads, indent=2))
    agent_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(request, args.url, args.model, prompt, f"{nonce}-agent-{i}", f"agent-concurrent-{i}")
                for i, prompt in enumerate(agent_prompts)]
        for i, job in enumerate(jobs):
            compare(agent_cold[i], save(job.result()))
    agent_wall = time.monotonic() - agent_started
    changed = [dict(m) for m in transcripts[-1]]
    changed[0] = {"role": "system", "content": "Use a concise review format. Verify ownership using the supplied implementation."}
    changed_ids = tokenize(args.url, args.model, changed, tools)
    changed_result = save(request(args.url, args.model, changed_ids, nonce + "-warm", "changed-prefix"))
    if changed_result.get("cached_tokens") != 0:
        failures.append("changed prefix reused cached tokens")
    compare(changed_result, save(request(args.url, args.model, changed_ids, nonce + "-changed-cold", "changed-prefix-cold")))
    save(request(args.url, args.model, prompts[-1], nonce + "-cancel", "cancel-prefill", cancel=True))
    compare(cold[-1], save(request(args.url, args.model, prompts[-1], nonce + "-cancel", "after-cancel")))
    compare(cold[1], save(request(args.url, args.model, prompts[1], nonce + "-warm", "before-pressure-prime")))
    pressure_baseline = save(request(args.url, args.model, prompts[1], nonce + "-warm", "pressure-baseline"))
    compare(cold[1], pressure_baseline)
    if pressure_baseline.get("cached_tokens", 0) <= 0:
        failures.append("pressure baseline has no reusable state")
    for i in range(args.pressure_prompts):
        save(request(args.url, args.model, prompts[0], f"{nonce}-pressure-{i}", f"pressure-{i}"))
    after_pressure = save(request(args.url, args.model, prompts[1], nonce + "-warm", "after-pressure"))
    compare(cold[1], after_pressure)
    pressure_reduced_reuse = after_pressure.get("cached_tokens", 0) < pressure_baseline.get("cached_tokens", 0)
    if not pressure_reduced_reuse:
        failures.append("pressure did not reduce reuse; increase pressure before accepting this gate")
    summary = {"failures": failures, "passed": not failures, "endpoint_policy": args.endpoint_policy,
               "pressure_reduced_reuse": pressure_reduced_reuse,
               "sibling_batch_wall_s": sibling_wall,
               "distinct_agent_batch_wall_s": agent_wall,
               "distinct_agent_output_tokens_per_s": 192 / agent_wall if not failures else None,
               "note": "TTFT is incremental request latency; cache hits are not cold prefill throughput. Compare independent candidate boots."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
