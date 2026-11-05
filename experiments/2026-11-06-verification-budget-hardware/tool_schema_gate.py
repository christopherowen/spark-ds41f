#!/usr/bin/env python3
"""Real HTTP tool/schema correctness and seeded batch invariance; no timing claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.error
import urllib.request
import uuid


def post(base, body):
    request = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def normalize(reply, mode, expected):
    choice = reply["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("output exhausted its token allowance")
    message = choice["message"]
    if mode == "tool":
        calls = message.get("tool_calls") or []
        if len(calls) != 1 or calls[0]["function"]["name"] != "record":
            raise ValueError("forced tool call is missing or wrong")
        value = json.loads(calls[0]["function"]["arguments"])
    else:
        value = json.loads(message["content"])
    if value != expected:
        raise ValueError(f"{mode}: output did not satisfy the exact requested record")
    tokens = choice.get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
        raise ValueError("native output token IDs are missing")
    return {"value": value, "token_ids": tokens, "finish_reason": choice["finish_reason"]}


def run(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    signatures = {}
    for mode in ("schema", "tool"):
        for temperature in (0.0, 1.0):
            def request(index):
                expected = {"label": f"record-{index}", "value": index + 7}
                schema = {"type": "object", "properties": {
                    "label": {"type": "string", "enum": [expected["label"]]},
                    "value": {"type": "integer", "enum": [expected["value"]]}},
                    "required": ["label", "value"], "additionalProperties": False}
                body = {"model": model, "messages": [{"role": "user", "content":
                        "Record exactly these fields: " + json.dumps(expected)}],
                        "temperature": temperature, "seed": 20261106 + index,
                        "max_tokens": 96, "return_token_ids": True,
                        "cache_salt": uuid.uuid4().hex,
                        "chat_template_kwargs": {"thinking": False}}
                if mode == "schema":
                    body["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": "record", "strict": True, "schema": schema}}
                else:
                    body["tools"] = [{"type": "function", "function": {
                        "name": "record", "description": "Record the requested fields exactly.",
                        "strict": True, "parameters": schema}}]
                    body["tool_choice"] = {"type": "function", "function": {"name": "record"}}
                return normalize(post(base, body), mode, expected)
            reference = None
            for concurrency in (1, 8, 16):
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(request, range(16)))
                if reference is not None and results != reference:
                    raise ValueError(f"{mode}/{temperature}: output changed at c{concurrency}")
                reference = results
                key = f"{mode}/t{temperature}/c{concurrency}"
                signatures[key] = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
                print(json.dumps({"case": key, "requests": 16, "sha256": signatures[key]}), flush=True)
    # A rejected request must not poison the next ordinary request.
    bad = {"model": model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": -1}
    try:
        post(base, bad)
    except urllib.error.HTTPError as error:
        if error.code not in (400, 422):
            raise
        signatures["invalid_request_status"] = error.code
    else:
        raise ValueError("negative token allowance was accepted")
    good = post(base, {**bad, "max_tokens": 1, "temperature": 0,
                       "chat_template_kwargs": {"thinking": False}})
    if not good.get("choices"):
        raise ValueError("ordinary request failed after invalid request")
    return signatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    args = parser.parse_args()
    signatures = run(args.base_url.rstrip("/"))
    print(json.dumps({"summary": "tool-schema gate", "passed": True, "signatures": signatures}), flush=True)
