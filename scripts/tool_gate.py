#!/usr/bin/env python3
"""Auto tool-calling and finish_reason qualification gate for stock and GLP-steered models.

Exercises both stock and steered served names with explicit sampling temperature
(never inheriting ambiguous server defaults) across:
  - single and parallel tool_choice="auto" calls
  - tool selection from a 10-tool catalog
  - schema-typed arguments (integers, booleans, enums, nested objects, arrays)
  - agent-style shell/file calls
  - multi-turn tool continuation (tool result -> final answer)
  - tool_choice="required", "none", and named function modes
  - plain-text finish_reason="stop" controls (with and without tools attached)

Fails (exit 1) on derailment signatures:
  - missing tool call where one is required (vanish / finish_reason=length / empty stop)
  - plain-text or final-turn finish_reason != "stop"
  - malformed JSON arguments or empty arguments when required keys exist
  - exact duplicate tool calls in one response
  - raw DSML markup leaked into message content or tool arguments

Exit codes: 0 = gate passed, 1 = derailment detected, 2 = endpoint/request error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

DSML_LEAK_RE = re.compile(
    r"<\s*/?\s*[｜|]\s*DSML\s*[｜|]|<\s*/?\s*function_calls\b", re.IGNORECASE
)

S = {"type": "string"}
N = {"type": "number"}
I = {"type": "integer"}
B = {"type": "boolean"}


def _fn(
    name: str,
    desc: str,
    props: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": _fn("get_weather", "Get current weather for a city", {"city": S}, ["city"]),
    "get_time": _fn("get_time", "Get current local time for a city", {"city": S}, ["city"]),
    "add": _fn("add", "Add two numbers", {"a": N, "b": N}, ["a", "b"]),
    "multiply": _fn("multiply", "Multiply two numbers", {"a": N, "b": N}, ["a", "b"]),
    "convert_units": _fn(
        "convert_units",
        "Convert a value between units",
        {"value": N, "from_unit": S, "to_unit": S},
        ["value", "from_unit", "to_unit"],
    ),
    "get_stock_price": _fn(
        "get_stock_price", "Get latest stock price for a ticker", {"ticker": S}, ["ticker"]
    ),
    "send_email": _fn(
        "send_email",
        "Send an email",
        {"to": S, "subject": S, "body": S},
        ["to", "subject", "body"],
    ),
    "create_event": _fn(
        "create_event",
        "Create a calendar event",
        {"title": S, "date": S, "time": S},
        ["title", "date"],
    ),
    "flight_status": _fn(
        "flight_status", "Check status of a flight", {"flight_number": S}, ["flight_number"]
    ),
    "book_table": _fn(
        "book_table",
        "Book a restaurant table",
        {"restaurant": S, "party_size": I, "date": S},
        ["restaurant", "party_size", "date"],
    ),
    "create_ticket": _fn(
        "create_ticket",
        "Create an issue ticket with structured metadata",
        {
            "title": S,
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "urgent_page": B,
            "tags": {"type": "array", "items": S},
            "owner": {
                "type": "object",
                "properties": {"team": S, "oncall_id": I},
                "required": ["team", "oncall_id"],
            },
        },
        ["title", "priority", "urgent_page", "tags", "owner"],
    ),
    "set_limits": _fn(
        "set_limits",
        "Configure numeric limits for a service",
        {
            "service": S,
            "max_concurrency": I,
            "strict_mode": B,
            "regions": {"type": "array", "items": S},
        },
        ["service", "max_concurrency", "strict_mode", "regions"],
    ),
    "terminal": _fn(
        "terminal", "Run a shell command", {"command": S}, ["command"]
    ),
    "read_file": _fn(
        "read_file", "Read contents of a file path", {"path": S}, ["path"]
    ),
}


def _tools(*names: str) -> list[dict[str, Any]]:
    return [TOOLS[n] for n in names]


CATALOG_10 = _tools(
    "get_weather",
    "get_time",
    "add",
    "multiply",
    "convert_units",
    "get_stock_price",
    "send_email",
    "create_event",
    "flight_status",
    "book_table",
)


def build_scenarios() -> list[dict[str, Any]]:
    return [
        # Single auto tool calls
        {
            "id": "auto_single_weather",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "What is the current weather in Tokyo?"}],
            "tools": _tools("get_weather", "get_time"),
            "expect_names": ["get_weather"],
            "required_keys": {"get_weather": ["city"]},
        },
        {
            "id": "auto_single_convert",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Convert 42 kilometers to miles using the converter tool."}],
            "tools": _tools("convert_units", "add"),
            "expect_names": ["convert_units"],
            "required_keys": {"convert_units": ["value", "from_unit", "to_unit"]},
        },
        {
            "id": "auto_single_stock",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Look up the latest stock price for NVDA."}],
            "tools": _tools("get_stock_price", "get_weather"),
            "expect_names": ["get_stock_price"],
            "required_keys": {"get_stock_price": ["ticker"]},
        },
        {
            "id": "auto_single_email",
            "kind": "tool_call",
            "messages": [{
                "role": "user",
                "content": "Send an email to ops@example.com with subject 'Deploy Complete' and body 'Cluster is green.'",
            }],
            "tools": _tools("send_email", "create_event"),
            "expect_names": ["send_email"],
            "required_keys": {"send_email": ["to", "subject", "body"]},
        },
        # Parallel auto tool calls
        {
            "id": "auto_parallel_weather_time",
            "kind": "tool_call",
            "messages": [{
                "role": "user",
                "content": "Check both the current weather and the current local time in Zurich.",
            }],
            "tools": _tools("get_weather", "get_time"),
            "min_calls": 2,
            "expect_names": ["get_weather", "get_time"],
            "required_keys": {"get_weather": ["city"], "get_time": ["city"]},
        },
        {
            "id": "auto_parallel_math",
            "kind": "tool_call",
            "messages": [{
                "role": "user",
                "content": "Use the tools to compute 17 + 25 and also 9 * 12 in parallel.",
            }],
            "tools": _tools("add", "multiply"),
            "min_calls": 2,
            "expect_names": ["add", "multiply"],
            "required_keys": {"add": ["a", "b"], "multiply": ["a", "b"]},
        },
        # Selection from 10-tool catalog
        {
            "id": "auto_catalog10_flight",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Check the status of flight LX188."}],
            "tools": CATALOG_10,
            "expect_names": ["flight_status"],
            "required_keys": {"flight_status": ["flight_number"]},
        },
        {
            "id": "auto_catalog10_event",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Schedule a calendar event titled 'Kernel Review' on 2026-12-01 at 14:00."}],
            "tools": CATALOG_10,
            "expect_names": ["create_event"],
            "required_keys": {"create_event": ["title", "date"]},
        },
        # Schema-typed nested/array/enum/bool/int calls
        {
            "id": "auto_schema_nested_ticket",
            "kind": "tool_call",
            "messages": [{
                "role": "user",
                "content": (
                    "Create a ticket titled 'RoCE link flap' with priority 'high', urgent_page true, "
                    "tags ['network', 'roce'], and owner team 'infra' with oncall_id 42."
                ),
            }],
            "tools": _tools("create_ticket", "set_limits"),
            "expect_names": ["create_ticket"],
            "required_keys": {"create_ticket": ["title", "priority", "urgent_page", "tags", "owner"]},
            "type_checks": {
                "create_ticket": {
                    "priority": ("enum", ("low", "medium", "high", "urgent")),
                    "urgent_page": ("bool", None),
                    "tags": ("list", None),
                    "owner": ("dict", ["team", "oncall_id"]),
                }
            },
        },
        {
            "id": "auto_schema_limits",
            "kind": "tool_call",
            "messages": [{
                "role": "user",
                "content": (
                    "Configure limits for service 'gateway': max_concurrency 32, strict_mode false, "
                    "regions ['us-east', 'eu-central']."
                ),
            }],
            "tools": _tools("set_limits", "create_ticket"),
            "expect_names": ["set_limits"],
            "required_keys": {"set_limits": ["service", "max_concurrency", "strict_mode", "regions"]},
            "type_checks": {
                "set_limits": {
                    "max_concurrency": ("int", None),
                    "strict_mode": ("bool", None),
                    "regions": ("list", None),
                }
            },
        },
        {
            "id": "auto_schema_book_table",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Book a table at 'Kronenhalle' for 4 guests on 2026-12-05."}],
            "tools": _tools("book_table", "create_event"),
            "expect_names": ["book_table"],
            "required_keys": {"book_table": ["restaurant", "party_size", "date"]},
            "type_checks": {"book_table": {"party_size": ("int", None)}},
        },
        # Agent-style calls
        {
            "id": "auto_agent_terminal",
            "kind": "tool_call",
            "messages": [
                {"role": "system", "content": "You are a coding agent. Use tools for shell and file operations."},
                {"role": "user", "content": "Run `git status --short` in the terminal."},
            ],
            "tools": _tools("terminal", "read_file"),
            "expect_names": ["terminal"],
            "required_keys": {"terminal": ["command"]},
        },
        {
            "id": "auto_agent_read_file",
            "kind": "tool_call",
            "messages": [
                {"role": "system", "content": "You are a coding agent. Use tools for shell and file operations."},
                {"role": "user", "content": "Read the file `/etc/hosts`."},
            ],
            "tools": _tools("terminal", "read_file"),
            "expect_names": ["read_file"],
            "required_keys": {"read_file": ["path"]},
        },
        # Multi-turn continuation
        {
            "id": "multiturn_weather_chain",
            "kind": "multiturn",
            "messages": [{"role": "user", "content": "What is the weather in Geneva? Reply in one short sentence once you know."}],
            "tools": _tools("get_weather"),
            "expect_names": ["get_weather"],
            "required_keys": {"get_weather": ["city"]},
            "canned_result": json.dumps({"city": "Geneva", "temp_c": 11, "condition": "clear"}),
            "expect_final_substr": "11",
        },
        {
            "id": "multiturn_math_chain",
            "kind": "multiturn",
            "messages": [{"role": "user", "content": "Use the multiply tool to calculate 23 * 19, then state the result."}],
            "tools": _tools("multiply"),
            "expect_names": ["multiply"],
            "required_keys": {"multiply": ["a", "b"]},
            "canned_result": json.dumps({"result": 437}),
            "expect_final_substr": "437",
        },
        # tool_choice modes
        {
            "id": "mode_required",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Weather in Oslo please."}],
            "tools": _tools("get_weather", "get_time"),
            "tool_choice": "required",
            "expect_names": ["get_weather"],
            "required_keys": {"get_weather": ["city"]},
        },
        {
            "id": "mode_named",
            "kind": "tool_call",
            "messages": [{"role": "user", "content": "Tell me about Bern."}],
            "tools": _tools("get_weather", "get_time"),
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
            "expect_names": ["get_time"],
            "required_keys": {"get_time": ["city"]},
        },
        {
            "id": "mode_none_no_call",
            "kind": "no_call",
            "messages": [{"role": "user", "content": "Explain in one sentence what a TCP checksum does."}],
            "tools": _tools("get_weather", "terminal"),
            "tool_choice": "none",
        },
        # Plain-text finish_reason="stop" controls
        {
            "id": "plain_no_tools_stop",
            "kind": "plain",
            "messages": [{"role": "user", "content": "Compute 17 * 23 + 19 and reply with the integer only."}],
            "expect_substr": "410",
        },
        {
            "id": "plain_with_tools_no_call_stop",
            "kind": "no_call",
            "messages": [{"role": "user", "content": "What is the capital of France? Answer directly without calling any tool."}],
            "tools": _tools("get_weather", "get_stock_price"),
            "expect_substr": "Paris",
        },
    ]


def _check_types(args: dict[str, Any], checks: dict[str, tuple[str, Any]]) -> list[str]:
    misses: list[str] = []
    for key, (kind, extra) in checks.items():
        if key not in args:
            continue
        val = args[key]
        if kind == "int" and (isinstance(val, bool) or not isinstance(val, int)):
            misses.append(f"arg {key} expected int, got {type(val).__name__}")
        elif kind == "bool" and not isinstance(val, bool):
            misses.append(f"arg {key} expected bool, got {type(val).__name__}")
        elif kind == "list" and not isinstance(val, list):
            misses.append(f"arg {key} expected list, got {type(val).__name__}")
        elif kind == "enum" and val not in extra:
            misses.append(f"arg {key}={val!r} not in {extra}")
        elif kind == "dict":
            if not isinstance(val, dict):
                misses.append(f"arg {key} expected object, got {type(val).__name__}")
            else:
                for sub in extra:
                    if sub not in val:
                        misses.append(f"arg {key} missing subkey {sub}")
    return misses


def _has_dsml_leak(text: str | None) -> bool:
    return bool(text and DSML_LEAK_RE.search(text))


def score_response(scenario: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any]:
    """Score a single chat completion choice for strict pass and derailment signatures."""
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    derailments: list[str] = []
    quality_misses: list[str] = []

    if _has_dsml_leak(content):
        derailments.append("dsml_leak_in_content")

    kind = scenario["kind"]
    if kind in ("plain", "no_call"):
        if finish != "stop":
            derailments.append(f"finish_reason={finish!r} (expected 'stop')")
        if kind == "no_call" and tool_calls:
            quality_misses.append(f"unexpected tool_calls={len(tool_calls)}")
        if not content.strip():
            derailments.append("empty_content")
        substr = scenario.get("expect_substr")
        if substr and substr.lower() not in content.lower():
            quality_misses.append(f"missing expected {substr!r}")
        return {
            "strict_pass": not derailments and not quality_misses,
            "derailments": derailments,
            "quality_misses": quality_misses,
            "finish_reason": finish,
            "num_tool_calls": len(tool_calls),
        }

    # Tool-calling turn (single, parallel, catalog, schema, agent, multiturn turn 1)
    min_calls = scenario.get("min_calls", 1)
    if len(tool_calls) < min_calls:
        derailments.append(
            f"missing_tool_call(got={len(tool_calls)}, min={min_calls}, finish={finish!r})"
        )
        return {
            "strict_pass": False,
            "derailments": derailments,
            "quality_misses": quality_misses,
            "finish_reason": finish,
            "num_tool_calls": len(tool_calls),
        }

    seen_calls: Counter[tuple[str, str]] = Counter()
    names: list[str] = []
    req_keys_map = scenario.get("required_keys") or {}
    type_checks_map = scenario.get("type_checks") or {}

    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        names.append(name)
        if not isinstance(raw_args, str):
            derailments.append(f"non_string_args:{name}")
            continue
        if _has_dsml_leak(raw_args):
            derailments.append(f"dsml_leak_in_args:{name}")
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            derailments.append(f"malformed_json_args:{name}")
            continue
        if not isinstance(parsed, dict):
            derailments.append(f"non_object_args:{name}")
            continue
        canon = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        seen_calls[(name, canon)] += 1
        req_keys = req_keys_map.get(name, [])
        if req_keys and not parsed:
            derailments.append(f"empty_args:{name}")
        else:
            missing = [k for k in req_keys if k not in parsed or parsed[k] in (None, "")]
            if missing:
                derailments.append(f"missing_required_args:{name}:{','.join(missing)}")
        if name in type_checks_map:
            quality_misses.extend(_check_types(parsed, type_checks_map[name]))

    for (name, _), count in seen_calls.items():
        if count > 1:
            derailments.append(f"duplicate_call:{name}x{count}")

    expect_names = scenario.get("expect_names")
    if expect_names and Counter(names) != Counter(expect_names):
        quality_misses.append(f"names={names} expected={expect_names}")

    return {
        "strict_pass": not derailments and not quality_misses,
        "derailments": derailments,
        "quality_misses": quality_misses,
        "finish_reason": finish,
        "num_tool_calls": len(tool_calls),
        "names": names,
    }


def post_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_model_gate(
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    scenarios = build_scenarios()
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for sc in scenarios:
        resp = post_chat(
            base_url,
            model,
            sc["messages"],
            temperature=temperature,
            max_tokens=max_tokens,
            tools=sc.get("tools"),
            tool_choice=sc.get("tool_choice"),
            timeout=timeout,
        )
        choice = resp["choices"][0]
        scored = score_response(sc, choice)
        entry: dict[str, Any] = {"id": sc["id"], "kind": sc["kind"], **scored}

        if sc["kind"] == "multiturn" and not scored["derailments"]:
            msg = choice["message"]
            tcs = msg.get("tool_calls") or []
            follow_msgs = [
                *sc["messages"],
                {"role": "assistant", "content": msg.get("content"), "tool_calls": tcs},
                {
                    "role": "tool",
                    "tool_call_id": tcs[0].get("id", "call_0"),
                    "content": sc["canned_result"],
                },
            ]
            resp2 = post_chat(
                base_url,
                model,
                follow_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=sc.get("tools"),
                timeout=timeout,
            )
            ch2 = resp2["choices"][0]
            sc2 = {"kind": "plain", "expect_substr": sc.get("expect_final_substr")}
            s2 = score_response(sc2, ch2)
            entry["turn2"] = s2
            entry["derailments"] = [*entry["derailments"], *(f"turn2:{d}" for d in s2["derailments"])]
            entry["quality_misses"] = [*entry["quality_misses"], *(f"turn2:{q}" for q in s2["quality_misses"])]
            entry["strict_pass"] = entry["strict_pass"] and s2["strict_pass"]

        results.append(entry)

    elapsed = round(time.perf_counter() - t0, 2)
    strict = sum(1 for r in results if r["strict_pass"])
    derailed = [r for r in results if r["derailments"]]
    return {
        "model": model,
        "temperature": temperature,
        "scenarios": len(results),
        "strict_passed": strict,
        "derailment_count": len(derailed),
        "gate_passed": len(derailed) == 0,
        "elapsed_s": elapsed,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto tool_choice and finish_reason qualification gate for stock and GLP-steered models."
    )
    parser.add_argument(
        "--base",
        default=os.environ.get("TOOL_GATE_BASE", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--models",
        default="deepseek-v4.1-flash,deepseek-v4.1-flash-weightless",
        help="Comma-separated served model names to test (default: stock + weightless)",
    )
    parser.add_argument(
        "--temperatures",
        default="0.0",
        help="Comma-separated explicit temperatures to test (default: 0.0)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", help="Optional JSON report path")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    temps = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]

    report: dict[str, Any] = {
        "base": args.base,
        "models": models,
        "temperatures": temps,
        "runs": [],
    }
    any_failed = False
    for temp in temps:
        for model in models:
            try:
                run = run_model_gate(args.base, model, temp, args.max_tokens, args.timeout)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                print(f"ERROR: {model} @ temp={temp}: {exc}", file=sys.stderr)
                return 2
            report["runs"].append(run)
            verdict = "PASS" if run["gate_passed"] else "FAIL"
            print(
                f"{verdict}: {model} @ temp={temp} — "
                f"strict {run['strict_passed']}/{run['scenarios']}, "
                f"derailments {run['derailment_count']}, {run['elapsed_s']}s"
            )
            for r in run["results"]:
                if r["derailments"] or r["quality_misses"]:
                    print(
                        f"  [{r['id']}] derailments={r['derailments']} "
                        f"quality_misses={r['quality_misses']} finish={r.get('finish_reason')}"
                    )
            if not run["gate_passed"]:
                any_failed = True

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
