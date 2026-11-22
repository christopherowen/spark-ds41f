#!/usr/bin/env python3
"""Live long-context OpenCode DSML reproducer and token-ID classifier.

usage:
  repro_live_long_dsml.py URL [--record FILE] [--target-tokens 4096 597000]
                              [--arms clean poisoned_turn1] [--modes whole stream]
                              [--trials N] [--temperature T] [--reasoning-effort high]
  repro_live_long_dsml.py --self-test

Reconstructs an OpenCode-style multi-turn coding session (`bash`, `read`, `edit`,
`grep`, `task` tools) up to ~600K tokens (`reasoning_effort="high"`, prose-before-tool
turns, optional Turn-1 poisoned `assistant.content` replay), sends requests to
`URL/v1/chat/completions` with `return_token_ids: true`, and classifies every reply by
both `content`/`tool_calls` structure and post-`</think>` token-128825 provenance:

  - `ok_tool_call`: clean tool call with JSON arguments and no raw DSML in `content`
  - `headless_dsml_in_content_genuine_128825`: screenshot leak (`<｜DSML｜ parameter`
    in `content` with genuine token 128825, `<｜DSML｜ calls>`/`invoke` omitted)
  - `headless_dsml_in_content_bpe_spelled`: screenshot leak spelled with BPE pieces
    only (no 128825 after `</think>`, e.g. imitated from Turn 1 `assistant.content`)
  - `full_dsml_in_content_bpe_spelled`: full `<｜DSML｜ calls>` block rejected by 0768
    token provenance because `｜DSML｜` was spelled with ordinary BPE pieces
  - `v4_unspaced_dsml_in_content`: V4 unspaced `<｜DSML｜tool_calls>` / `<｜DSML｜invoke`
  - `swallowed_after_tool_start_preamble_blackhole`: genuine `<｜DSML｜ calls>` followed
    by unspaced or BPE `<｜DSML｜ invoke>`, silently dropped in `TOOL_PREAMBLE`
  - `malformed_tool_json`: tool call emitted with invalid JSON or missing `command`
  - `prose_only_no_tool`: plain prose response with neither tool call nor DSML leak

Stdlib only. Shares one `cache_salt` per `(arm, target_tokens)` cohort so multi-trial
sampling at `temperature=1.0` prefills a 597K history once and decodes trials cheaply.
"""

from __future__ import annotations

import argparse
import http.server
import json
import random
import socketserver
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

DSML_ATOM = "｜DSML｜"
DSML_TOKEN_ID = 128825
THINK_START_ID = 128821
THINK_END_ID = 128822

OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes a bash command in the workspace with an optional timeout in ms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {"type": "integer", "description": "Timeout in milliseconds"},
                    "description": {"type": "string", "description": "Short summary of command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Reads a file from the workspace filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replaces an exact string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string"},
                    "newString": {"type": "string"},
                    "replaceAll": {"type": "boolean"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Searches file contents using a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "include": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Delegates a focused investigation task to a subagent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        },
    },
]

CARRIERS = (
    "royalmail",
    "ukrposhta",
    "cainiao",
    "usps",
    "laposte",
    "correos",
    "dhl",
    "dpd",
    "gls",
    "bpost",
    "postnl",
    "japanpost",
)

TURN1_POISONED_CONTENT = (
    "Let me inspect the ukrposhta captcha mapping lines before checking retry bounds.\n\n"
    f'<{DSML_ATOM} parameter name="command">echo "=== ukr captcha mapping"; '
    f"sed -n '60,75p' src/carriers/ukrposhta/index.ts</{DSML_ATOM} parameter>\n"
    f'<{DSML_ATOM} parameter name="timeout">400000</{DSML_ATOM} parameter>\n'
    f"</{DSML_ATOM} invoke>\n"
    f"</{DSML_ATOM} calls>"
)

FINAL_USER_PROMPT = (
    "Subagent finished: Bank ukrposhta, cu.\n"
    "royalmail still shows ~9 MB/lookup under retry churn on unbounded lookups. "
    "First state your 1-sentence diagnosis in plain text, then invoke the `bash` tool "
    "(with `timeout` 400000) to run `bun run typecheck` and "
    "`bun test src/core/mint.test.ts scripts/bench-5rps.test.ts` before editing."
)


def _approx_tokens(text: str) -> int:
    # Calibrated against DeepSeek-V4.1-Flash-1017 BPE on carrier trace + TS tool turns (~3.01 chars/tok).
    return max(1, int(len(text) / 3.01))


def _render_carrier_dump(rng: random.Random, carrier: str, turn_idx: int, lines: int) -> str:
    rows = [
        f"// src/carriers/{carrier}/index.ts (snapshot turn={turn_idx})",
        f"export interface {carrier.capitalize()}LookupOptions {{ timeoutMs?: number; maxRetries?: number; }}",
    ]
    for line_no in range(1, lines + 1):
        status = rng.choice((200, 200, 200, 429, 502, 503))
        bytes_read = rng.randint(14_000, 960_000)
        delay_ms = rng.randint(25, 2400)
        rows.append(
            f"[{line_no:04d}] trace_id={carrier}-{turn_idx:04d}-{line_no:04d} "
            f"http_status={status} payload_bytes={bytes_read} backoff_ms={delay_ms} "
            f"endpoint=\"https://track.{carrier}.example/v2/events?raw=1&history=full\""
        )
    return "\n".join(rows)


BARE_SUBAGENT_WAKEUP = (
    "General finished · Bank ukrposhta, cu.\n"
    "<system-reminder>Goal active: verify pending edits (typecheck + mint/bench-5rps "
    "tests) before bounding royalmail's 9 MB/lookup retry churn.</system-reminder>"
)

POISONED_CASCADE_TURNS = (
    (
        "cainiao's 6.4 MB/lookup comes from retaining raw tracking HTML per retry. "
        "Let me check its parser slice first.\n\n"
        f'<{DSML_ATOM} parameter name="command">cd /Users/herlon/dev/woncadigital/'
        "workspace/multitracking && sed -n '40,85p' src/carriers/cainiao/index.ts"
        f"</{DSML_ATOM} parameter>\n"
        f'<{DSML_ATOM} parameter name="timeout">400000</{DSML_ATOM} parameter>\n'
        f"</{DSML_ATOM} invoke>\n"
        f"</{DSML_ATOM} calls>",
        "General finished · Bank usps, laposte.",
    ),
    (
        "usps and laposte are bounded. Checking ukrposhta's captcha mapping next.\n\n"
        f'<{DSML_ATOM} parameter name="command">echo "=== ukr captcha mapping"; '
        f"sed -n '60,75p' src/carriers/ukrposhta/index.ts</{DSML_ATOM} parameter>\n"
        f'<{DSML_ATOM} parameter name="timeout">400000</{DSML_ATOM} parameter>\n'
        f"</{DSML_ATOM} invoke>\n"
        f"</{DSML_ATOM} calls>",
        BARE_SUBAGENT_WAKEUP,
    ),
)


def build_opencode_messages(
    target_tokens: int,
    arm: str,
    seed: int,
) -> tuple[list[dict], int]:
    """Build an OpenCode multi-turn history scaled to approximately `target_tokens`."""
    assert arm in (
        "clean",
        "poisoned_turn1",
        "with_reasoning_history",
        "sparse_long_gap",
        "sparse_subagent_cascade",
    )
    rng = random.Random(seed)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are OpenCode (Build mode), an autonomous software engineering agent "
                "working in `/Users/herlon/dev/woncadigital/workspace/multitracking`."
            ),
        },
        {
            "role": "user",
            "content": (
                "Audit memory per lookup across all carrier adapters in `src/carriers/`, "
                "check captcha/retry handling, and verify `typecheck` + `bench-5rps`."
            ),
        },
    ]

    est_tokens = sum(_approx_tokens(m["content"]) for m in messages) + 950

    if arm in ("sparse_long_gap", "sparse_subagent_cascade"):
        # Only 1 early clean tool call near token ~1K, followed by massive ~60K-token
        # subagent / inventory reports so no clean `<｜DSML｜ calls>` appears in the
        # final ~590K tokens of context.
        early_args = json.dumps(
            {
                "description": "Audit carrier adapters",
                "prompt": "Check retry loops and per-lookup memory across carriers.",
                "subagent_type": "general",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "Dispatching subagents to inventory carrier adapters.",
                "tool_calls": [
                    {
                        "id": "call_init_01",
                        "type": "function",
                        "function": {"name": "task", "arguments": early_args},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": "call_init_01",
                "content": "Launched 7 background General subagents.",
            }
        )
        est_tokens += 140
        reserve = 650 if arm == "sparse_subagent_cascade" else 220
        sub_idx = 1
        while est_tokens + reserve < target_tokens:
            carrier = CARRIERS[(sub_idx - 1) % len(CARRIERS)]
            remaining = max(200, target_tokens - reserve - est_tokens)
            chunk_tokens = min(remaining, 48_000)
            dump_lines = max(16, int((chunk_tokens * 3.01) / 145))
            report = _render_carrier_dump(rng, carrier, sub_idx, dump_lines)
            messages.append(
                {
                    "role": "user",
                    "content": f"Subagent inventory batch {sub_idx} ({carrier}):\n{report}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Recorded `{carrier}` batch {sub_idx} payload retention.",
                }
            )
            est_tokens += _approx_tokens(report) + 48
            sub_idx += 1

        if arm == "sparse_subagent_cascade":
            for poisoned_assistant, followup_user in POISONED_CASCADE_TURNS:
                messages.append({"role": "assistant", "content": poisoned_assistant})
                messages.append({"role": "user", "content": followup_user})
                est_tokens += (
                    _approx_tokens(poisoned_assistant)
                    + _approx_tokens(followup_user)
                    + 32
                )
        else:
            messages.append({"role": "user", "content": BARE_SUBAGENT_WAKEUP})
            est_tokens += _approx_tokens(BARE_SUBAGENT_WAKEUP) + 24
        return messages, est_tokens

    reserve = 420 if arm == "poisoned_turn1" else 220
    turn_idx = 1

    while est_tokens + reserve < target_tokens:
        carrier = CARRIERS[(turn_idx - 1) % len(CARRIERS)]
        remaining = max(120, target_tokens - reserve - est_tokens)
        # Keep each tool turn ~2K-8K tokens so a 600K prompt has ~90-120 realistic tool turns.
        desired_turn_tokens = min(remaining, rng.randint(2400, 6800))
        dump_lines = max(8, min(320, int((desired_turn_tokens * 3.01) / 145)))

        call_id = f"call_{turn_idx:04d}"
        if turn_idx % 2 == 1:
            prose = (
                f"Checking `{carrier}` retry loop and per-lookup response buffering "
                f"before comparing heap deltas."
            )
            cmd = (
                f"cd /workspace/multitracking && "
                f"echo '=== {carrier} audit {turn_idx} ==='; "
                f"sed -n '1,120p' src/carriers/{carrier}/index.ts"
            )
            tool_args = json.dumps({"command": cmd, "timeout": 400000})
            tool_name = "bash"
        else:
            prose = f"Reading `{carrier}` fixture trace log to verify unbounded retry payloads."
            tool_args = json.dumps(
                {
                    "filePath": f"src/carriers/{carrier}/trace-{turn_idx:04d}.log",
                    "offset": 1,
                    "limit": dump_lines,
                }
            )
            tool_name = "read"

        assistant_msg: dict = {
            "role": "assistant",
            "content": prose,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args},
                }
            ],
        }
        if arm == "with_reasoning_history":
            assistant_msg["reasoning_content"] = (
                f"Inspect {carrier} adapter state on turn {turn_idx} and bound retry allocations."
            )

        tool_output = _render_carrier_dump(rng, carrier, turn_idx, dump_lines)
        tool_msg = {"role": "tool", "tool_call_id": call_id, "content": tool_output}

        messages.append(assistant_msg)
        messages.append(tool_msg)
        est_tokens += (
            _approx_tokens(prose)
            + _approx_tokens(tool_args)
            + _approx_tokens(tool_output)
            + 48
        )
        turn_idx += 1

    if arm == "poisoned_turn1":
        messages.append({"role": "assistant", "content": TURN1_POISONED_CONTENT})
        est_tokens += _approx_tokens(TURN1_POISONED_CONTENT) + 24

    messages.append({"role": "user", "content": FINAL_USER_PROMPT})
    est_tokens += _approx_tokens(FINAL_USER_PROMPT) + 24
    return messages, est_tokens


def model_name(base: str, timeout: float) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as response:
        return json.load(response)["data"][0]["id"]


def send_chat(
    base: str,
    model: str,
    messages: list[dict],
    *,
    stream: bool,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
    cache_salt: str,
    timeout: float,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "tools": OPENCODE_TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not stream:
            payload = json.load(response)
            choice = payload["choices"][0]
            msg = choice["message"]
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "calls": [
                    {
                        "name": tc["function"].get("name") or "",
                        "arguments": tc["function"].get("arguments") or "",
                    }
                    for tc in (msg.get("tool_calls") or [])
                ],
                "token_ids": choice.get("token_ids") or [],
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage") or {},
            }

        content, reasoning, finish_reason = "", "", None
        usage: dict = {}
        token_ids: list[int] = []
        parts: dict[int, dict[str, str]] = {}
        for raw_line in response:
            if not raw_line.startswith(b"data:") or raw_line.strip() == b"data: [DONE]":
                continue
            event = json.loads(raw_line[5:].strip())
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                token_ids.extend(choice.get("token_ids") or [])
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                reasoning += (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                finish_reason = choice.get("finish_reason") or finish_reason
                for call in delta.get("tool_calls") or []:
                    slot = parts.setdefault(
                        call.get("index", 0), {"name": "", "arguments": ""}
                    )
                    fn = call.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
        return {
            "content": content,
            "reasoning": reasoning,
            "calls": [parts[idx] for idx in sorted(parts)],
            "token_ids": token_ids,
            "finish_reason": finish_reason,
            "usage": usage,
        }


def classify_reply(reply: dict) -> dict:
    """Classify a `/v1/chat/completions` reply into exact DSML failure modes."""
    content: str = reply["content"]
    calls: list[dict] = reply["calls"]
    ids: list[int] = reply["token_ids"]

    think_end_positions = [i for i, tid in enumerate(ids) if tid == THINK_END_ID]
    post_think_start = think_end_positions[-1] + 1 if think_end_positions else 0
    post_think_ids = ids[post_think_start:]
    dsml_positions_all = [i for i, tid in enumerate(ids) if tid == DSML_TOKEN_ID]
    dsml_positions_post_think = [
        i for i, tid in enumerate(ids) if i >= post_think_start and tid == DSML_TOKEN_ID
    ]

    has_any_dsml_tag = f"<{DSML_ATOM}" in content or f"</{DSML_ATOM}" in content
    has_spaced_param = f"<{DSML_ATOM} parameter" in content or f"</{DSML_ATOM} parameter>" in content
    has_spaced_calls_or_invoke = (
        f"<{DSML_ATOM} calls>" in content or f'<{DSML_ATOM} invoke name="' in content
    )
    has_v4_unspaced = any(
        marker in content
        for marker in (
            f"<{DSML_ATOM}tool_calls>",
            f"<{DSML_ATOM}function_calls>",
            f"<{DSML_ATOM}calls>",
            f'<{DSML_ATOM}invoke name="',
            f"<{DSML_ATOM}parameter",
        )
    )

    parsed_calls = []
    json_errors = []
    for call in calls:
        try:
            args_obj = json.loads(call["arguments"] or "{}")
            parsed_calls.append({"name": call["name"], "arguments": args_obj})
        except ValueError as exc:
            json_errors.append(f"{call['name']}: {exc}")
            parsed_calls.append(
                {"name": call["name"], "arguments": {"_raw": call["arguments"]}}
            )

    if calls and not json_errors and not has_any_dsml_tag:
        first_args = parsed_calls[0]["arguments"]
        if parsed_calls[0]["name"] == "bash" and not isinstance(
            first_args.get("command"), str
        ):
            verdict = "malformed_tool_json"
        else:
            verdict = "ok_tool_call"
    elif json_errors:
        verdict = "malformed_tool_json"
    elif has_spaced_param and not has_spaced_calls_or_invoke and not calls:
        if dsml_positions_post_think:
            verdict = "headless_dsml_in_content_genuine_128825"
        else:
            verdict = "headless_dsml_in_content_bpe_spelled"
    elif has_spaced_calls_or_invoke and not calls:
        verdict = "full_dsml_in_content_bpe_spelled"
    elif has_v4_unspaced and not calls:
        verdict = "v4_unspaced_dsml_in_content"
    elif not calls and not has_any_dsml_tag and dsml_positions_post_think:
        verdict = "swallowed_after_tool_start_preamble_blackhole"
    elif has_any_dsml_tag:
        verdict = "other_dsml_leak_in_content"
    else:
        verdict = "prose_only_no_tool"

    return {
        "verdict": verdict,
        "replicated_screenshot_error": verdict
        in (
            "headless_dsml_in_content_genuine_128825",
            "headless_dsml_in_content_bpe_spelled",
        ),
        "any_dsml_failure": verdict
        not in ("ok_tool_call", "prose_only_no_tool"),
        "parsed_calls": parsed_calls,
        "dsml_token_count_total": len(dsml_positions_all),
        "dsml_token_count_post_think": len(dsml_positions_post_think),
        "dsml_token_positions_post_think": dsml_positions_post_think[:24],
    }


# ── Local CPU `--self-test` Harness ──────────────────────────────────────────


def _run_self_test() -> int:
    """Verify prompt builder, whole + SSE streaming client, and classifier locally."""
    for arm in ("clean", "poisoned_turn1", "with_reasoning_history"):
        msgs, est = build_opencode_messages(24_000, arm=arm, seed=20261122)
        assert est >= 23_500, (arm, est)
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        if arm == "poisoned_turn1":
            assert msgs[-2]["role"] == "assistant"
            assert f'<{DSML_ATOM} parameter name="command">' in msgs[-2]["content"]

    screenshot_content = (
        "More precise: royalmail's 9 MB/lookup is retry churn from an unbounded lookup. "
        "Let me verify my pending edits first, then bound it.\n\n"
        f'<{DSML_ATOM} parameter name="command">cd /Users/herlon/dev/woncadigital/'
        "workspace/multitracking && echo \"typecheck: $(bun run typecheck 2>&1 | "
        "grep -cE 'error TS')\"; bun test src/core/mint.test.ts "
        f"scripts/bench-5rps.test.ts 2>&1 | tail -3</{DSML_ATOM} parameter>\n"
        f'<{DSML_ATOM} parameter name="timeout">400000</{DSML_ATOM} parameter>\n'
        f"</{DSML_ATOM} invoke>\n"
        f"</{DSML_ATOM} calls>"
    )

    canned_cases = [
        (
            "ok_tool_call",
            {
                "reasoning": "Run typecheck and tests first.",
                "content": "Checking pending edits with typecheck and tests.\n\n",
                "tool_calls": [
                    {
                        "id": "call_ok",
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {"command": "bun run typecheck", "timeout": 400000}
                            ),
                        },
                    }
                ],
                "token_ids": [101, THINK_END_ID, 202, DSML_TOKEN_ID, 303],
            },
        ),
        (
            "headless_dsml_in_content_genuine_128825",
            {
                "reasoning": "Verify pending edits before bounding royalmail.",
                "content": screenshot_content,
                "tool_calls": [],
                "token_ids": [
                    11,
                    THINK_END_ID,
                    22,
                    DSML_TOKEN_ID,
                    DSML_TOKEN_ID,
                    DSML_TOKEN_ID,
                ],
            },
        ),
        (
            "headless_dsml_in_content_bpe_spelled",
            {
                "reasoning": "Imitating previous assistant turn.",
                "content": screenshot_content,
                "tool_calls": [],
                "token_ids": [11, THINK_END_ID, 22, 33, 44, 55],
            },
        ),
        (
            "swallowed_after_tool_start_preamble_blackhole",
            {
                "reasoning": "Calling bash tool.",
                "content": "Checking pending edits.\n\n",
                "tool_calls": [],
                "token_ids": [11, THINK_END_ID, 22, DSML_TOKEN_ID, 77, DSML_TOKEN_ID],
            },
        ),
    ]
    cursor = {"idx": 0}

    class MockHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                body = json.dumps({"data": [{"id": "deepseek-v41-mock"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            _, spec = canned_cases[cursor["idx"] % len(canned_cases)]
            cursor["idx"] += 1

            if not req.get("stream"):
                payload = {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls" if spec["tool_calls"] else "stop",
                            "token_ids": spec["token_ids"],
                            "message": {
                                "role": "assistant",
                                "reasoning_content": spec["reasoning"],
                                "content": spec["content"],
                                "tool_calls": spec["tool_calls"] or None,
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 597800, "completion_tokens": len(spec["token_ids"])},
                }
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            mid = len(spec["content"]) // 2
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": spec["token_ids"][:2],
                            "delta": {
                                "reasoning_content": spec["reasoning"],
                                "content": spec["content"][:mid],
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": spec["token_ids"][2:],
                            "finish_reason": "tool_calls" if spec["tool_calls"] else "stop",
                            "delta": {
                                "content": spec["content"][mid:],
                                "tool_calls": spec["tool_calls"] or None,
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 597800, "completion_tokens": len(spec["token_ids"])},
                },
            ]
            for ch in chunks:
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    class ReusableTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    with ReusableTCPServer(("127.0.0.1", 0), MockHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"
        resolved_model = model_name(base, 5.0)
        assert resolved_model == "deepseek-v41-mock"

        msgs, _ = build_opencode_messages(4096, arm="poisoned_turn1", seed=20261122)
        for mode in ("whole", "stream"):
            cursor["idx"] = 0
            for expected_verdict, _ in canned_cases:
                reply = send_chat(
                    base,
                    resolved_model,
                    msgs,
                    stream=(mode == "stream"),
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1024,
                    reasoning_effort="high",
                    cache_salt="selftest",
                    timeout=10.0,
                )
                judged = classify_reply(reply)
                assert judged["verdict"] == expected_verdict, (mode, judged, expected_verdict)
        httpd.shutdown()

    print(
        json.dumps(
            {
                "self_test": "passed",
                "verified_verdicts": [name for name, _ in canned_cases],
                "modes": ["whole", "stream"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", help="Serving base URL, e.g. http://dgx1:8000")
    parser.add_argument("--self-test", action="store_true", help="Run offline local mock verification")
    parser.add_argument("--model", help="Override served model id")
    parser.add_argument(
        "--target-tokens",
        type=int,
        nargs="+",
        default=[4096, 597000],
        help="Approximate prompt token lengths to test (default: 4096 597000)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "clean",
            "poisoned_turn1",
            "with_reasoning_history",
            "sparse_long_gap",
            "sparse_subagent_cascade",
        ),
        default=["clean", "poisoned_turn1", "sparse_long_gap", "sparse_subagent_cascade"],
        help="History construction arms to test",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("whole", "stream"),
        default=["whole", "stream"],
    )
    parser.add_argument("--trials", type=int, default=4, help="Trials per (arm, tokens, mode)")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default 1.0, matching config/cluster.json)",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--record", type=Path, help="Optional JSONL file to save full replies")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()
    if not args.url:
        parser.error("URL is required unless --self-test is given")

    base = args.url.rstrip("/")
    try:
        served_model = args.model or model_name(base, 60.0)
    except OSError as exc:
        print(json.dumps({"summary": "dsml_long_context_repro", "error": str(exc)}))
        return 1

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)

    verdict_counts: dict[str, int] = {}
    screenshot_hits = 0
    any_dsml_failures = 0

    for arm in args.arms:
        for target_tokens in args.target_tokens:
            messages, est_tokens = build_opencode_messages(
                target_tokens, arm=arm, seed=args.seed
            )
            cohort_salt = f"dsml-repro-{arm}-{target_tokens}-{uuid.uuid4().hex[:8]}"
            for mode in args.modes:
                for trial in range(1, args.trials + 1):
                    started = time.monotonic()
                    try:
                        reply = send_chat(
                            base,
                            served_model,
                            messages,
                            stream=(mode == "stream"),
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            reasoning_effort=args.reasoning_effort,
                            cache_salt=cohort_salt,
                            timeout=args.timeout,
                        )
                    except (OSError, ValueError, KeyError) as exc:
                        row = {
                            "arm": arm,
                            "target_tokens": target_tokens,
                            "est_tokens": est_tokens,
                            "mode": mode,
                            "trial": trial,
                            "verdict": "request_error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds": round(time.monotonic() - started, 3),
                        }
                    else:
                        judged = classify_reply(reply)
                        verdict = judged["verdict"]
                        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                        screenshot_hits += int(judged["replicated_screenshot_error"])
                        any_dsml_failures += int(judged["any_dsml_failure"])
                        row = {
                            "arm": arm,
                            "target_tokens": target_tokens,
                            "est_tokens": est_tokens,
                            "prompt_tokens": reply["usage"].get("prompt_tokens"),
                            "completion_tokens": reply["usage"].get("completion_tokens"),
                            "mode": mode,
                            "trial": trial,
                            "verdict": verdict,
                            "replicated_screenshot_error": judged["replicated_screenshot_error"],
                            "any_dsml_failure": judged["any_dsml_failure"],
                            "dsml_token_count_post_think": judged["dsml_token_count_post_think"],
                            "finish_reason": reply["finish_reason"],
                            "reasoning_chars": len(reply["reasoning"]),
                            "content_preview": reply["content"][:500],
                            "calls": judged["parsed_calls"],
                            "seconds": round(time.monotonic() - started, 3),
                        }
                        if args.record:
                            with args.record.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        {**row, "full_reply": reply},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                    print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "summary": "dsml_long_context_repro",
        "model": served_model,
        "verdict_counts": verdict_counts,
        "replicated_screenshot_hits": screenshot_hits,
        "any_dsml_failures": any_dsml_failures,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
