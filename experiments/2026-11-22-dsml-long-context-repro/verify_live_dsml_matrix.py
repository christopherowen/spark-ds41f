#!/usr/bin/env python3
"""Live end-to-end DSML serving verification matrix on `/v1/chat/completions`.

usage:
  verify_live_dsml_matrix.py URL [--record FILE] [--multi-turns 12]
  verify_live_dsml_matrix.py --self-test

Covers:
  1. `tool_choice="auto"` (whole + stream)
  2. `tool_choice="required"` (whole + stream, xgrammar structural tag)
  3. Named `tool_choice={"type": "function", ...}` (whole + stream)
  4. Quoted DSML closers inside a `write` string argument (whole + stream)
  5. Prose DSML explanation without tool execution (`tool_choice="none"` + `auto`)
  6. Multi-turn tool loop omitting `reasoning_content` and echoing streamed
     `assistant.content` verbatim (including trailing `\\n\\n`), checking that
     reasoning does not collapse to 0-char `<think></think>` skips.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repro_live_long_dsml import (  # noqa: E402
    DSML_ATOM,
    OPENCODE_TOOLS,
    classify_reply,
    model_name,
)


def _call_chat(
    base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    temperature: float = 0.0,
    max_tokens: int = 1536,
    tool_choice: Any = "auto",
    tools: list[dict[str, Any]] | None = OPENCODE_TOOLS,
    reasoning_effort: str = "high",
    cache_salt: str | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "return_token_ids": True,
    }
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if cache_salt:
        body["cache_salt"] = cache_salt
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            payload = json.load(resp)
            choice = payload["choices"][0]
            msg = choice["message"]
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "calls": [
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "name": tc["function"].get("name") or "",
                        "arguments": tc["function"].get("arguments") or "",
                    }
                    for i, tc in enumerate(msg.get("tool_calls") or [])
                ],
                "token_ids": choice.get("token_ids") or [],
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage") or {},
            }

        content, reasoning, finish_reason = "", "", None
        usage: dict[str, Any] = {}
        token_ids: list[int] = []
        parts: dict[int, dict[str, str]] = {}
        for raw in resp:
            if not raw.startswith(b"data:") or raw.strip() == b"data: [DONE]":
                continue
            ev = json.loads(raw[5:].strip())
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices") or []:
                token_ids.extend(choice.get("token_ids") or [])
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                reasoning += (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                finish_reason = choice.get("finish_reason") or finish_reason
                for tc in delta.get("tool_calls") or []:
                    slot = parts.setdefault(
                        tc.get("index", 0),
                        {"id": tc.get("id") or "", "name": "", "arguments": ""},
                    )
                    if tc.get("id") and not slot["id"]:
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
        return {
            "content": content,
            "reasoning": reasoning,
            "calls": [
                {
                    "id": parts[idx]["id"] or f"call_{idx}",
                    "name": parts[idx]["name"],
                    "arguments": parts[idx]["arguments"],
                }
                for idx in sorted(parts)
            ],
            "token_ids": token_ids,
            "finish_reason": finish_reason,
            "usage": usage,
        }


WRITE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write exact text content to a file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Full text content to write."},
            },
            "required": ["path", "content"],
        },
    },
}

TOOLS_WITH_WRITE: list[dict[str, Any]] = [*OPENCODE_TOOLS, WRITE_TOOL]


def _has_raw_dsml(text: str) -> bool:
    return f"<{DSML_ATOM}" in text or f"</{DSML_ATOM}" in text


def run_matrix(base: str, served_model: str, *, multi_turns: int = 12, record: Path | None = None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    def log_case(name: str, passed: bool, detail: dict[str, Any]) -> None:
        row = {"case": name, "passed": passed, **detail}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if record is not None:
            with record.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for stream in (False, True):
        mode = "stream" if stream else "whole"

        # 1. tool_choice="auto"
        t0 = time.monotonic()
        rep = _call_chat(
            base,
            served_model,
            [
                {
                    "role": "user",
                    "content": "State a 1-sentence note, then use `bash` to run `bun test` with timeout 400000.",
                }
            ],
            stream=stream,
            tool_choice="auto",
            cache_salt=f"dsml-auto-{mode}-{uuid.uuid4().hex[:8]}",
        )
        args0 = json.loads(rep["calls"][0]["arguments"]) if rep["calls"] else {}
        ok = (
            len(rep["calls"]) == 1
            and rep["calls"][0]["name"] == "bash"
            and "bun test" in str(args0.get("command", ""))
            and args0.get("timeout") == 400000
            and not _has_raw_dsml(rep["content"])
            and len(rep["reasoning"]) > 0
        )
        log_case(
            f"auto_bash_{mode}",
            ok,
            {
                "calls": rep["calls"],
                "reasoning_chars": len(rep["reasoning"]),
                "content": rep["content"],
                "seconds": round(time.monotonic() - t0, 3),
            },
        )

        # 2. tool_choice="required"
        t0 = time.monotonic()
        rep_req = _call_chat(
            base,
            served_model,
            [{"role": "user", "content": "Inspect `src/index.ts` or run `ls`."}],
            stream=stream,
            tool_choice="required",
            cache_salt=f"dsml-req-{mode}-{uuid.uuid4().hex[:8]}",
        )
        ok_req = (
            len(rep_req["calls"]) >= 1
            and rep_req["calls"][0]["name"] in {"bash", "read", "write"}
            and not _has_raw_dsml(rep_req["content"])
        )
        log_case(
            f"required_tool_choice_{mode}",
            ok_req,
            {
                "calls": rep_req["calls"],
                "reasoning_chars": len(rep_req["reasoning"]),
                "content": rep_req["content"],
                "seconds": round(time.monotonic() - t0, 3),
            },
        )

        # 3. Named tool_choice={"type": "function", "function": {"name": "read"}}
        t0 = time.monotonic()
        rep_named = _call_chat(
            base,
            served_model,
            [{"role": "user", "content": "Open `src/carriers/royalmail/index.ts` lines 1 to 50."}],
            stream=stream,
            tool_choice={"type": "function", "function": {"name": "read"}},
            cache_salt=f"dsml-named-{mode}-{uuid.uuid4().hex[:8]}",
        )
        n_args = json.loads(rep_named["calls"][0]["arguments"]) if rep_named["calls"] else {}
        ok_named = (
            len(rep_named["calls"]) == 1
            and rep_named["calls"][0]["name"] == "read"
            and "royalmail" in str(n_args.get("filePath", ""))
            and not _has_raw_dsml(rep_named["content"])
        )
        log_case(
            f"named_tool_choice_{mode}",
            ok_named,
            {
                "calls": rep_named["calls"],
                "reasoning_chars": len(rep_named["reasoning"]),
                "content": rep_named["content"],
                "seconds": round(time.monotonic() - t0, 3),
            },
        )

        # 4. Quoted DSML closers inside `write` string argument
        t0 = time.monotonic()
        rep_quote = _call_chat(
            base,
            served_model,
            [
                {
                    "role": "user",
                    "content": (
                        "Call `write` with `path` set to `dsml-notes.md` and `content` "
                        "containing a 3-line example showing `</｜DSML｜ parameter>`, "
                        "`</｜DSML｜ invoke>`, and `</｜DSML｜ calls>` followed by the final line `EOF`."
                    ),
                }
            ],
            stream=stream,
            tool_choice="auto",
            tools=TOOLS_WITH_WRITE,
            cache_salt=f"dsml-quote-{mode}-{uuid.uuid4().hex[:8]}",
        )
        q_args = json.loads(rep_quote["calls"][0]["arguments"]) if rep_quote["calls"] else {}
        q_body = str(q_args.get("content", ""))
        ok_quote = (
            len(rep_quote["calls"]) >= 1
            and rep_quote["calls"][0]["name"] == "write"
            and q_args.get("path") == "dsml-notes.md"
            and "EOF" in q_body
            and DSML_ATOM in q_body
            and not _has_raw_dsml(rep_quote["content"])
        )
        log_case(
            f"quoted_dsml_in_write_arg_{mode}",
            ok_quote,
            {
                "calls": rep_quote["calls"],
                "reasoning_chars": len(rep_quote["reasoning"]),
                "content": rep_quote["content"],
                "seconds": round(time.monotonic() - t0, 3),
            },
        )

        # 5. Prose DSML example with tool_choice="none" stays literal content
        t0 = time.monotonic()
        rep_none = _call_chat(
            base,
            served_model,
            [
                {
                    "role": "user",
                    "content": "Do not run any tools; briefly show a 2-line example of `<｜DSML｜ calls>` syntax in markdown.",
                }
            ],
            stream=stream,
            tool_choice="none",
            cache_salt=f"dsml-none-{mode}-{uuid.uuid4().hex[:8]}",
        )
        ok_none = len(rep_none["calls"]) == 0 and len(rep_none["content"].strip()) > 0
        log_case(
            f"tool_choice_none_prose_{mode}",
            ok_none,
            {
                "calls": rep_none["calls"],
                "reasoning_chars": len(rep_none["reasoning"]),
                "content_len": len(rep_none["content"]),
                "seconds": round(time.monotonic() - t0, 3),
            },
        )

    files = [
        "src/carriers/royalmail/index.ts",
        "src/carriers/ukrposhta/index.ts",
        "src/carriers/dhl/index.ts",
        "src/carriers/usps/index.ts",
        "src/carriers/fedex/index.ts",
        "src/carriers/ups/index.ts",
        "src/carriers/cainiao/index.ts",
        "src/carriers/dpd/index.ts",
        "src/carriers/gls/index.ts",
        "src/carriers/colissimo/index.ts",
        "src/carriers/correos/index.ts",
        "src/carriers/postnl/index.ts",
    ]

    # 6a. Multi-user-turn loop omitting `reasoning_content` across prior user turns:
    # `<｜Assistant｜></think>` before `last_user_idx` must keep full reasoning on each new user turn.
    user_loop_salt = f"dsml-user-loop-{uuid.uuid4().hex[:8]}"
    user_msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an engineer auditing carrier retry bounds. State one brief sentence, then call `read` on the requested file.",
        },
        {
            "role": "user",
            "content": f"Audit retry bounds in `{files[0]}` by calling `read` on it first.",
        },
    ]
    u_zero_reasoning = 0
    u_tool_calls = 0
    u_dsml_leaks = 0
    for turn_idx in range(multi_turns):
        rep_u = _call_chat(
            base,
            served_model,
            user_msgs,
            stream=True,
            temperature=0.0,
            tool_choice="auto",
            cache_salt=user_loop_salt,
        )
        judged_u = classify_reply(rep_u)
        if len(rep_u["reasoning"]) == 0:
            u_zero_reasoning += 1
        if judged_u["any_dsml_failure"]:
            u_dsml_leaks += 1
        next_file = files[(turn_idx + 1) % len(files)]
        if rep_u["calls"]:
            u_tool_calls += 1
            tc_list = [
                {
                    "id": c["id"] or f"u_call_{turn_idx}_{j}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for j, c in enumerate(rep_u["calls"])
            ]
            # Deliberately omit `reasoning_content` and echo `content` verbatim (with trailing `\n\n`)
            user_msgs.append(
                {
                    "role": "assistant",
                    "content": rep_u["content"],
                    "tool_calls": tc_list,
                }
            )
            for j, tc in enumerate(tc_list):
                user_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"// {files[turn_idx % len(files)]}\n"
                            f"export const MAX_RETRIES = {3 + (turn_idx % 4)};\n"
                            "export async function lookup(id: string) { return fetch(id); }\n"
                            if j == 0
                            else "ok"
                        ),
                    }
                )
            user_msgs.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Verified `{files[turn_idx % len(files)]}` bounds `MAX_RETRIES = {3 + (turn_idx % 4)}`."
                    ),
                }
            )
            user_msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"Now audit retry bounds and backoff behavior in `{next_file}` by calling `read` on it."
                    ),
                }
            )
        else:
            user_msgs.append({"role": "assistant", "content": rep_u["content"]})
            user_msgs.append(
                {
                    "role": "user",
                    "content": f"Now audit retry bounds and backoff behavior in `{next_file}` by calling `read` on it.",
                }
            )

    log_case(
        "multiturn_user_omitted_reasoning_loop",
        u_tool_calls == multi_turns and u_dsml_leaks == 0,
        {
            "turns": multi_turns,
            "tool_call_turns": u_tool_calls,
            "zero_reasoning_turns": u_zero_reasoning,
            "dsml_leak_turns": u_dsml_leaks,
        },
    )

    # 6b. Intra-turn tool loop preserving active-turn `reasoning_content` (and echoing `\n\n` content)
    active_salt = f"dsml-active-loop-{uuid.uuid4().hex[:8]}"
    active_msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an engineer auditing carrier files one by one. Think through each file's retry constant before calling `read` on the next target.",
        },
        {
            "role": "user",
            "content": f"Inspect retry constants across carrier files starting with `{files[0]}`.",
        },
    ]
    a_zero_reasoning = 0
    a_tool_calls = 0
    a_dsml_leaks = 0
    for turn_idx in range(multi_turns):
        rep_a = _call_chat(
            base,
            served_model,
            active_msgs,
            stream=True,
            temperature=0.0,
            tool_choice="auto",
            cache_salt=active_salt,
        )
        judged_a = classify_reply(rep_a)
        if len(rep_a["reasoning"]) == 0:
            a_zero_reasoning += 1
        if judged_a["any_dsml_failure"]:
            a_dsml_leaks += 1
        next_file = files[(turn_idx + 1) % len(files)]
        if rep_a["calls"]:
            a_tool_calls += 1
            tc_list = [
                {
                    "id": c["id"] or f"a_call_{turn_idx}_{j}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for j, c in enumerate(rep_a["calls"])
            ]
            active_msgs.append(
                {
                    "role": "assistant",
                    "reasoning_content": rep_a["reasoning"],
                    "content": rep_a["content"],
                    "tool_calls": tc_list,
                }
            )
            for j, tc in enumerate(tc_list):
                active_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"// {files[turn_idx % len(files)]}\n"
                            f"export const MAX_RETRIES = {2 << (turn_idx % 4)};\n"
                            f"// Next carrier to inspect with `read`: {next_file}"
                            if j == 0
                            else "ok"
                        ),
                    }
                )
        else:
            active_msgs.append(
                {
                    "role": "assistant",
                    "reasoning_content": rep_a["reasoning"],
                    "content": rep_a["content"],
                }
            )
            active_msgs.append({"role": "user", "content": f"Call `read` on `{next_file}`."})

    log_case(
        "intraturn_active_reasoning_loop",
        a_tool_calls == multi_turns and a_zero_reasoning == 0 and a_dsml_leaks == 0,
        {
            "turns": multi_turns,
            "tool_call_turns": a_tool_calls,
            "zero_reasoning_turns": a_zero_reasoning,
            "dsml_leak_turns": a_dsml_leaks,
        },
    )

    # 6c. Intra-turn tool loop omitting `reasoning_content` even inside the active turn:
    # must extract 100% valid tool_calls with zero DSML leaks across all steps.
    loop_salt = f"dsml-loop-{uuid.uuid4().hex[:8]}"
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are an engineer auditing carrier files one by one. State one brief sentence, then call `read` on the requested file.",
        },
        {
            "role": "user",
            "content": f"Start by reading `{files[0]}`.",
        },
    ]
    zero_reasoning_turns = 0
    tool_call_turns = 0
    dsml_leak_turns = 0
    for turn_idx in range(multi_turns):
        rep_t = _call_chat(
            base,
            served_model,
            msgs,
            stream=True,
            temperature=0.0,
            tool_choice="auto",
            cache_salt=loop_salt,
        )
        judged = classify_reply(rep_t)
        if len(rep_t["reasoning"]) == 0:
            zero_reasoning_turns += 1
        if judged["any_dsml_failure"]:
            dsml_leak_turns += 1
        if rep_t["calls"]:
            tool_call_turns += 1
            tc_list = [
                {
                    "id": c["id"] or f"call_{turn_idx}_{j}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for j, c in enumerate(rep_t["calls"])
            ]
            msgs.append(
                {
                    "role": "assistant",
                    "content": rep_t["content"],
                    "tool_calls": tc_list,
                }
            )
            next_file = files[(turn_idx + 1) % len(files)]
            for j, tc in enumerate(tc_list):
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"// {files[turn_idx % len(files)]}\nexport const MAX_RETRIES = 3;\n"
                            f"// Next target to inspect with `read`: {next_file}"
                            if j == 0
                            else "ok"
                        ),
                    }
                )
        else:
            msgs.append({"role": "assistant", "content": rep_t["content"]})
            next_file = files[(turn_idx + 1) % len(files)]
            msgs.append({"role": "user", "content": f"Now call `read` on `{next_file}`."})

    log_case(
        "intraturn_omitted_reasoning_resilience",
        tool_call_turns == multi_turns and dsml_leak_turns == 0,
        {
            "turns": multi_turns,
            "tool_call_turns": tool_call_turns,
            "zero_reasoning_turns": zero_reasoning_turns,
            "dsml_leak_turns": dsml_leak_turns,
        },
    )

    all_passed = all(r["passed"] for r in results)
    summary = {
        "event": "matrix_summary",
        "total_cases": len(results),
        "passed_cases": sum(int(r["passed"]) for r in results),
        "all_passed": all_passed,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _run_self_test() -> int:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": "mock-v41"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            req = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
            )
            tc = req.get("tool_choice")
            user_txt = str(req["messages"][-1].get("content", ""))
            if tc == "none":
                calls = []
                content = f"Example: `<{DSML_ATOM} calls>`"
            elif "dsml-notes.md" in user_txt:
                calls = [
                    {
                        "index": 0,
                        "id": "c_write",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": json.dumps(
                                {
                                    "path": "dsml-notes.md",
                                    "content": f"</{DSML_ATOM} parameter>\n</{DSML_ATOM} invoke>\n</{DSML_ATOM} calls>\nEOF",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ]
                content = "Writing notes.\n\n"
            elif isinstance(tc, dict) or "read" in user_txt or req["messages"][-1]["role"] == "tool":
                calls = [
                    {
                        "index": 0,
                        "id": "c_read",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"filePath":"src/carriers/royalmail/index.ts"}',
                        },
                    }
                ]
                content = "Reading file.\n\n"
            else:
                calls = [
                    {
                        "index": 0,
                        "id": "c_bash",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"bun test","timeout":400000}',
                        },
                    }
                ]
                content = "Running tests.\n\n"

            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                ev = {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": [1, 128822, 2],
                            "finish_reason": "tool_calls" if calls else "stop",
                            "delta": {
                                "reasoning_content": "plan",
                                "content": content,
                                **({"tool_calls": calls} if calls else {}),
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
                self.wfile.write(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n".encode())
                return

            payload = {
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [1, 128822, 2],
                        "finish_reason": "tool_calls" if calls else "stop",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "plan",
                            "content": content,
                            **({"tool_calls": calls} if calls else {}),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    with Server(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        summary = run_matrix(base, model_name(base, 5.0), multi_turns=4)
        assert summary["all_passed"] is True, summary
        httpd.shutdown()

    print(json.dumps({"self_test": "passed"}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--multi-turns", type=int, default=12)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args()

    if args.self_test:
        return _run_self_test()
    if not args.url:
        ap.error("URL is required unless --self-test is given")

    base = args.url.rstrip("/")
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
    summary = run_matrix(
        base,
        args.model or model_name(base, 60.0),
        multi_turns=args.multi_turns,
        record=args.record,
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
