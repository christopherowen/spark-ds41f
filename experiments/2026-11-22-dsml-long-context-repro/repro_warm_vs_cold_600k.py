#!/usr/bin/env python3
"""Incremental multi-turn warm-cache vs cold-recompute 600K DSML & SWA parity probe.

usage:
  repro_warm_vs_cold_600k.py URL [--record FILE] [--target-tokens 590000]
                                 [--turn-tokens 16000] [--sampled-trials 8]
  repro_warm_vs_cold_600k.py --self-test

Tests the three multi-turn mechanisms surfaced during the long-context investigation
that single-shot cold prefill misses:
  1. Incremental KV + SWABoundedReplay growth across 35-40 real cached turns under one
     `cache_salt` (each turn generates a live tool call on the engine, then appends
     the streamed `assistant.content` — including its trailing `\\n\\n` — with
     `reasoning_content` omitted, plus a multi-thousand-token `tool` output).
  2. Exact Warm-vs-Cold greedy parity (`temperature=0`, `logprobs=True`) at intermediate
     and ~600K checkpoints: sends the identical accumulated `messages` array once with
     the warm `session_salt` (`cached_tokens` ~ 97%+) and once with a fresh `cold_salt`
     (`cached_tokens == 0`), checking token-ID identity and max logprob divergence to
     isolate SWA / hybrid-KV cache drift from pure
     prompt OOD (`<think></think>` + `\\n\\n\\n\\n`).
  3. Sampled (`temperature=1.0`, `stream=True`) burst at the ~600K warm frontier to
     measure headless DSML leaks, `0`-char immediate-`</think>` skips,
     missing `string=` attributes, and premature prose stops.

Stdlib only.
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

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repro_live_long_dsml import (  # noqa: E402
    CARRIERS,
    DSML_ATOM,
    DSML_TOKEN_ID,
    OPENCODE_TOOLS,
    THINK_END_ID,
    _approx_tokens,
    _render_carrier_dump,
    classify_reply,
    model_name,
)


def send_probe_request(
    base: str,
    model: str,
    messages: list[dict],
    *,
    stream: bool,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    cache_salt: str,
    logprobs: bool = False,
    timeout: float = 1800.0,
) -> dict:
    body: dict = {
        "model": model,
        "messages": messages,
        "tools": OPENCODE_TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
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
            lp_items = (choice.get("logprobs") or {}).get("content") or []
            usage = payload.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "calls": [
                    {
                        "id": tc.get("id") or f"call_{idx}",
                        "name": tc["function"].get("name") or "",
                        "arguments": tc["function"].get("arguments") or "",
                    }
                    for idx, tc in enumerate(msg.get("tool_calls") or [])
                ],
                "token_ids": choice.get("token_ids") or [],
                "logprobs": [item.get("logprob") for item in lp_items],
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "cached_tokens": cached,
            }

        content, reasoning, finish_reason = "", "", None
        usage = {}
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
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
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
            "logprobs": [],
            "finish_reason": finish_reason,
            "usage": usage,
            "cached_tokens": cached,
        }


def compare_warm_vs_cold(warm: dict, cold: dict) -> dict:
    w_ids = warm["token_ids"]
    c_ids = cold["token_ids"]
    first_diff = next(
        (i for i in range(min(len(w_ids), len(c_ids))) if w_ids[i] != c_ids[i]),
        None if len(w_ids) == len(c_ids) else min(len(w_ids), len(c_ids)),
    )
    prefix_len = first_diff if first_diff is not None else len(w_ids)
    w_lp = warm.get("logprobs") or []
    c_lp = cold.get("logprobs") or []
    shared_lp = min(prefix_len, len(w_lp), len(c_lp))
    max_lp_diff = (
        max(
            abs(float(w_lp[i]) - float(c_lp[i]))
            for i in range(shared_lp)
            if w_lp[i] is not None and c_lp[i] is not None
        )
        if shared_lp > 0
        else 0.0
    )
    return {
        "tokens_identical": w_ids == c_ids,
        "first_token_diff": first_diff,
        "warm_len": len(w_ids),
        "cold_len": len(c_ids),
        "shared_prefix_tokens": prefix_len,
        "max_abs_logprob_diff_on_shared_prefix": round(max_lp_diff, 6),
        "warm_cached_tokens": warm["cached_tokens"],
        "cold_cached_tokens": cold["cached_tokens"],
        "warm_verdict": classify_reply(warm)["verdict"],
        "cold_verdict": classify_reply(cold)["verdict"],
        "warm_reasoning_chars": len(warm["reasoning"]),
        "cold_reasoning_chars": len(cold["reasoning"]),
    }


def _run_self_test() -> int:
    calls_seen: list[dict] = []

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
            calls_seen.append(req)
            is_warm = str(req.get("cache_salt", "")).startswith("warm")
            tids = [10, THINK_END_ID, 20, DSML_TOKEN_ID, 30]
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                ev = {
                    "choices": [
                        {
                            "index": 0,
                            "token_ids": tids,
                            "finish_reason": "tool_calls",
                            "delta": {
                                "reasoning_content": "plan",
                                "content": "Checking next carrier.\n\n",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_m1",
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"ls","timeout":400000}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20000,
                        "completion_tokens": 5,
                        "prompt_tokens_details": {
                            "cached_tokens": 16000 if is_warm else 0
                        },
                    },
                }
                self.wfile.write(f"data: {json.dumps(ev)}\n\ndata: [DONE]\n\n".encode())
                return

            payload = {
                "choices": [
                    {
                        "index": 0,
                        "token_ids": tids,
                        "finish_reason": "tool_calls",
                        "logprobs": {
                            "content": [{"logprob": -0.125} for _ in tids]
                        },
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "plan",
                            "content": "Checking next carrier.\n\n",
                            "tool_calls": [
                                {
                                    "id": "call_m1",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": '{"command":"ls","timeout":400000}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20000,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {
                        "cached_tokens": 16000 if is_warm else 0
                    },
                },
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
        m = model_name(base, 5.0)
        msgs = [{"role": "user", "content": "audit carriers"}]
        w = send_probe_request(
            base,
            m,
            msgs,
            stream=False,
            temperature=0.0,
            max_tokens=256,
            reasoning_effort="high",
            cache_salt="warm-1",
            logprobs=True,
        )
        c = send_probe_request(
            base,
            m,
            msgs,
            stream=False,
            temperature=0.0,
            max_tokens=256,
            reasoning_effort="high",
            cache_salt="cold-1",
            logprobs=True,
        )
        cmp = compare_warm_vs_cold(w, c)
        assert cmp["tokens_identical"] is True
        assert cmp["warm_cached_tokens"] == 16000
        assert cmp["cold_cached_tokens"] == 0
        httpd.shutdown()

    print(json.dumps({"self_test": "passed"}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--target-tokens", type=int, default=590000)
    ap.add_argument(
        "--turn-tokens",
        type=int,
        default=16500,
        help="Approximate tool-output tokens appended per live cached turn",
    )
    ap.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[65536, 262144, 580000],
        help="Token milestones at which to run Warm-vs-Cold greedy parity checks",
    )
    ap.add_argument("--sampled-trials", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--record", type=Path)
    args = ap.parse_args()

    if args.self_test:
        return _run_self_test()
    if not args.url:
        ap.error("URL is required unless --self-test is given")

    base = args.url.rstrip("/")
    served_model = args.model or model_name(base, 60.0)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    session_salt = f"warm-session-{uuid.uuid4().hex[:10]}"
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are OpenCode (Build mode) working in "
                "`/Users/herlon/dev/woncadigital/workspace/multitracking`. "
                "State a brief 1-sentence status note before invoking `bash` or `read`."
            ),
        },
        {
            "role": "user",
            "content": (
                "Audit memory per lookup across all carrier adapters in `src/carriers/`, "
                "inspect retry payloads step by step using `bash` or `read`, and verify "
                "`bun run typecheck` + `bun test src/core/mint.test.ts scripts/bench-5rps.test.ts`."
            ),
        },
    ]

    pending_checkpoints = sorted(args.checkpoints)
    parity_results: list[dict] = []
    turn_summaries: list[dict] = []
    prompt_tokens_now = 0
    step = 0

    while prompt_tokens_now < args.target_tokens:
        step += 1
        t0 = time.monotonic()
        # Stream each live turn at temperature=1.0 just like OpenCode, so real generated
        # KV + SWA blocks accumulate incrementally and `content` retains trailing `\n\n`.
        reply = send_probe_request(
            base,
            served_model,
            messages,
            stream=True,
            temperature=1.0,
            max_tokens=args.max_tokens,
            reasoning_effort="high",
            cache_salt=session_salt,
        )
        elapsed = round(time.monotonic() - t0, 3)
        judged = classify_reply(reply)
        prompt_tokens_now = reply["usage"].get("prompt_tokens") or (
            prompt_tokens_now + args.turn_tokens
        )

        step_row = {
            "event": "live_turn",
            "step": step,
            "prompt_tokens": prompt_tokens_now,
            "cached_tokens": reply["cached_tokens"],
            "completion_tokens": reply["usage"].get("completion_tokens"),
            "reasoning_chars": len(reply["reasoning"]),
            "trailing_double_newline": reply["content"].endswith("\n\n"),
            "verdict": judged["verdict"],
            "replicated_screenshot_error": judged["replicated_screenshot_error"],
            "any_dsml_failure": judged["any_dsml_failure"],
            "seconds": elapsed,
        }
        turn_summaries.append(step_row)
        print(json.dumps(step_row, ensure_ascii=False), flush=True)
        if args.record:
            with args.record.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({**step_row, "reply": reply}, ensure_ascii=False) + "\n"
                )

        # Append the turn into history EXACTLY as OpenCode's OpenAI-compatible client does:
        # `reasoning_content` omitted, `content` echoed verbatim (including trailing `\n\n`).
        carrier = CARRIERS[(step - 1) % len(CARRIERS)]
        # Vary chunk lengths by non-multiples of 256 so unaligned block/SWA boundaries are exercised.
        jitter_tokens = args.turn_tokens + rng.randint(-1900, 1900)
        dump_lines = max(24, int((jitter_tokens * 3.01) / 145))
        dump_body = _render_carrier_dump(rng, carrier, step, dump_lines)

        if reply["calls"]:
            tc_list = [
                {
                    "id": c["id"] or f"call_{step}_{idx}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for idx, c in enumerate(reply["calls"])
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": reply["content"],
                    "tool_calls": tc_list,
                }
            )
            for idx, tc in enumerate(tc_list):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            dump_body
                            if idx == 0
                            else f"(completed auxiliary command {idx + 1})"
                        ),
                    }
                )
            # Nudge next step without coaching DSML syntax.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"General finished · Bank {carrier} batch {step}. "
                        "Continue auditing the next carrier and verify pending edits with tests."
                    ),
                }
            )
        else:
            # If the turn produced prose or leaked DSML without tool_calls, echo `content`
            # verbatim and deliver the next subagent wakeup!
            messages.append({"role": "assistant", "content": reply["content"]})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"General finished · Bank {carrier} batch {step}.\n{dump_body}\n"
                        "Continue verification and run the pending test checks."
                    ),
                }
            )

        est_next = prompt_tokens_now + _approx_tokens(dump_body) + 180
        while pending_checkpoints and est_next >= pending_checkpoints[0]:
            milestone = pending_checkpoints.pop(0)
            t_w = time.monotonic()
            warm_res = send_probe_request(
                base,
                served_model,
                messages,
                stream=False,
                temperature=0.0,
                max_tokens=args.max_tokens,
                reasoning_effort="high",
                cache_salt=session_salt,
                logprobs=True,
            )
            warm_s = round(time.monotonic() - t_w, 3)

            t_c = time.monotonic()
            cold_res = send_probe_request(
                base,
                served_model,
                messages,
                stream=False,
                temperature=0.0,
                max_tokens=args.max_tokens,
                reasoning_effort="high",
                cache_salt=f"cold-{milestone}-{uuid.uuid4().hex[:8]}",
                logprobs=True,
            )
            cold_s = round(time.monotonic() - t_c, 3)

            cmp = compare_warm_vs_cold(warm_res, cold_res)
            p_row = {
                "event": "warm_vs_cold_parity",
                "milestone": milestone,
                "prompt_tokens": warm_res["usage"].get("prompt_tokens"),
                **cmp,
                "warm_seconds": warm_s,
                "cold_seconds": cold_s,
            }
            parity_results.append(p_row)
            print(json.dumps(p_row, ensure_ascii=False), flush=True)
            if args.record:
                with args.record.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {**p_row, "warm": warm_res, "cold": cold_res},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            prompt_tokens_now = warm_res["usage"].get("prompt_tokens") or est_next

    # Final sampled burst at the warm ~600K frontier
    sampled_verdicts: dict[str, int] = {}
    zero_think_count = 0
    for trial in range(1, args.sampled_trials + 1):
        t_s = time.monotonic()
        s_rep = send_probe_request(
            base,
            served_model,
            messages,
            stream=True,
            temperature=1.0,
            max_tokens=args.max_tokens,
            reasoning_effort="high",
            cache_salt=session_salt,
        )
        s_judged = classify_reply(s_rep)
        v = s_judged["verdict"]
        sampled_verdicts[v] = sampled_verdicts.get(v, 0) + 1
        zero_think_count += int(len(s_rep["reasoning"]) == 0)
        s_row = {
            "event": "warm_frontier_sampled",
            "trial": trial,
            "prompt_tokens": s_rep["usage"].get("prompt_tokens"),
            "cached_tokens": s_rep["cached_tokens"],
            "verdict": v,
            "replicated_screenshot_error": s_judged["replicated_screenshot_error"],
            "any_dsml_failure": s_judged["any_dsml_failure"],
            "reasoning_chars": len(s_rep["reasoning"]),
            "content_preview": s_rep["content"][:320],
            "seconds": round(time.monotonic() - t_s, 3),
        }
        print(json.dumps(s_row, ensure_ascii=False), flush=True)
        if args.record:
            with args.record.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({**s_row, "reply": s_rep}, ensure_ascii=False) + "\n"
                )

    summary = {
        "event": "summary",
        "model": served_model,
        "live_turns": len(turn_summaries),
        "live_turn_dsml_failures": sum(
            int(r["any_dsml_failure"]) for r in turn_summaries
        ),
        "live_turn_zero_think": sum(
            int(r["reasoning_chars"] == 0) for r in turn_summaries
        ),
        "parity_checks": parity_results,
        "all_parity_identical": all(p["tokens_identical"] for p in parity_results),
        "frontier_sampled_verdicts": sampled_verdicts,
        "frontier_zero_think_count": zero_think_count,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
