#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and bitwise-parity harness for DeepSeek-V4.1 multi-turn chat
rendering, message normalization, and tokenization on Grace ARM64.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import order_tool_results

DEFAULT_MODEL = (
    "/home/chris/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
    "snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
)


def make_tools(n_tools: int = 24) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_action_{i}",
                "description": (
                    f"Execute coding agent operation {i} on workspace files "
                    "with line-range filtering and regex diagnostics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or workspace-relative path",
                        },
                        "content": {
                            "type": "string",
                            "description": "Replacement snippet or search pattern",
                        },
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "dry_run": {"type": "boolean"},
                    },
                    "required": ["file_path"],
                },
            },
        }
        for i in range(n_tools)
    ]


def make_conversation(
    n_turns: int,
    *,
    user_every: int = 50,
    include_control_spelling: bool = False,
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an autonomous coding agent operating on a large "
                "repository. Always inspect files before editing and run "
                "targeted unit tests after each change."
            ),
        },
        {
            "role": "user",
            "content": (
                "Investigate the latency regression in the request pipeline "
                "across 500 tool turns and verify token-level equivalence."
            ),
        },
    ]
    for t in range(n_turns):
        if t > 0 and t % user_every == 0:
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"Continue step {t}: verify edge cases in module_{t % 17}.py "
                        "where `idx < limit` and `vec<int>` templates are parsed."
                    ),
                }
            )
        call_id = f"call_{t:04d}"
        if t % 2 == 0:
            args = json.dumps(
                {
                    "file_path": f"/workspace/src/module_{t % 30}.py",
                    "start_line": t * 2 + 1,
                    "end_line": t * 2 + 60,
                }
            )
        else:
            args = json.dumps(
                {
                    "file_path": f"/workspace/src/module_{t % 30}.py",
                    "content": (
                        f"def compute_{t}(x: int, y: int) -> int:\n"
                        f"    if x < y and y > 0:\n"
                        f"        return (x * {t + 3}) ^ y\n"
                        f"    return x + y\n"
                    ),
                    "dry_run": False,
                }
            )
        msgs.append(
            {
                "role": "assistant",
                "content": "" if t % 5 != 0 else f"Checking module_{t % 30}.py.",
                "reasoning_content": (
                    f"Turn {t}: inspecting `/workspace/src/module_{t % 30}.py` "
                    f"around lines {t * 2 + 1}..{t * 2 + 60} where `a < b`."
                ),
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": f"tool_action_{t % 24}",
                            "arguments": args,
                        },
                    }
                ],
            }
        )
        tool_text = "\n".join(
            f"{t * 2 + line_no}:     val_{line_no} = process_item("
            f"data[{line_no}], threshold={t}) if data[{line_no}] < {t + 10} else 0"
            for line_no in range(1, 13)
        )
        if include_control_spelling and t == 25:
            tool_text += (
                "\n# Parser fixture contains literal <think> and </think> "
                "and ｜DSML｜ markers in user source code"
            )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_text,
            }
        )
    return msgs


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def summarize_ms(samples_s: list[float]) -> dict[str, float]:
    ms = [x * 1e3 for x in samples_s]
    s_ms = sorted(ms)
    p95_idx = min(len(s_ms) - 1, max(0, int(len(s_ms) * 0.95)))
    return {
        "mean_ms": round(statistics.mean(ms), 3),
        "median_ms": round(statistics.median(ms), 3),
        "p95_ms": round(s_ms[p95_idx], 3),
        "min_ms": round(s_ms[0], 3),
        "max_ms": round(s_ms[-1], 3),
    }


def clear_segment_cache(tok: Any) -> None:
    for attr in ("_segment_cache", "_plain_cache"):
        cache = getattr(tok, attr, None)
        if isinstance(cache, dict):
            cache.clear()


def run_scenario(
    tok: Any,
    dummy_model_config: Any,
    tools: list[dict[str, Any]],
    *,
    n_turns: int,
    user_every: int,
    include_control_spelling: bool,
    iters: int,
) -> dict[str, Any]:
    msgs_prev = make_conversation(
        n_turns - 1,
        user_every=user_every,
        include_control_spelling=include_control_spelling,
    )
    msgs = make_conversation(
        n_turns,
        user_every=user_every,
        include_control_spelling=include_control_spelling,
    )
    req = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=msgs,
        tools=tools,
    )
    tools_dicts = [t.model_dump(exclude_none=True) for t in req.tools]

    # Reference single-string encode check when no control spellings in text:
    prompt_str = tok.apply_chat_template(
        req.messages,
        tools=tools_dicts,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert isinstance(prompt_str, str)

    clear_segment_cache(tok)
    token_ids = tok.apply_chat_template(
        req.messages,
        tools=tools_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(token_ids, list)
    if not include_control_spelling:
        direct_ids = tok.encode(prompt_str, add_special_tokens=False)
        assert token_ids == direct_ids, "Mismatch against direct tok.encode!"
    else:
        assert tok.decode(token_ids) == prompt_str

    # Verify caller immutability:
    raw_copy = copy.deepcopy(msgs)
    _ = tok.apply_chat_template(
        msgs,
        tools=tools_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )
    assert msgs == raw_copy, "Caller messages were mutated!"

    # Measure stage timings:
    parse_times: list[float] = []
    render_times: list[float] = []
    cold_e2e_times: list[float] = []
    incr_e2e_times: list[float] = []
    cold_tok_only_times: list[float] = []
    incr_tok_only_times: list[float] = []

    req_prev = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=msgs_prev,
        tools=tools,
    )

    # Warmup:
    _ = tok.apply_chat_template(
        req.messages,
        tools=tools_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )

    for _ in range(iters):
        # 1. Parse + normalize (order_tool_results + parse_chat_messages)
        t0 = time.perf_counter()
        ordered = order_tool_results(req.messages)
        conv, _, _ = parse_chat_messages(
            ordered,
            dummy_model_config,
            content_format="string",
        )
        t1 = time.perf_counter()
        parse_times.append(t1 - t0)

        # 2. Render only (tokenize=False)
        t2 = time.perf_counter()
        rendered = tok.apply_chat_template(
            ordered,
            conversation=conv,
            tools=tools_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )
        t3 = time.perf_counter()
        render_times.append(t3 - t2)
        assert rendered == prompt_str

        # 3. Cold tokenize E2E (clear segment cache first)
        clear_segment_cache(tok)
        t4 = time.perf_counter()
        ordered_c = order_tool_results(req.messages)
        conv_c, _, _ = parse_chat_messages(
            ordered_c,
            dummy_model_config,
            content_format="string",
        )
        t_mid = time.perf_counter()
        ids_cold = tok.apply_chat_template(
            ordered_c,
            conversation=conv_c,
            tools=tools_dicts,
            tokenize=True,
            add_generation_prompt=True,
        )
        t5 = time.perf_counter()
        cold_e2e_times.append(t5 - t4)
        cold_tok_only_times.append((t5 - t_mid) - (t3 - t2))
        assert ids_cold == token_ids

        # 4. Incremental multi-turn E2E (populate cache with turn N-1, then run turn N)
        clear_segment_cache(tok)
        _ = tok.apply_chat_template(
            req_prev.messages,
            tools=tools_dicts,
            tokenize=True,
            add_generation_prompt=True,
        )
        t6 = time.perf_counter()
        ordered_w = order_tool_results(req.messages)
        conv_w, _, _ = parse_chat_messages(
            ordered_w,
            dummy_model_config,
            content_format="string",
        )
        t_mid_w = time.perf_counter()
        ids_warm = tok.apply_chat_template(
            ordered_w,
            conversation=conv_w,
            tools=tools_dicts,
            tokenize=True,
            add_generation_prompt=True,
        )
        t7 = time.perf_counter()
        incr_e2e_times.append(t7 - t6)
        incr_tok_only_times.append((t7 - t_mid_w) - (t3 - t2))
        assert ids_warm == token_ids

    # Peak heap allocation measurement on incremental turn N:
    clear_segment_cache(tok)
    _ = tok.apply_chat_template(
        req_prev.messages,
        tools=tools_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )
    tracemalloc.start()
    ordered_m = order_tool_results(req.messages)
    conv_m, _, _ = parse_chat_messages(
        ordered_m,
        dummy_model_config,
        content_format="string",
    )
    ids_m = tok.apply_chat_template(
        ordered_m,
        conversation=conv_m,
        tools=tools_dicts,
        tokenize=True,
        add_generation_prompt=True,
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert ids_m == token_ids

    return {
        "n_turns": n_turns,
        "num_messages": len(msgs),
        "user_every": user_every,
        "include_control_spelling": include_control_spelling,
        "prompt_chars": len(prompt_str),
        "num_tokens": len(token_ids),
        "tokens_sha256": sha256_ids(token_ids),
        "peak_heap_kib": round(peak_bytes / 1024.0, 2),
        "parse_chat_messages": summarize_ms(parse_times),
        "render_only": summarize_ms(render_times),
        "cold_e2e": summarize_ms(cold_e2e_times),
        "incremental_e2e": summarize_ms(incr_e2e_times),
        "cold_tokenize_est": summarize_ms(cold_tok_only_times),
        "incremental_tokenize_est": summarize_ms(incr_tok_only_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    tok = get_tokenizer(
        args.model, tokenizer_mode="deepseek_v41", trust_remote_code=True
    )
    dummy_model_config = SimpleNamespace(
        multimodal_config=None,
        is_multimodal_model=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
        encoder_config=None,
    )
    tools = make_tools(24)

    scenarios = [
        # (n_turns, user_every, include_control_spelling)
        (100, 50, False),
        (250, 50, False),
        (500, 50, False),
        (100, 1000, False),
        (250, 1000, False),
        (500, 1000, False),
        (250, 50, True),
        (500, 50, True),
    ]

    results = []
    for n_turns, user_every, ctrl in scenarios:
        res = run_scenario(
            tok,
            dummy_model_config,
            tools,
            n_turns=n_turns,
            user_every=user_every,
            include_control_spelling=ctrl,
            iters=args.iters,
        )
        results.append(res)
        print(
            f"[{args.label}] turns={n_turns:3d} user_every={user_every:4d} ctrl={str(ctrl):5s} | "
            f"tokens={res['num_tokens']:6d} sha256={res['tokens_sha256'][:12]} | "
            f"parse={res['parse_chat_messages']['mean_ms']:5.2f}ms "
            f"render={res['render_only']['mean_ms']:6.2f}ms "
            f"cold_e2e={res['cold_e2e']['mean_ms']:6.2f}ms "
            f"incr_e2e={res['incremental_e2e']['mean_ms']:6.2f}ms "
            f"peak_heap={res['peak_heap_kib']:7.1f}KiB",
            flush=True,
        )

    payload = {
        "label": args.label,
        "iters": args.iters,
        "scenarios": results,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
