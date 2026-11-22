"""Target contract suite for DeepSeek V4.1 prompt encoding & DSML parser completeness.

Verifies all seven concrete contracts plus non-regression safety invariants:
  1. Historical assistant turns with omitted `reasoning_content` under `tools`
     render `<｜Assistant｜></think>` instead of OOD `<｜Assistant｜><think></think>`,
     preserving non-empty `reasoning_content`, `prefix: True` continuations, and
     `reasoning_content` when `"reasoning": None` is also present on the dict.
  2. Streamed `assistant.content` ending in `\\n\\n` before `tool_calls` does not
     double into `\\n\\n\\n\\n<｜DSML｜ calls>` on prompt round-trip.
  3. `ParserState.TOOL_PREAMBLE` exits cleanly on `TOOL_END` or non-invoke text
     instead of silently swallowing the remainder of the turn.
  4. `128825`-gated dialect/spacing variants (`<｜DSML｜calls>`, `<｜DSML｜ call>`,
     `<｜DSML｜ tool_calls>`, V4 unspaced `<｜DSML｜tool_calls>`/`<｜DSML｜invoke>`/
     `<｜DSML｜parameter>`, `<｜DSML｜ function_calls>`/`<｜DSML｜ function name="`,
     `<｜DSML｜ invoke name="bash" >`, and mixed spaced/unspaced tags) parse when
     carrying genuine token 128825 and stay literal text when BPE-spelled under
     `tool_choice="auto"`.
  5. Genuine-128825 headless `<｜DSML｜ parameter ...>` blocks:
     - Recover into `ToolCall`s when parameter keys uniquely match one declared
       tool in `request.tools` (Screenshot Turn 1 & Turn 2 `{command, timeout}`
       -> `bash`), when `<｜DSML｜ calls>` wraps headless parameters, when
       parameter 1 substitutes `parameter` for `invoke` (`name="<tool_name>"`),
       across parallel headless calls, inside `<think>`, and on mid-value EOS
       truncation.
     - Fail-closed quarantine (strip malformed genuine-128825 DSML from `content`
       without guessing a tool) when parameter names are ambiguous across tools
       or match no declared tool.
     - Preserve BPE-spelled `｜DSML｜` as literal `content` and preserve quoted
       DSML closers inside string tool arguments.
  6. Genuine-128825 orphan `<｜DSML｜ invoke name="...">` emitted directly inside
     `<think>` without `</think>` or `<｜DSML｜ calls>` implicitly ends reasoning
     and extracts the tool call.
  7. Forced `tool_choice="required"` and named `tool_choice={"type": "function", ...}`
     accept BPE-spelled DSML tool markup after `</think>` (structural-tag grammar
     compatibility) while still keeping BPE-spelled DSML inside `<think>` as
     reasoning prose.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DSML_ATOM = "｜DSML｜"
SCHEDULES = ("nonstream", "one_chunk", "per_token", "text_lag3")

OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
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
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

AMBIGUOUS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TURN1_COMMAND = (
    'echo "=== ukr captcha mapping"; '
    "sed -n '60,75p' src/carriers/ukrposhta/index.ts"
)
TURN1_HEADLESS = (
    f'<｜DSML｜ parameter name="command">{TURN1_COMMAND}</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)

TURN2_REASONING = "Verify pending edits with typecheck and tests before bounding royalmail."
TURN2_PROSE = (
    "More precise: royalmail's 9 MB/lookup is retry churn from an unbounded "
    "lookup. Let me verify my pending edits first, then bound it."
)
TURN2_COMMAND = (
    "cd /Users/herlon/dev/woncadigital/workspace/multitracking && "
    "echo \"typecheck: $(bun run typecheck 2>&1 | grep -cE 'error TS')\"; "
    "bun test src/core/mint.test.ts scripts/bench-5rps.test.ts 2>&1 | tail -3"
)
TURN2_HEADLESS = (
    f'<｜DSML｜ parameter name="command">{TURN2_COMMAND}</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)


@pytest.fixture(scope="module")
def tokenizer():
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshots = sorted(
        hub.glob("models--deepseek-ai--DeepSeek-V4.1-Flash*/snapshots/*")
    )
    for candidate in (
        *(["/models"] if Path("/models/tokenizer.json").is_file() else []),
        *(str(s) for s in snapshots),
        MODEL,
    ):
        try:
            return get_tokenizer(candidate, tokenizer_mode="deepseek_v41")
        except Exception:
            if candidate == MODEL:
                if os.environ.get("HF_HUB_OFFLINE") or os.environ.get(
                    "TRANSFORMERS_OFFLINE"
                ):
                    pytest.skip(f"needs Hugging Face {MODEL} tokenizer (offline)")
                raise


def think_ids(tokenizer) -> tuple[int, int]:
    return (
        tokenizer.convert_tokens_to_ids("<think>"),
        tokenizer.convert_tokens_to_ids("</think>"),
    )


def genuine_ids(tokenizer, text: str) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if DSML_ATOM in text:
        assert tokenizer.convert_tokens_to_ids(DSML_ATOM) in ids
    return ids


def bpe_spelled_ids(tokenizer, text: str) -> list[int]:
    added = set(tokenizer.get_added_vocab().values())
    ids: list[int] = []
    for tid in tokenizer.encode(text, add_special_tokens=False):
        if tid not in added:
            ids.append(tid)
            continue
        for ch in tokenizer.decode([tid]):
            ids.extend(tokenizer.encode(ch, add_special_tokens=False))
    assert added.isdisjoint(ids) and tokenizer.decode(ids) == text
    return ids


def run_parser(
    tokenizer,
    ids: list[int],
    *,
    thinking: bool = True,
    schedule: str = "per_token",
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
) -> tuple[str, str, list[tuple[str, dict]]]:
    active_tools = OPENCODE_TOOLS if tools is None else tools
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    assert cls is not None
    req = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "go"}],
        tools=active_tools,
        tool_choice=tool_choice,
        include_reasoning=True,
        chat_template_kwargs={"thinking": thinking, "reasoning_effort": "high"},
    )
    parser = cls(
        tokenizer,
        active_tools,
        chat_template_kwargs={"thinking": thinking, "reasoning_effort": "high"},
    )
    t_start, t_end = think_ids(tokenizer)
    prompt_ids = [t_start if thinking else t_end]

    if schedule == "nonstream":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, calls = parser.parse(
            tokenizer.decode(ids),
            req,
            enable_auto_tools=True,
            model_output_token_ids=ids,
        )
        return (
            reasoning or "",
            content or "",
            [(c.name, json.loads(c.arguments)) for c in calls or []],
        )

    if schedule == "one_chunk":
        deltas = [(tokenizer.decode(ids), ids)]
    elif schedule == "per_token":
        deltas = []
        tokens, p_off, r_off = None, 0, 0
        for idx in range(len(ids)):
            new_toks, text, p_off, r_off = detokenize_incrementally(
                tokenizer,
                prompt_ids + ids[: idx + 1],
                tokens,
                p_off,
                r_off,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            tokens = new_toks if tokens is None else tokens + new_toks
            deltas.append((text, [ids[idx]]))
    else:
        assert schedule == "text_lag3"
        deltas = []
        sent = 0
        for idx in range(len(ids)):
            text = tokenizer.decode(ids[: idx + 1]).rstrip("\ufffd")[:-3]
            deltas.append((text[sent:], [ids[idx]]))
            sent = max(sent, len(text))
        deltas.append((tokenizer.decode(ids)[sent:], []))

    reasoning, content = "", ""
    merged: dict[int, list[str]] = {}
    for idx, (d_text, d_ids) in enumerate(deltas):
        delta = parser.parse_delta(
            d_text,
            d_ids,
            req,
            prompt_token_ids=prompt_token_ids if (prompt_token_ids := prompt_ids) else None,
            finished=(idx == len(deltas) - 1),
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            slot = merged.setdefault(call.index, ["", ""])
            slot[0] += call.function.name or ""
            slot[1] += call.function.arguments or ""
    return (
        reasoning,
        content,
        [(name, json.loads(args)) for _, (name, args) in sorted(merged.items())],
    )


# ── Contract 1: Historical Empty `<think></think>` & `reasoning` Field ────────


def test_historical_tool_turns_without_reasoning_render_closed_think(tokenizer):
    messages = [
        {"role": "user", "content": "Audit royalmail."},
        {
            "role": "assistant",
            "content": "Reading file.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": '{"filePath":"src/a.ts"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "const x = 1;"},
        {
            "role": "assistant",
            "reasoning": None,
            "reasoning_content": "Now run tests.",
            "content": "Running tests.",
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"bun test"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": "1 passed"},
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tools=OPENCODE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="high",
    )

    assert "<think></think>" not in rendered
    assert "<｜Assistant｜></think>Reading file." in rendered
    assert "<｜Assistant｜><think>Now run tests.</think>Running tests." in rendered
    assert rendered.endswith("<｜Assistant｜><think>")


def test_final_assistant_prefix_continuation_still_preserves_reference_contract(
    tokenizer,
):
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Write a haiku."},
            {"role": "assistant", "content": "Autumn", "prefix": True},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered.endswith("<｜Assistant｜><think></think>Autumn")


# ── Contract 2: Streamed `\n\n` Tool-Separator Round-Trip Deduplication ──────


def test_streamed_double_newline_before_tool_calls_is_not_quadrupled_on_roundtrip(
    tokenizer,
):
    messages = [
        {"role": "user", "content": "Run tests."},
        {
            "role": "assistant",
            "reasoning_content": "Check tests.",
            "content": "Checking.\n\n",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"bun test"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=OPENCODE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "Checking.\n\n<｜DSML｜ calls>" in rendered
    assert "Checking.\n\n\n\n<｜DSML｜ calls>" not in rendered


# ── Contract 3: `TOOL_PREAMBLE` Silent Black-Hole Elimination ────────────────


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_empty_calls_block_does_not_swallow_following_prose(tokenizer, schedule: str):
    _, t_end = think_ids(tokenizer)
    ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Before. ")
        + genuine_ids(tokenizer, "<｜DSML｜ calls>\n</｜DSML｜ calls>")
        + bpe_spelled_ids(tokenizer, " After.")
    )
    reasoning, content, calls = run_parser(tokenizer, ids, schedule=schedule)
    assert reasoning == "Plan."
    assert calls == []
    assert "Before." in content and "After." in content


# ── Contract 4: V4 / V4.1 Dialect & Spacing Variants (`128825`-Gated) ────────


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "variant_call",
    [
        # Unspaced V4.1 `<｜DSML｜calls>`
        '<｜DSML｜calls>\n<｜DSML｜ invoke name="bash">\n'
        '<｜DSML｜ parameter name="command" string="true">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜calls>",
        # Spaced `<｜DSML｜ tool_calls>`
        '<｜DSML｜ tool_calls>\n<｜DSML｜ invoke name="bash">\n'
        '<｜DSML｜ parameter name="command" string="true">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ tool_calls>",
        # Full V4 unspaced tags
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="command" string="true">bun test</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        # Legacy `function_calls` + `function name=` + space before `>`
        '<｜DSML｜ function_calls>\n<｜DSML｜ function name="bash" >\n'
        '<｜DSML｜ parameter name="command">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ function>\n</｜DSML｜ function_calls>",
        # Mixed V4.1 spaced `<｜DSML｜ calls>` + V4 unspaced `<｜DSML｜invoke>`
        '<｜DSML｜ calls>\n<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜ parameter name="command">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ],
    ids=[
        "unspaced_calls",
        "spaced_tool_calls",
        "v4_unspaced_full",
        "legacy_function_calls_space_before_gt",
        "mixed_calls_unspaced_invoke",
    ],
)
def test_dsml_spacing_variants_parse_with_genuine_128825_and_stay_text_when_bpe(
    tokenizer, schedule: str, variant_call: str
):
    _, t_end = think_ids(tokenizer)
    genuine = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Running:\n\n")
        + genuine_ids(tokenizer, variant_call)
    )
    reasoning, content, calls = run_parser(tokenizer, genuine, schedule=schedule)
    assert reasoning == "Plan."
    assert content == "Running:\n\n"
    assert calls == [("bash", {"command": "bun test"})]

    spelled = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Example:\n\n" + variant_call)
    )
    _, s_content, s_calls = run_parser(tokenizer, spelled, schedule=schedule)
    assert s_calls == []
    assert s_content == "Example:\n\n" + variant_call


# ── Contract 5: Screenshot Headless `<｜DSML｜ parameter>` Recovery & Safety ─


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_screenshot_turn1_and_turn2_recover_clean_bash_tool_calls(
    tokenizer, schedule: str
):
    _, t_end = think_ids(tokenizer)

    t1_ids = (
        bpe_spelled_ids(tokenizer, "Check ukrposhta.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Checking ukrposhta:\n\n")
        + genuine_ids(tokenizer, TURN1_HEADLESS)
    )
    r1, c1, calls1 = run_parser(tokenizer, t1_ids, schedule=schedule)
    assert r1 == "Check ukrposhta."
    assert c1 == "Checking ukrposhta:\n\n"
    assert calls1 == [("bash", {"command": TURN1_COMMAND, "timeout": 400000})]

    t2_ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [t_end]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + genuine_ids(tokenizer, TURN2_HEADLESS)
    )
    r2, c2, calls2 = run_parser(tokenizer, t2_ids, schedule=schedule)
    assert r2 == TURN2_REASONING
    assert c2 == TURN2_PROSE + "\n\n"
    assert calls2 == [("bash", {"command": TURN2_COMMAND, "timeout": 400000})]


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "headless_variant",
    [
        '<｜DSML｜ parameter name="bash">\n'
        '<｜DSML｜ parameter name="command">bun run typecheck</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ parameter name="command">bun run typecheck</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    ],
    ids=["param_substituted_for_invoke", "calls_wrapper_without_invoke"],
)
def test_headless_parameter_variants_recover_tool_call(
    tokenizer, schedule: str, headless_variant: str
):
    _, t_end = think_ids(tokenizer)
    ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Checking.\n\n")
        + genuine_ids(tokenizer, headless_variant)
    )
    reasoning, content, calls = run_parser(tokenizer, ids, schedule=schedule)
    assert reasoning == "Plan."
    assert content == "Checking.\n\n"
    assert calls == [("bash", {"command": "bun run typecheck", "timeout": 400000})]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_parallel_and_in_think_and_truncated_headless_parameters_recover(
    tokenizer, schedule: str
):
    _, t_end = think_ids(tokenizer)

    # 1. Two parallel headless calls in one turn (`bash` + `read`)
    parallel_headless = (
        '<｜DSML｜ parameter name="command">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        '<｜DSML｜ parameter name="filePath">src/a.ts</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    p_ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Both:\n\n")
        + genuine_ids(tokenizer, parallel_headless)
    )
    pr, pc, pcalls = run_parser(tokenizer, p_ids, schedule=schedule)
    assert pr == "Plan." and pc == "Both:\n\n"
    assert pcalls == [
        ("bash", {"command": "bun test"}),
        ("read", {"filePath": "src/a.ts"}),
    ]

    # 2. Headless parameter emitted directly inside `<think>` (omitting `</think>`)
    in_think_ids = bpe_spelled_ids(tokenizer, "Fast plan.") + genuine_ids(
        tokenizer,
        '<｜DSML｜ parameter name="filePath">src/b.ts</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    )
    tr, tc, tcalls = run_parser(tokenizer, in_think_ids, thinking=True, schedule=schedule)
    assert tr == "Fast plan." and tc == ""
    assert tcalls == [("read", {"filePath": "src/b.ts"})]

    # 3. Mid-value EOS truncation on a headless parameter
    trunc_ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + genuine_ids(tokenizer, '<｜DSML｜ parameter name="command">echo partial')
    )
    _, x_c, x_calls = run_parser(tokenizer, trunc_ids, schedule=schedule)
    assert x_c == ""
    assert x_calls == [("bash", {"command": "echo partial"})]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_ambiguous_or_unknown_headless_parameter_quarantines_without_content_leak(
    tokenizer, schedule: str
):
    _, t_end = think_ids(tokenizer)
    ambiguous_block = (
        '<｜DSML｜ parameter name="query" string="true">royalmail</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Let me search.\n\n")
        + genuine_ids(tokenizer, ambiguous_block)
    )
    reasoning, content, calls = run_parser(
        tokenizer, ids, schedule=schedule, tools=AMBIGUOUS_TOOLS
    )
    assert reasoning == "Plan."
    assert calls == []
    assert content == "Let me search.\n\n"
    assert DSML_ATOM not in content


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_bpe_spelled_headless_and_quoted_dsml_in_string_args_stay_safe(
    tokenizer, schedule: str
):
    _, t_end = think_ids(tokenizer)

    bpe_ids = (
        bpe_spelled_ids(tokenizer, "Plan.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, "Quoted:\n\n" + TURN2_HEADLESS)
    )
    _, c_bpe, calls_bpe = run_parser(tokenizer, bpe_ids, schedule=schedule)
    assert calls_bpe == []
    assert c_bpe == "Quoted:\n\n" + TURN2_HEADLESS

    quoted_body = (
        "line1 </｜DSML｜ parameter>\n"
        "line2 </｜DSML｜ invoke>\n"
        "line3 </｜DSML｜ calls>\nend"
    )
    write_call = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="write">\n'
        '<｜DSML｜ parameter name="path" string="true">notes.md</｜DSML｜ parameter>\n'
        f'<｜DSML｜ parameter name="content" string="true">{quoted_body}</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    w_ids = bpe_spelled_ids(tokenizer, "Plan.") + [t_end] + genuine_ids(tokenizer, write_call)
    _, c_w, calls_w = run_parser(tokenizer, w_ids, schedule=schedule)
    assert c_w == ""
    assert calls_w == [("write", {"path": "notes.md", "content": quoted_body})]


# ── Contract 6: Orphan `<｜DSML｜ invoke>` Inside `<think>` Without `</think>` ─


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_orphan_invoke_inside_reasoning_implicitly_ends_thinking(
    tokenizer, schedule: str
):
    ids = bpe_spelled_ids(tokenizer, "Plan directly.") + genuine_ids(
        tokenizer,
        '<｜DSML｜ invoke name="bash">\n'
        '<｜DSML｜ parameter name="command" string="true">bun test</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    )
    reasoning, content, calls = run_parser(tokenizer, ids, thinking=True, schedule=schedule)
    assert reasoning == "Plan directly."
    assert content == ""
    assert calls == [("bash", {"command": "bun test"})]


# ── Contract 7: Forced / Named `tool_choice` BPE-Spelled Grammar Output ──────


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "forced_choice",
    ["required", {"type": "function", "function": {"name": "bash"}}],
    ids=["required", "named_bash"],
)
def test_forced_and_named_tool_choice_accept_bpe_spelled_dsml_after_think(
    tokenizer, schedule: str, forced_choice: Any
):
    """Under `tool_choice="required"` or named function `tool_choice`, string-level
    structural tags may spell `｜DSML｜` out of ordinary BPE tokens after `</think>`.
    BPE-spelled DSML inside `<think>` must remain reasoning prose, while the forced
    post-`</think>` DSML block must parse into `tool_calls`."""
    _, t_end = think_ids(tokenizer)
    forced_call = (
        '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
        '<｜DSML｜ parameter name="command" string="true">bun test</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="timeout" string="false">400000</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    ids = (
        bpe_spelled_ids(tokenizer, "Mention <｜DSML｜ calls> in reasoning.")
        + [t_end]
        + bpe_spelled_ids(tokenizer, forced_call)
    )
    reasoning, content, calls = run_parser(
        tokenizer, ids, thinking=True, schedule=schedule, tool_choice=forced_choice
    )
    assert reasoning == "Mention <｜DSML｜ calls> in reasoning."
    assert content == ""
    assert calls == [("bash", {"command": "bun test", "timeout": 400000})]
