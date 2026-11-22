"""Replicate the long-context OpenCode DSML control-token failure on spark-ds41f:1120.

Covers four coupled layers behind the screenshot at ~597.8K context:
1. Exact Turn 1 and Turn 2 parser reproduction (whole + 3 streaming schedules,
   both genuine token 128825 and BPE-spelled DSML): `<think>...</think>` parses
   into `reasoning`, while a headless `<｜DSML｜ parameter ...>` block after prose
   produces `tool_calls == []` and leaks verbatim DSML tags into `content`.
2. State-machine reachability and V4/V4.1 header-drift matrix:
   - Missing `string="true"` / `string="false"` attributes parse cleanly when the
     spaced `<｜DSML｜ invoke name="bash">` header is present (even without
     `<｜DSML｜ calls>`).
   - No transition out of any tool state returns to `ParserState.CONTENT`, proving
     the screenshot's leaked `<｜DSML｜ parameter ...>` was emitted directly from
     `ParserState.CONTENT` without `<｜DSML｜ calls>` or `<｜DSML｜ invoke>`.
   - Emitting `<｜DSML｜ calls>` (token 128825) followed by an unspaced V4
     `<｜DSML｜invoke name="bash">` or BPE-spelled `<｜DSML｜ invoke name="bash">`
     traps the parser in `ParserState.TOOL_PREAMBLE` and silently drops the
     entire remainder of the assistant turn (`tool_calls == []`, no DSML in
     `content`).
3. Multi-turn prompt-encoding reproduction (Turn 1 -> Turn 2 cascade):
   - Leaked DSML in Turn 1 `assistant.content` is stripped of token 128825 by
     `_ControlSpellings` and encoded as plain BPE pieces right inside the final
     128-token sliding-window attention span.
   - Non-empty `tools` forces `effective_drop_thinking = False`, injecting empty
     `<think></think>` blocks into every historical assistant tool turn whose
     `reasoning_content` was omitted by the client.
4. Token-provenance rejection when Turn 2 copies or imitates BPE-spelled DSML
   from Turn 1's poisoned `assistant.content`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DSML_ATOM = "｜DSML｜"
SCHEDULES = ("nonstream", "one_chunk", "per_token", "text_lag3")

# OpenCode `bash` tool schema matching the screenshot's `command` + `timeout` call.
OPENCODE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes a given bash command in a persistent shell session with optional timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds",
                    },
                    "description": {
                        "type": "string",
                        "description": "Clear, concise description of what this command does",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Reads a file from the local filesystem.",
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
]

# Exact visible text from Turn 1 (top of screenshot) and Turn 2 (bottom of screenshot).
TURN1_PROSE = "Let me check the ukrposhta captcha mapping lines directly.\n\n"
TURN1_COMMAND = (
    'echo "=== ukr captcha mapping"; '
    "sed -n '60,75p' src/carriers/ukrposhta/index.ts"
)
TURN1_HEADLESS_DSML = (
    f'<｜DSML｜ parameter name="command">{TURN1_COMMAND}</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)

TURN2_REASONING = (
    "The ukrposhta subagent finished. Looking closely at royalmail's 9 MB per lookup, "
    "that comes from retry churn on an unbounded lookup. I should run typecheck and the "
    "mint and bench-5rps tests before bounding it."
)
TURN2_PROSE = (
    "More precise: royalmail's 9 MB/lookup is retry churn from an unbounded "
    "lookup. Let me verify my pending edits first, then bound it."
)
TURN2_COMMAND = (
    "cd /Users/herlon/dev/woncadigital/workspace/multitracking && "
    "echo \"typecheck: $(bun run typecheck 2>&1 | grep -cE 'error TS')\"; "
    "bun test src/core/mint.test.ts scripts/bench-5rps.test.ts 2>&1 | tail -3"
)
TURN2_HEADLESS_DSML = (
    f'<｜DSML｜ parameter name="command">{TURN2_COMMAND}</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n"
    "</｜DSML｜ calls>"
)
TURN2_EXPECTED_LEAKED_CONTENT = f"{TURN2_PROSE}\n\n{TURN2_HEADLESS_DSML}"


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


def dsml_token_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids(DSML_ATOM)
    assert isinstance(tid, int) and tid == 128825
    return tid


def think_ids(tokenizer) -> tuple[int, int]:
    start_id = tokenizer.convert_tokens_to_ids("<think>")
    end_id = tokenizer.convert_tokens_to_ids("</think>")
    assert isinstance(start_id, int) and isinstance(end_id, int)
    return start_id, end_id


def genuine_ids(tokenizer, text: str) -> list[int]:
    """Encode *text* with real added tokens (including 128825 for `｜DSML｜`)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if DSML_ATOM in text:
        assert dsml_token_id(tokenizer) in ids
    return ids


def bpe_spelled_ids(tokenizer, text: str) -> list[int]:
    """Encode *text* using ordinary BPE pieces only (no added tokens such as 128825)."""
    added = set(tokenizer.get_added_vocab().values())
    ids: list[int] = []
    for token_id in tokenizer.encode(text, add_special_tokens=False):
        if token_id not in added:
            ids.append(token_id)
            continue
        for ch in tokenizer.decode([token_id]):
            ids.extend(tokenizer.encode(ch, add_special_tokens=False))
    assert added.isdisjoint(ids)
    assert tokenizer.decode(ids) == text
    return ids


def make_serving_parser(tokenizer, *, thinking: bool = True):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    assert cls is not None
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "check royalmail lookup churn"}],
        tools=OPENCODE_TOOLS,
        include_reasoning=True,
        chat_template_kwargs={"thinking": thinking, "reasoning_effort": "high"},
    )
    parser = cls(
        tokenizer,
        OPENCODE_TOOLS,
        chat_template_kwargs={"thinking": thinking, "reasoning_effort": "high"},
    )
    think_start, think_end = think_ids(tokenizer)
    prompt_ids = [think_start if thinking else think_end]
    return parser, request, prompt_ids


def schedule_deltas(
    tokenizer, ids: list[int], prompt_ids: list[int], schedule: str
) -> list[tuple[str, list[int]]]:
    if schedule == "one_chunk":
        return [(tokenizer.decode(ids), ids)]
    if schedule == "per_token":
        deltas: list[tuple[str, list[int]]] = []
        tokens, prefix_offset, read_offset = None, 0, 0
        for index in range(len(ids)):
            new_tokens, text, prefix_offset, read_offset = detokenize_incrementally(
                tokenizer,
                prompt_ids + ids[: index + 1],
                tokens,
                prefix_offset,
                read_offset,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            tokens = new_tokens if tokens is None else tokens + new_tokens
            deltas.append((text, [ids[index]]))
        return deltas
    assert schedule == "text_lag3"
    deltas = []
    sent = 0
    for index in range(len(ids)):
        text = tokenizer.decode(ids[: index + 1]).rstrip("\ufffd")[:-3]
        deltas.append((text[sent:], [ids[index]]))
        sent = max(sent, len(text))
    deltas.append((tokenizer.decode(ids)[sent:], []))
    return deltas


def run_serving_parser(
    tokenizer, ids: list[int], *, thinking: bool, schedule: str
) -> tuple[str, str, list[tuple[str, dict]]]:
    parser, request, prompt_ids = make_serving_parser(tokenizer, thinking=thinking)
    if schedule == "nonstream":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, calls = parser.parse(
            tokenizer.decode(ids),
            request,
            enable_auto_tools=True,
            model_output_token_ids=ids,
        )
        return (
            reasoning or "",
            content or "",
            [(call.name, json.loads(call.arguments)) for call in calls or []],
        )

    deltas = schedule_deltas(tokenizer, ids, prompt_ids, schedule)
    reasoning, content = "", ""
    calls_by_index: dict[int, list[str]] = {}
    for index, (delta_text, delta_ids) in enumerate(deltas):
        delta = parser.parse_delta(
            delta_text,
            delta_ids,
            request,
            prompt_token_ids=prompt_ids,
            finished=(index == len(deltas) - 1),
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            slot = calls_by_index.setdefault(call.index, ["", ""])
            slot[0] += call.function.name or ""
            slot[1] += call.function.arguments or ""
    return (
        reasoning,
        content,
        [
            (name, json.loads(arguments))
            for _, (name, arguments) in sorted(calls_by_index.items())
        ],
    )


# ── 1. Exact Screenshot Turn 1 & Turn 2 Reproduction ─────────────────────────


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("genuine_dsml", [True, False], ids=["atom_128825", "bpe_text"])
def test_screenshot_turn2_headless_dsml_replicates_exact_leak(
    tokenizer, schedule: str, genuine_dsml: bool
):
    """Replicate Turn 2 of the screenshot verbatim across whole and streaming paths."""
    _, think_end_id = think_ids(tokenizer)
    encode_dsml = genuine_ids if genuine_dsml else bpe_spelled_ids
    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + encode_dsml(tokenizer, TURN2_HEADLESS_DSML)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert calls == []
    assert content == TURN2_EXPECTED_LEAKED_CONTENT


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("genuine_dsml", [True, False], ids=["atom_128825", "bpe_text"])
def test_screenshot_turn1_headless_dsml_replicates_exact_leak(
    tokenizer, schedule: str, genuine_dsml: bool
):
    """Replicate Turn 1 at the top of the screenshot (`ukrposhta` bash check)."""
    _, think_end_id = think_ids(tokenizer)
    encode_dsml = genuine_ids if genuine_dsml else bpe_spelled_ids
    ids = (
        bpe_spelled_ids(tokenizer, "Need to inspect ukrposhta lines 60-75.")
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN1_PROSE)
        + encode_dsml(tokenizer, TURN1_HEADLESS_DSML)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == "Need to inspect ukrposhta lines 60-75."
    assert calls == []
    assert content == TURN1_PROSE + TURN1_HEADLESS_DSML


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_control_with_spaced_header_parses_screenshot_missing_string_attributes(
    tokenizer, schedule: str
):
    """Control: adding only `<｜DSML｜ calls>` + `<｜DSML｜ invoke name="bash">` parses
    the exact screenshot parameter block cleanly despite missing `string=` attrs."""
    _, think_end_id = think_ids(tokenizer)
    header = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + genuine_ids(tokenizer, header + TURN2_HEADLESS_DSML)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert content == TURN2_PROSE + "\n\n"
    assert calls == [("bash", {"command": TURN2_COMMAND, "timeout": 400000})]


# ── 2. State-Machine Reachability & V4/V4.1 Header-Drift Matrix ──────────────


def test_state_machine_has_no_transition_from_tool_states_back_to_content():
    """Prove no tool-state transition can swallow a header and return to CONTENT.

    Because the screenshot's leaked `<｜DSML｜ parameter ...>` was emitted as
    `TEXT_CHUNK` (`ParserState.CONTENT`) after prose, the parser never left
    `ParserState.CONTENT` after `</think>`.
    """
    cfg = deepseek_v41_config(thinking=True)
    tool_states = {
        ParserState.TOOL_PREAMBLE,
        ParserState.TOOL_NAME,
        ParserState.TOOL_ARGS,
        ParserState.TOOL_BETWEEN,
        ParserState.TOOL_ARG_END_PENDING,
    }
    returning_to_content = [
        (state, terminal, tr.next_state)
        for (state, terminal), tr in cfg.transitions.items()
        if state in tool_states and tr.next_state == ParserState.CONTENT
    ]
    assert returning_to_content == []

    # Only TOOL_START and INVOKE_PREFIX leave CONTENT for a tool state;
    # PARAM_START, PARAM_CLOSE, INVOKE_END, and TOOL_END have no transition in CONTENT.
    for unhandled in ("PARAM_START", "PARAM_CLOSE", "INVOKE_END", "TOOL_END"):
        assert (ParserState.CONTENT, unhandled) not in cfg.transitions


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_wrapperless_spaced_invoke_still_parses_as_tool_call(tokenizer, schedule: str):
    """Omitting `<｜DSML｜ calls>` alone does NOT cause the screenshot leak:
    `(CONTENT, INVOKE_PREFIX)` transitions directly into `TOOL_NAME`."""
    _, think_end_id = think_ids(tokenizer)
    wrapperless = f'<｜DSML｜ invoke name="bash">\n{TURN2_HEADLESS_DSML}'
    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + genuine_ids(tokenizer, wrapperless)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert content == TURN2_PROSE + "\n\n"
    assert calls == [("bash", {"command": TURN2_COMMAND, "timeout": 400000})]


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "bad_invoke",
    [
        '<｜DSML｜invoke name="bash">\n',  # V4 unspaced invoke with genuine 128825
        "bpe_spelled_spaced_invoke",  # V4.1 spaced invoke spelled in BPE tokens
    ],
)
def test_spaced_calls_then_malformed_invoke_hits_tool_preamble_black_hole(
    tokenizer, schedule: str, bad_invoke: str
):
    """Companion bug uncovered in `deepseek_v41`: if `<｜DSML｜ calls>` is genuine
    (128825) but `<｜DSML｜ invoke>` is V4-unspaced or BPE-spelled, `TOOL_PREAMBLE`
    has no `content_events` entry and no exit transition, silently swallowing the
    entire tool block (`calls == []` and zero DSML in `content`)."""
    _, think_end_id = think_ids(tokenizer)
    if bad_invoke == "bpe_spelled_spaced_invoke":
        invoke_ids = bpe_spelled_ids(tokenizer, '<｜DSML｜ invoke name="bash">\n')
    else:
        invoke_ids = genuine_ids(tokenizer, bad_invoke)

    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + genuine_ids(tokenizer, "<｜DSML｜ calls>\n")
        + invoke_ids
        + genuine_ids(tokenizer, TURN2_HEADLESS_DSML)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert calls == []
    assert content == TURN2_PROSE + "\n\n"
    assert DSML_ATOM not in content


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_v4_unspaced_wrapper_and_invoke_leaks_headers_too(tokenizer, schedule: str):
    """Full V4 unspaced `<｜DSML｜tool_calls>` + `<｜DSML｜invoke>` stays in CONTENT
    and leaks both headers alongside parameters into `content`."""
    _, think_end_id = think_ids(tokenizer)
    v4_block = (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="bash">\n'
        f'<｜DSML｜parameter name="command">{TURN2_COMMAND}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="timeout">400000</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    )
    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + genuine_ids(tokenizer, v4_block)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert calls == []
    assert content == f"{TURN2_PROSE}\n\n{v4_block}"


# ── 3. Multi-Turn Prompt Encoding: Poisoned History & `<think></think>` ──────


def test_turn1_leaked_dsml_is_reencoded_as_bpe_inside_128_token_swa_window(tokenizer):
    """Verify what OpenCode sends into Turn 2 after Turn 1 leaks headless DSML:

    1. Earlier parsed `tool_calls` encode `｜DSML｜` as genuine token 128825.
    2. Turn 1's leaked `assistant.content` has every `｜DSML｜` stripped of 128825
       by `_ControlSpellings` and encoded as plain BPE tokens inside the final
       128-token exact sliding-window attention span.
    3. Because `tools` is present, `effective_drop_thinking = False` injects empty
       `<think></think>` on every prior assistant turn missing `reasoning_content`.
    """
    atom_id = dsml_token_id(tokenizer)
    think_start_id, think_end_id = think_ids(tokenizer)

    messages = [
        {"role": "system", "content": "You are OpenCode, an interactive CLI coding agent."},
        {"role": "user", "content": "Audit carrier lookup memory churn across src/carriers/."},
        # Earlier valid tool turn where client did not replay `reasoning_content`
        {
            "role": "assistant",
            "content": "Checking carrier index files.",
            "tool_calls": [
                {
                    "id": "call_01",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps(
                            {"filePath": "src/carriers/royalmail/index.ts"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_01",
            "content": "export async function lookupRoyalMail(id: string) { /* retry loop */ }",
        },
        # Turn 1 from screenshot: parser returned tool_calls=[], so the leaked DSML
        # sits inside `assistant.content` on the next API request.
        {
            "role": "assistant",
            "content": TURN1_PROSE + TURN1_HEADLESS_DSML,
        },
        # Subagent completion / user continuation before Turn 2
        {
            "role": "user",
            "content": "Subagent finished: Bank ukrposhta, cu.",
        },
    ]

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tools=OPENCODE_TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="high",
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tools=OPENCODE_TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = list(prompt_ids["input_ids"])

    # (a) Empty `<think></think>` injected on historical assistant turns lacking `reasoning_content`.
    assert rendered_text.count("<｜Assistant｜><think></think>") == 2
    empty_think_pairs = [
        i
        for i in range(len(prompt_ids) - 1)
        if prompt_ids[i] == think_start_id and prompt_ids[i + 1] == think_end_id
    ]
    assert len(empty_think_pairs) == 2

    # (b) System `TOOLS_TEMPLATE` (9) + `call_01` (6) produce 15 genuine 128825 tokens,
    # whereas Turn 1's 6 `｜DSML｜` occurrences in `assistant.content` produce ZERO 128825 tokens!
    atom_positions = [i for i, tid in enumerate(prompt_ids) if tid == atom_id]
    assert len(atom_positions) == 15

    # Locate the sub-slice corresponding to Turn 1's assistant turn (after `<tool_output>`).
    user_sp_id = tokenizer.convert_tokens_to_ids("<｜User｜>")
    final_user_pos = max(i for i, tid in enumerate(prompt_ids) if tid == user_sp_id)
    assistant_sp_id = tokenizer.convert_tokens_to_ids("<｜Assistant｜>")
    turn1_assistant_pos = max(
        i for i in range(final_user_pos) if prompt_ids[i] == assistant_sp_id
    )
    assert all(pos < turn1_assistant_pos for pos in atom_positions)
    turn1_slice = prompt_ids[turn1_assistant_pos:final_user_pos]
    turn1_decoded = tokenizer.decode(turn1_slice)

    assert TURN1_HEADLESS_DSML in turn1_decoded
    assert atom_id not in turn1_slice

    # (c) Turn 1's headless BPE-spelled DSML sits directly inside the 128-token SWA window
    # right before Turn 2's `<｜Assistant｜><think>` generation prompt!
    swa_tail_decoded = tokenizer.decode(prompt_ids[-128:])
    assert '</｜DSML｜ parameter>' in swa_tail_decoded
    assert '</｜DSML｜ invoke>' in swa_tail_decoded
    assert '</｜DSML｜ calls>' in swa_tail_decoded
    assert atom_id not in prompt_ids[-128:]


# ── 4. Turn 2 Imitation / CopyDrafter of Poisoned BPE DSML Fails Closed ──────


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_turn2_full_call_copied_with_bpe_dsml_still_leaks_into_content(
    tokenizer, schedule: str
):
    """Even if Turn 2 restores `<｜DSML｜ calls>` and `<｜DSML｜ invoke name="bash">`,
    copying or imitating Turn 1's BPE-spelled `｜DSML｜` tokens omits token 128825,
    so `0768` token provenance rejects the entire call and leaks raw DSML to `content`."""
    _, think_end_id = think_ids(tokenizer)
    full_call = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n' + TURN2_HEADLESS_DSML
    )
    ids = (
        bpe_spelled_ids(tokenizer, TURN2_REASONING)
        + [think_end_id]
        + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
        + bpe_spelled_ids(tokenizer, full_call)
    )

    reasoning, content, calls = run_serving_parser(
        tokenizer, ids, thinking=True, schedule=schedule
    )

    assert reasoning == TURN2_REASONING
    assert calls == []
    assert content == f"{TURN2_PROSE}\n\n{full_call}"


# ── 5. Live Reproducer Prompt Builder & Classifier Integration ───────────────


def test_live_reproducer_prompt_and_classifier_match_real_parser(tokenizer):
    """Verify `repro_live_long_dsml.py` builds prompts with the exact SWA poison
    footprint and classifies real `DeepSeekV41ServingParser` outputs accurately."""
    import importlib.util

    probe_path = Path(__file__).with_name("repro_live_long_dsml.py")
    spec = importlib.util.spec_from_file_location("repro_live_long_dsml", probe_path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    atom_id = dsml_token_id(tokenizer)
    _, think_end_id = think_ids(tokenizer)

    # Verify `poisoned_turn1` prompt tokenization on the real V4.1 tokenizer.
    msgs, _ = probe.build_opencode_messages(4096, arm="poisoned_turn1", seed=20261122)
    prompt_ids = tokenizer.apply_chat_template(
        msgs,
        tools=probe.OPENCODE_TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="high",
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = list(prompt_ids["input_ids"])
    assert 3000 <= len(prompt_ids) <= 6500
    assert atom_id not in prompt_ids[-128:]
    assert "</｜DSML｜ calls>" in tokenizer.decode(prompt_ids[-128:])

    # Feed real `DeepSeekV41ServingParser` outputs into `probe.classify_reply`.
    cases = [
        (
            "ok_tool_call",
            bpe_spelled_ids(tokenizer, TURN2_REASONING)
            + [think_end_id]
            + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
            + genuine_ids(
                tokenizer,
                '<｜DSML｜ calls>\n<｜DSML｜ invoke name="bash">\n'
                + TURN2_HEADLESS_DSML,
            ),
        ),
        (
            "headless_dsml_in_content_genuine_128825",
            bpe_spelled_ids(tokenizer, TURN2_REASONING)
            + [think_end_id]
            + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
            + genuine_ids(tokenizer, TURN2_HEADLESS_DSML),
        ),
        (
            "headless_dsml_in_content_bpe_spelled",
            bpe_spelled_ids(tokenizer, TURN2_REASONING)
            + [think_end_id]
            + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
            + bpe_spelled_ids(tokenizer, TURN2_HEADLESS_DSML),
        ),
        (
            "swallowed_after_tool_start_preamble_blackhole",
            bpe_spelled_ids(tokenizer, TURN2_REASONING)
            + [think_end_id]
            + bpe_spelled_ids(tokenizer, TURN2_PROSE + "\n\n")
            + genuine_ids(
                tokenizer,
                '<｜DSML｜ calls>\n<｜DSML｜invoke name="bash">\n'
                + TURN2_HEADLESS_DSML,
            ),
        ),
    ]
    for expected_verdict, ids in cases:
        reasoning, content, calls = run_serving_parser(
            tokenizer, ids, thinking=True, schedule="per_token"
        )
        reply = {
            "reasoning": reasoning,
            "content": content,
            "calls": [
                {"name": name, "arguments": json.dumps(args)} for name, args in calls
            ],
            "token_ids": ids,
        }
        judged = probe.classify_reply(reply)
        assert judged["verdict"] == expected_verdict
