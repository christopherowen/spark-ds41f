# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import (
    IMAGE_PLACEHOLDER,
    encode_messages,
    order_tool_results,
)

FIXTURES = Path(__file__).parent / "fixtures"


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


@pytest.mark.parametrize("case_id", [1, 2])
def test_reference_encoder_fixtures(case_id):
    # Expected prompts come from the DeepSeek V4.1 reference encoder.
    data = json.loads(
        (FIXTURES / "deepseek_v4" / f"test_input_{case_id}.json").read_text()
    )
    messages = data["messages"] if isinstance(data, dict) else data
    tools = data.get("tools") if isinstance(data, dict) else None
    expected = (FIXTURES / "deepseek_v41" / f"test_output_{case_id}.txt").read_text()
    assert render(messages, tools=tools, reasoning_effort=50) == expected


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_numeric_reasoning_effort(effort, budget):
    assert render(
        [{"role": "user", "content": "question"}], reasoning_effort=effort
    ) == (
        "<｜begin▁of▁sentence｜><｜System｜>"
        f"Reasoning Effort: {budget} (range 1-100, the higher the value, "
        "the more thorough the reasoning)\n\n"
        "<｜User｜>question<｜Assistant｜><think>"
    )


@pytest.mark.parametrize("effort", ["medium", "minimal", -1, 0, 101, True, 1.5, []])
def test_invalid_effort_is_a_request_error(effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        render([{"role": "user", "content": "question"}], reasoning_effort=effort)


@pytest.mark.parametrize(
    "controls",
    [
        {"thinking": False},
        {"enable_thinking": False},
        {"thinking": True, "reasoning_effort": "none"},
        {"enable_thinking": True, "reasoning_effort": "none"},
    ],
)
def test_chat_mode_has_closed_thinking_prefix(controls):
    assert render([{"role": "user", "content": "question"}], **controls) == (
        "<｜begin▁of▁sentence｜><｜User｜>question<｜Assistant｜></think>"
    )


def test_raw_text_parts_preserve_reference_separator():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    original = copy.deepcopy(messages)
    assert (
        render(
            messages,
            conversation=[{"role": "user", "content": "first\nsecond"}],
            thinking=False,
        )
        == "<｜begin▁of▁sentence｜><｜User｜>first\n\nsecond<｜Assistant｜></think>"
    )
    assert messages == original


@pytest.mark.parametrize(
    ("role", "responses_type"),
    [("user", "input_text"), ("assistant", "output_text")],
)
def test_responses_text_parts_match_chat_text_parts(role, responses_type):
    responses_message = {
        "role": role,
        "content": [{"type": responses_type, "text": "hello"}],
    }
    chat_message = {
        "role": role,
        "content": [{"type": "text", "text": "hello"}],
    }
    assert render([responses_message], thinking=False) == render(
        [chat_message], thinking=False
    )


def test_mid_system_gets_its_own_marker_and_generation_header():
    assert render(
        [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "answer briefly"},
        ],
        thinking=False,
    ) == (
        "<｜begin▁of▁sentence｜><｜User｜>question"
        "<｜System｜>answer briefly<｜Assistant｜></think>"
    )


def test_top_level_effort_overrides_template_effort():
    request = ChatCompletionRequest(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        messages=[{"role": "user", "content": "question"}],
        reasoning_effort="low",
        chat_template_kwargs={"reasoning_effort": 100},
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert "Reasoning Effort: 50 " in render(request.messages, **kwargs)


def test_encode_uses_one_bos_and_forwards_truncation():
    tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        thinking=False,
        truncation=True,
        max_length=16,
    )
    text, add_special_tokens, kwargs = tokenizer.last_encode
    assert text.startswith("<｜begin▁of▁sentence｜>")
    assert add_special_tokens is False
    assert kwargs == {"truncation": True, "max_length": 16}


@pytest.mark.parametrize("image_type", ["image_url", "input_image", "image_pil"])
def test_images_preserve_content_order_and_reference_separator(image_type):
    from PIL import Image

    def image_part(color):
        if image_type == "image_pil":
            return {"type": image_type, "image_pil": Image.new("RGB", (2, 2), color)}
        url = f"https://example.com/{color}.png"
        return {
            "type": image_type,
            "image_url": {"url": url} if image_type == "image_url" else url,
        }

    parts = [
        {"type": "text", "text": "first"},
        image_part("red"),
        {"type": "text", "text": "second"},
        image_part("blue"),
        {"type": "text", "text": "compare"},
    ]
    assert render([{"role": "user", "content": parts}], thinking=False) == (
        "<｜begin▁of▁sentence｜><｜User｜>first\n\n<｜deepseek_image｜>\n\n"
        "second\n\n<｜deepseek_image｜>\n\ncompare<｜Assistant｜></think>"
    )
    assert parts[1]["type"] == parts[3]["type"] == image_type


def test_unsupported_media_is_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])


_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_v41/encoding.json").read_text()
)


@pytest.mark.parametrize("case", _FIXTURES["cases"], ids=lambda case: case["name"])
def test_prompt_matches_reference_fixture(case):
    original = copy.deepcopy(case["messages"])
    request = ChatCompletionRequest(
        messages=case["messages"],
        tools=case.get("tools"),
    )
    prompt = get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        request.messages,
        tools=[tool.model_dump(exclude_none=True) for tool in request.tools]
        if request.tools
        else None,
        tokenize=False,
        **case["kwargs"],
    )
    assert prompt == case["expected"]
    assert case["messages"] == original


@pytest.mark.parametrize("location", ["tool", "function"])
@pytest.mark.parametrize("namespace", ["inventory", {"name": "inventory"}])
def test_direct_encoder_and_api_preserve_the_same_tool_identity(location, namespace):
    tool: dict[str, Any] = {"type": "function", "function": {"name": "lookup"}}
    call: dict[str, Any] = {
        "id": "call_a",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"sku":"A1"}'},
    }
    for item in (tool, call):
        (item if location == "tool" else item["function"])["namespace"] = namespace
    messages = [
        {"role": "system", "content": "Use tools.", "tools": [tool]},
        {"role": "user", "content": "Find A1."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_a", "content": "In stock."},
    ]
    original = copy.deepcopy(messages)
    direct = encode_messages(messages, thinking_mode="chat")
    request = ChatCompletionRequest(messages=messages)
    rendered = get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        request.messages,
        tokenize=False,
        thinking=False,
    )
    assert rendered == direct
    assert '"name": "inventory::lookup"' in rendered
    assert 'name="inventory::lookup"' in rendered
    assert "<tool_result>In stock.</tool_result>" in rendered
    assert messages == original


def test_reminder_does_not_enable_arbitrary_message_roles():
    with pytest.raises(ValueError, match="Invalid role: SYSTEM"):
        get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
            [{"role": "SYSTEM", "content": "Hello"}],
            tokenize=False,
        )


@pytest.mark.parametrize("role", ["user", "tool", "assistant"])
def test_image_token_in_text_is_a_request_error(role):
    """Image tokens come only from image parts, in string content, text parts and
    reasoning alike (DeepSeek's reference rejects them before merging)."""
    message = {"role": role, "content": f"hi {IMAGE_PLACEHOLDER}"}
    if role == "tool":
        message["tool_call_id"] = "c1"
    with pytest.raises(ValueError, match="image special token"):
        render([message])
    with pytest.raises(ValueError, match="image placeholder"):
        render(
            [{"role": "user", "content": [{"type": "text", "text": IMAGE_PLACEHOLDER}]}]
        )
    with pytest.raises(ValueError, match="reasoning_content contains image"):
        render(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "a",
                    "reasoning_content": IMAGE_PLACEHOLDER,
                },
                {"role": "user", "content": "q2"},
            ]
        )


def _call(name):
    return {
        "id": name,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _tool(name, *, image=False):
    content = [{"type": "text", "text": f"result {name}"}]
    if image:
        content.append({"type": "image_url", "image_url": {"url": f"{name}.png"}})
    return {"role": "tool", "tool_call_id": name, "content": content}


def test_tool_messages_take_call_order_within_their_merged_turn():
    """order_tool_results permutes a merged turn's tool messages into the calls'
    order and leaves everything else in place, so the encoder reorders nothing."""
    user = {"role": "user", "content": "also look at this"}
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [_call("a"), _call("b"), _call("c")]},
        _tool("c"),
        user,
        _tool("a"),
        _tool("unknown"),
        _tool("b"),
    ]
    original = copy.deepcopy(messages)
    ordered = order_tool_results(messages)
    assert [m.get("tool_call_id") for m in ordered] == [
        None, None, "a", None, "unknown", "b", "c"
    ]  # fmt: skip
    assert ordered[3] is user
    assert messages == original
    assert render(ordered) == render(messages)
    # A completed turn is untouched, and a single result has nothing to order.
    single = [{"role": "user", "content": "q"}, _tool("x")]
    assert order_tool_results(single) == single


def test_tool_result_images_follow_prompt_order(monkeypatch):
    """Tool results answered out of call order render in call order, and the
    renderer hands media collection the messages in that order, so each image
    follows its own placeholder (DeepSeek's 2026-10-16 encoding fix)."""
    from vllm.renderers import deepseek_v41
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer

    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "h.png"}}],
        },
        {"role": "assistant", "tool_calls": [_call("a"), _call("b")]},
        _tool("b", image=True),
        _tool("a", image=True),
    ]
    assert render(messages, thinking=False).endswith(
        f"<｜User｜><tool_result>result a\n\n{IMAGE_PLACEHOLDER}</tool_result>\n\n"
        f"<tool_result>result b\n\n{IMAGE_PLACEHOLDER}</tool_result>"
        "<｜Assistant｜></think>"
    )
    seen = []
    monkeypatch.setattr(
        DeepseekV4Renderer, "render_messages", lambda self, m, params: seen.append(m)
    )
    renderer = deepseek_v41.DeepseekV41Renderer.__new__(
        deepseek_v41.DeepseekV41Renderer
    )
    renderer.render_messages(messages, None)
    urls = [
        part["image_url"]["url"]
        for message in seen[0]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert urls == ["h.png", "a.png", "b.png"]


def test_current_turn_reasoning_survives_tool_results():
    """Reasoning is dropped for completed turns but kept across the current turn's
    tool results; the next user message completes the turn."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = render(messages)
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")
    next_turn = render(messages + [{"role": "user", "content": "q3"}])
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")
