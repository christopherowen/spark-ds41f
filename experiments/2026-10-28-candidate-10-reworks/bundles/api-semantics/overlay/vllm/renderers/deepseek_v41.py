# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from typing import Any

from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
)
from vllm.tokenizers.deepseek_v41_encoding import (
    chat_template_kwargs_for_api,
    order_tool_results,
    response_schema_for_prompt,
)

from .deepseek_v4 import DeepseekV4Renderer
from .inputs import DictPrompt
from .params import ChatParams


def _response_format_schema(response_format: Any) -> dict[str, Any] | None:
    """The schema DeepSeek's API writes into the prompt for a response format."""
    return response_schema_for_prompt(response_format)


def _with_api_semantics(params: ChatParams) -> ChatParams:
    """The request's API metadata the V4.1 prompt renders, as template kwargs.

    The returned params share ``chat_template_kwargs`` with the input when the
    request does not render a schema and does not drop tools. Otherwise the
    kwargs dict is a copy, so a mapping reused across requests is not mutated.
    """
    kwargs = chat_template_kwargs_for_api(
        params.chat_template_kwargs,
        response_format=params.response_format,
        tool_choice=params.tool_choice,
    )
    if kwargs is params.chat_template_kwargs:
        return params
    return replace(params, chat_template_kwargs=kwargs)


class DeepseekV41Renderer(DeepseekV4Renderer):
    """The V4 renderer with tool messages in V4.1's prompt order first.

    The encoder sorts a turn's tool results by the order the assistant called
    them; media is collected from the messages before encoding, so the messages
    are put in that order first and images follow their placeholders.

    The prompt is tokenized here, not re-tokenized from the rendered string
    downstream: the tokenizer encodes the template's control tokens as such and
    control-token spellings in request text as plain text, which the string
    form cannot carry (the online chat path asks for ``tokenize=False``).

    A request's response format is rendered into the system prompt, as
    DeepSeek's API does, so the model sees the schema it is constrained to.
    """

    def _apply_chat_template(self, *args, **kwargs):
        kwargs["tokenize"] = True
        return super()._apply_chat_template(*args, **kwargs)

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return super().render_messages(
            order_tool_results(messages),
            params if params is None else _with_api_semantics(params),
        )

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return await super().render_messages_async(
            order_tool_results(messages),
            params if params is None else _with_api_semantics(params),
        )
