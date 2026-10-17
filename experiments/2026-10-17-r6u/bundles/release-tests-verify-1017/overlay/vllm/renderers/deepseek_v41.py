# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
)
from vllm.tokenizers.deepseek_v41_encoding import order_tool_results

from .deepseek_v4 import DeepseekV4Renderer
from .inputs import DictPrompt
from .params import ChatParams


class DeepseekV41Renderer(DeepseekV4Renderer):
    """The V4 renderer with tool messages in V4.1's prompt order first.

    The encoder sorts a turn's tool results by the order the assistant called
    them; media is collected from the messages before encoding, so the messages
    are put in that order first and images follow their placeholders.
    """

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return super().render_messages(order_tool_results(messages), params)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        return await super().render_messages_async(order_tool_results(messages), params)
