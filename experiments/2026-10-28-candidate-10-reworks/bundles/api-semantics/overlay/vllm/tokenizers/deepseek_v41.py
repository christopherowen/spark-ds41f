# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from transformers import TokenizersBackend

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    ASSISTANT_SP_TOKEN,
    IMAGE_PLACEHOLDER,
    REASONING_EFFORT_MAPPINGS,
    encode_messages,
    eos_token,
    insert_system_prompt_fields,
    thinking_end_token,
    thinking_start_token,
)
from .hf import HfTokenizer, get_cached_tokenizer
from .protocol import TokenizerLike

# Keys whose values are protocol identifiers, never rendered request text.
_PROTOCOL_KEYS = frozenset({"role", "type"})


class _ControlSpellings:
    """Keeps control-token spellings written in request text out of the vocabulary.

    The encoder renders one prompt string, so a control token the template writes
    and the same spelling inside a message, a tool result, a tool call or a tool
    definition are indistinguishable once rendered. Before rendering, every such
    spelling in request text is replaced by a sentinel that no request string
    contains; after rendering, the prompt is split at the control tokens that
    remain, which can only come from the template, and the text between them is
    encoded without the added vocabulary, with the sentinels restored. This
    matches DeepSeek's encoder, which encodes a typed spelling as plain text.

    The image placeholder is left alone: `_normalize_messages` already rejects it
    in text, and image parts render it.
    """

    @staticmethod
    def compile(spellings: list[str]) -> re.Pattern[str] | None:
        """One alternation, longest spelling first; built once per tokenizer."""
        ordered = sorted(spellings, key=len, reverse=True)
        return re.compile("|".join(re.escape(s) for s in ordered)) if ordered else None

    def __init__(self, pattern: re.Pattern[str] | None) -> None:
        self.pattern = pattern
        self.originals: list[str] = []
        self.open = self.close = ""

    def protect(self, *values: Any) -> list[Any]:
        """Return the values with control spellings in text replaced by sentinels.

        Values without any such spelling are returned unchanged, and the prompt is
        then encoded exactly as before.
        """
        assert not self.originals, "one _ControlSpellings per prompt"
        if self.pattern is None or not any(
            self.pattern.search(text) for text in _strings(values)
        ):
            return list(values)
        self.open, self.close = _unused_private_pair(values)
        return [self._protect(value) for value in values]

    @property
    def active(self) -> bool:
        return bool(self.originals)

    def _protect(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            assert self.pattern is not None
            if key == "arguments":
                decoded = _decode_arguments(value)
                if isinstance(decoded, dict):
                    # The encoder decodes JSON arguments before rendering them, so
                    # spellings escaped in the JSON are protected in decoded form.
                    return self._protect(decoded)
            return self.pattern.sub(self._sentinel, value)
        if isinstance(value, Mapping):
            return {
                k: item if k in _PROTOCOL_KEYS else self._protect(item, k)
                for k, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return type(value)(self._protect(item) for item in value)
        return value

    def _sentinel(self, match: re.Match[str]) -> str:
        self.originals.append(match.group(0))
        return f"{self.open}{len(self.originals) - 1}{self.close}"

    def restore(self, text: str) -> str:
        if not self.originals:
            return text
        sentinel = re.compile(f"{re.escape(self.open)}(\\d+){re.escape(self.close)}")
        return sentinel.sub(lambda m: self.originals[int(m.group(1))], text)


def _decode_arguments(arguments: str) -> Any:
    """Tool-call arguments as `encode_arguments_to_dsml` decodes them."""
    for _ in range(2):
        if not isinstance(arguments, str):
            break
        try:
            arguments = json.loads(arguments)
        except Exception:
            break
    return arguments


def _strings(value: Any, key: str | None = None):
    if isinstance(value, str):
        yield value
        if key == "arguments":
            decoded = _decode_arguments(value)
            if not isinstance(decoded, str):
                yield from _strings(decoded)
    elif isinstance(value, Mapping):
        for k, item in value.items():
            yield from _strings(item, k)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _unused_private_pair(values: tuple[Any, ...]) -> tuple[str, str]:
    """Two supplementary private-use characters that no request string contains."""
    used: set[str] = set()
    for text in _strings(values):
        used.update(text)
    free = (chr(code) for code in range(0xF0000, 0xFFFFE) if chr(code) not in used)
    return next(free), next(free)


def _plain_text_encoder(tokenizer: Any) -> Callable[[list[str]], list[list[int]]]:
    """Encode text with the model vocabulary but none of the added tokens."""
    from tokenizers import Tokenizer as RustTokenizer

    spec = json.loads(tokenizer.backend_tokenizer.to_str())
    spec["added_tokens"] = []
    plain = RustTokenizer.from_str(json.dumps(spec))

    def encode(texts: list[str]) -> list[list[int]]:
        return [
            encoding.ids
            for encoding in plain.encode_batch(texts, add_special_tokens=False)
        ]

    return encode


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    result = [dict(message) for message in copy.deepcopy(messages)]
    for message in result:
        role = message.get("role")
        if role not in (
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
            "latest_reminder",
        ):
            raise ValueError(f"Invalid role: {role}")
        if role == "developer":
            message["role"] = "system"
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        content = message.get("content")
        # Image tokens come only from image parts, as the reference requires.
        if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
            raise ValueError(
                f"Message content contains image special token '{IMAGE_PLACEHOLDER}'. "
                "Images should be provided as image content blocks."
            )
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and IMAGE_PLACEHOLDER in reasoning:
            raise ValueError(
                f"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
            )
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type in ("text", "input_text", "output_text"):
                    text = block.get("text", "")
                    if IMAGE_PLACEHOLDER in text:
                        raise ValueError(
                            "Text block contains image placeholder "
                            f"'{IMAGE_PLACEHOLDER}'. Images should be separate "
                            "content blocks."
                        )
                    parts.append(text)
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\n\n".join(parts)
    return result


_PREFIX_ERROR = (
    "Invalid completion message: prefix set true with message which is not "
    "final message or message role is not assistant"
)


def _prefix_continuation(conversation: list[dict[str, Any]]) -> bool:
    """Whether the final assistant message is a prefix continuation.

    DeepSeek's API rejects ``prefix: true`` on any other message. ``prefix:
    false`` is an ordinary message.
    """
    last = len(conversation) - 1
    for index, message in enumerate(conversation):
        if not message.get("prefix"):
            continue
        if index != last or message.get("role") != "assistant":
            raise ValueError(_PREFIX_ERROR)
        return True
    return False


def _apply_generation_controls(
    prompt: str,
    final: dict[str, Any] | None,
    *,
    thinking: bool,
    add_generation_prompt: bool | None,
    continue_final_message: bool,
) -> str:
    """Honor `add_generation_prompt` and `continue_final_message` on the prompt
    the reference encoder rendered.

    The encoder opens the assistant turn (`<｜Assistant｜>` and the thinking
    token) after a final user, tool or mid-conversation system message, and
    closes a final assistant message with its end-of-sentence token. Left alone,
    its prompt is unchanged: a caller that does not pass the controls, or passes
    the API defaults on a conversation the encoder already opens, gets exactly
    the reference prompt.
    - `add_generation_prompt=True` after a closed assistant message opens a new
      assistant turn instead of asking the model to write after its
      end-of-sentence token. A message the encoder followed with a task token
      (e.g. `<｜title｜>`) already asks for that task and is left alone.
    - `add_generation_prompt=False` leaves the turn the encoder opened unopened.
    - `continue_final_message=True` ends the prompt at the final message's text:
      an assistant message without its end-of-sentence token (the encoder's
      `wo_eos`), its reasoning still open in thinking mode when it has no
      content or tool calls yet; any other message without the opened turn.
    """
    cue = ASSISTANT_SP_TOKEN + (
        thinking_start_token if thinking else thinking_end_token
    )
    final_is_assistant = final is not None and final["role"] == "assistant"
    if continue_final_message:
        if not final_is_assistant:
            return prompt.removesuffix(cue)
        if thinking and not final.get("content") and not final.get("tool_calls"):
            # The reasoning is still being written: leave it open.
            return prompt.removesuffix(thinking_end_token)
        return prompt
    if add_generation_prompt is False and not final_is_assistant:
        return prompt.removesuffix(cue)
    if add_generation_prompt and final_is_assistant and prompt.endswith(eos_token):
        return prompt + cue
    return prompt


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    # Every added-token spelling splits the rendered prompt; all but the image
    # placeholder are kept out of the vocabulary when they occur in request text.
    template_controls = (
        re.compile(
            "|".join(re.escape(s) for s in sorted(added_vocab, key=len, reverse=True))
        )
        if added_vocab
        else None
    )
    text_spellings = _ControlSpellings.compile(
        [s for s in added_vocab if s != IMAGE_PLACEHOLDER]
    )
    plain_encoder: list[Callable[[list[str]], list[list[int]]]] = []

    def encode_with_text_spellings(
        prompt: str, controls: _ControlSpellings
    ) -> list[int]:
        """Template controls become their IDs; everything else is plain text."""
        assert template_controls is not None
        if not plain_encoder:
            plain_encoder.append(_plain_text_encoder(tokenizer))
        pieces: list[int | str] = []
        position = 0
        for match in template_controls.finditer(prompt):
            if match.start() > position:
                pieces.append(controls.restore(prompt[position : match.start()]))
            pieces.append(added_vocab[match.group(0)])
            position = match.end()
        if position < len(prompt):
            pieces.append(controls.restore(prompt[position:]))
        texts = [piece for piece in pieces if isinstance(piece, str)]
        encoded = iter(plain_encoder[0](texts))
        ids: list[int] = []
        for piece in pieces:
            if isinstance(piece, str):
                ids.extend(next(encoded))
            else:
                ids.append(piece)
        return ids

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            messages: list[ChatCompletionMessageParam],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> str | list[int]:
            # The schema of a response format, which the encoder writes after
            # the tools in the first system message (the V4.1 renderer sets it
            # from the request's response_format, as DeepSeek's API does).
            response_format = kwargs.get("response_format")
            # Control-token spellings in request text must not become control
            # tokens (see _ControlSpellings); requests without any are unchanged.
            controls = _ControlSpellings(text_spellings)
            protected = controls.protect(messages, tools, response_format)
            messages, tools, response_format = protected
            # The generic renderer's conversation has already flattened text
            # parts with '\n'. V4.1 encodes the original messages with '\n\n'.
            conversation = _normalize_messages(messages)
            # response_format is the schema dict, or None. An empty schema is
            # still written; only None is left out. protect() returns one value
            # per argument, so a schema that is itself a list is not unpacked.
            insert_system_prompt_fields(conversation, tools, response_format)

            thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
            if "thinking" not in kwargs and "enable_thinking" not in kwargs:
                thinking = True
            effort = kwargs.get("reasoning_effort")
            if effort == "none":
                thinking = False
                effort = None
            if effort is None:
                effort = "high"
            if not (
                (type(effort) is int and 1 <= effort <= 100)
                or (isinstance(effort, str) and effort in REASONING_EFFORT_MAPPINGS)
            ):
                raise ValueError(
                    "DeepSeek V4.1 reasoning_effort must be minimal, low, medium, "
                    "high, xhigh, max, ultra, or an integer within [1, 100] in "
                    "chat_template_kwargs"
                )

            add_generation_prompt = kwargs.get("add_generation_prompt")
            continue_final_message = bool(kwargs.get("continue_final_message"))
            # A final assistant message with prefix=true is that message
            # continued, which is the encoder's continue_final_message. The
            # chat API defaults add_generation_prompt to true; the prefix
            # replaces that open turn instead of raising the both-controls error.
            if _prefix_continuation(conversation):
                add_generation_prompt = False
                continue_final_message = True
            if continue_final_message and add_generation_prompt:
                raise ValueError(
                    "Cannot set both `add_generation_prompt` and "
                    "`continue_final_message` to True."
                )
            if continue_final_message and not conversation:
                raise ValueError("`continue_final_message` needs a final message.")
            final = conversation[-1] if conversation else None
            final_is_assistant = final is not None and final["role"] == "assistant"
            # The reference encoder's own switch for an assistant message the
            # model continues: rendered without its end-of-sentence token.
            if continue_final_message and final_is_assistant:
                final["wo_eos"] = True

            prompt = encode_messages(
                conversation,
                thinking_mode="thinking" if thinking else "chat",
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=effort,
            )
            prompt = _apply_generation_controls(
                prompt,
                final,
                thinking=thinking,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
            )
            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                if not controls.active:
                    return self.encode(
                        prompt, add_special_tokens=False, **tokenizer_kwargs
                    )
                ids = encode_with_text_spellings(prompt, controls)
                max_length = tokenizer_kwargs.get("max_length")
                if tokenizer_kwargs.get("truncation") and max_length is not None:
                    if getattr(self, "truncation_side", "right") == "left":
                        ids = ids[-max_length:]
                    else:
                        ids = ids[:max_length]
                return ids
            # The text form cannot mark which spellings are controls; it is for
            # display, and serving encodes through the branch above.
            return controls.restore(prompt)

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
