# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from transformers import BatchEncoding, TokenizersBackend

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from .deepseek_v41_encoding import (
    ASSISTANT_SP_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_PLACEHOLDER_TEXT,
    REASONING_EFFORT_MAPPINGS,
    USER_SP_TOKEN,
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

# Bounded per-tokenizer piece cache (message/turn chunks -> token IDs).
_MAX_SEGMENT_CACHE = 8192
_MAX_CACHE_BYTES = 64 * 1024 * 1024
_MIN_FAST_PATH_CHARS = 1024
_PARALLEL_CHUNK_THRESHOLD = 65536
_PARALLEL_CHUNK_TARGET = 16384


class _BoundedPieceCache(OrderedDict):
    """Thread-safe LRU cache bounded by entry count and estimated byte cost."""

    def __init__(
        self,
        max_entries: int = _MAX_SEGMENT_CACHE,
        max_bytes: int = _MAX_CACHE_BYTES,
    ) -> None:
        super().__init__()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._total_bytes = 0
        self._lock = threading.RLock()

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @staticmethod
    def _entry_bytes(key: str, value: tuple[int, ...]) -> int:
        return len(key) * 2 + len(value) * 8

    def touch(self, key: str) -> None:
        with self._lock:
            if key in self:
                self.move_to_end(key)

    def __setitem__(self, key: str, value: tuple[int, ...]) -> None:
        entry_bytes = self._entry_bytes(key, value)
        with self._lock:
            old = super().pop(key, None)
            if old is not None:
                self._total_bytes -= self._entry_bytes(key, old)
            if entry_bytes > self._max_bytes:
                return
            super().__setitem__(key, value)
            self._total_bytes += entry_bytes
            while (
                len(self) > self._max_entries
                or self._total_bytes > self._max_bytes
            ):
                evicted_k, evicted_v = super().popitem(last=False)
                self._total_bytes -= self._entry_bytes(evicted_k, evicted_v)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            old = super().pop(key)
            self._total_bytes -= self._entry_bytes(key, old)

    def pop(self, key: str, *args: Any) -> Any:
        with self._lock:
            sentinel = object()
            old = super().pop(key, sentinel)
            if old is sentinel:
                if args:
                    return args[0]
                raise KeyError(key)
            self._total_bytes -= self._entry_bytes(key, old)
            return old

    def popitem(self, last: bool = True) -> tuple[str, tuple[int, ...]]:
        with self._lock:
            k, v = super().popitem(last=last)
            self._total_bytes -= self._entry_bytes(k, v)
            return k, v

    def clear(self) -> None:
        with self._lock:
            super().clear()
            self._total_bytes = 0


_SEGMENT_CACHES: dict[tuple[int, int], _BoundedPieceCache] = {}
_PLAIN_CACHES: dict[tuple[int, int], _BoundedPieceCache] = {}


def _compile_trie_regex(spellings: list[str]) -> re.Pattern[str] | None:
    """Compile a prefix-trie regex matching longest spelling first at any position.

    Factoring shared prefixes (such as the 800+ `<｜place▁holder...` and 440+
    `<|place_holder_mm_span...` added tokens) avoids O(V) backtracking at every
    `<` character in code or DSML markup.
    """
    if not spellings:
        return None
    root: dict[str, Any] = {}
    for spelling in spellings:
        node = root
        for ch in spelling:
            node = node.setdefault(ch, {})
        node[""] = None

    def _render(node: dict[str, Any]) -> str:
        is_end = "" in node
        children = [k for k in node if k != ""]
        if not children:
            return ""
        if len(children) == 1 and not is_end:
            ch = children[0]
            return re.escape(ch) + _render(node[ch])
        parts = [
            re.escape(ch) + _render(node[ch])
            for ch in sorted(children, reverse=True)
        ]
        body = "|".join(parts)
        return f"(?:{body}|)" if is_end else f"(?:{body})"

    return re.compile(_render(root))


def _split_after(chunk: str, delim: str) -> list[str]:
    if not chunk or delim not in chunk:
        return [chunk] if chunk else []
    parts = chunk.split(delim)
    out: list[str] = []
    last = len(parts) - 1
    for idx, part in enumerate(parts):
        piece = part + delim if idx < last else part
        if piece:
            out.append(piece)
    return out


def _split_parallel_newline_chunks(
    text: str, target_chars: int = _PARALLEL_CHUNK_TARGET
) -> list[str]:
    """Split large text just after BPE-invariant newlines for encode_batch.

    In DeepSeek-V4.1's pre-tokenizer regex, punctuation and whitespace rules
    consume trailing `[\\r\\n]*` / `[\\r\\n]+`. Cutting immediately after `\\n`
    (`pos + 1`) when the next character is non-whitespace preserves exact token
    boundaries across every regex branch.
    """
    n = len(text)
    if n < _PARALLEL_CHUNK_THRESHOLD:
        return [text]
    chunks: list[str] = []
    start = 0
    while start + target_chars < n:
        pos = text.find("\n", start + target_chars)
        while pos != -1 and (pos + 1 >= n or text[pos + 1].isspace()):
            pos = text.find("\n", pos + 1)
        if pos == -1 or pos + 1 >= n:
            break
        chunks.append(text[start : pos + 1])
        start = pos + 1
    if start < n:
        chunks.append(text[start:])
    return chunks if chunks else [text]


def _split_prompt_segments(prompt: str) -> list[str]:
    """Split a rendered V4.1 prompt along structural added-token boundaries.

    Splits immediately after `<｜end▁of▁sentence｜>` and `</think>`, and
    immediately before `<｜User｜>` or `<｜Assistant｜>`, so headers, tool
    catalogs, turns, reasoning prefixes, and responses form cacheable
    segments whose concatenation equals `prompt`.
    """
    raw_turns = prompt.split(eos_token)
    chunks: list[str] = []
    last_idx = len(raw_turns) - 1
    for i, turn in enumerate(raw_turns):
        if i < last_idx:
            turn = turn + eos_token
        elif not turn:
            break
        if i == 0 and USER_SP_TOKEN in turn:
            sys_part, sep, rest = turn.partition(USER_SP_TOKEN)
            if sys_part:
                chunks.extend(_split_after(sys_part, thinking_end_token))
            turn = sep + rest
        if ASSISTANT_SP_TOKEN in turn:
            u_part, sep, a_part = turn.partition(ASSISTANT_SP_TOKEN)
            if u_part:
                chunks.extend(_split_after(u_part, thinking_end_token))
            chunks.extend(_split_after(sep + a_part, thinking_end_token))
        else:
            chunks.extend(_split_after(turn, thinking_end_token))
    return chunks


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

    The image placeholder is left alone: image parts render it, and
    `_normalize_messages` escapes the typed spelling in text to ordinary text.
    """

    @staticmethod
    def compile(spellings: list[str]) -> re.Pattern[str] | None:
        """One prefix-factored pattern, longest spelling first; built once per tokenizer."""
        return _compile_trie_regex(spellings)

    def __init__(self, pattern: re.Pattern[str] | None) -> None:
        self.pattern = pattern
        self.originals: list[str] = []
        self.open = self.close = ""
        self._sentinel_re: re.Pattern[str] | None = None

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
        self._sentinel_re = re.compile(
            f"{re.escape(self.open)}(\\d+){re.escape(self.close)}"
        )
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
        if not self.originals or self.open not in text:
            return text
        assert self._sentinel_re is not None
        return self._sentinel_re.sub(
            lambda m: self.originals[int(m.group(1))], text
        )


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
        if key == "arguments" and "\\" in value:
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
    texts = list(_strings(values))
    free = (
        chr(code)
        for code in range(0xF0000, 0xFFFFE)
        if not any(chr(code) in text for text in texts)
    )
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


def _escape_image_token_literal(value: Any, key: str | None = None) -> Any:
    """Escape the image token's literal spelling in request text to plain text.

    The image token in the prompt comes only from image content blocks. The
    token's literal spelling typed into a message, text part, tool result,
    reasoning or tool-call arguments is ordinary text: escape it to the
    ordinary-text spelling, which is in no added-token vocabulary, so the
    encoder cannot mistake it for an image reference (spark-ds41f issue #1031).
    """
    if isinstance(value, str):
        if key == "arguments" and IMAGE_PLACEHOLDER not in value and "\\" in value:
            decoded = _decode_arguments(value)
            if isinstance(decoded, dict):
                return _escape_image_token_literal(decoded)
        return value.replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER_TEXT)
    if isinstance(value, Mapping):
        return {
            k: item
            if k in _PROTOCOL_KEYS
            else _escape_image_token_literal(item, k)
            for k, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_escape_image_token_literal(item) for item in value)
    return value


def _normalize_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    result = [dict(message) for message in messages]
    if any(IMAGE_PLACEHOLDER in text for text in _strings(result)):
        result = [_escape_image_token_literal(message) for message in result]
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
        if isinstance(content, list):
            has_tool_ref = role == "tool" and any(
                isinstance(block, Mapping) and block.get("type") == "tool_reference"
                for block in content
            )
            if has_tool_ref:
                structured_parts: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, Mapping):
                        raise ValueError(
                            "DeepSeek V4.1 supports text and image content only; "
                            f"got {type(block).__name__!r}"
                        )
                    part_type = block.get("type")
                    if part_type in ("text", "input_text", "output_text"):
                        structured_parts.append(
                            {"type": "text", "text": block.get("text") or ""}
                        )
                    elif part_type in ("image_url", "input_image", "image_pil"):
                        structured_parts.append(
                            {"type": "text", "text": IMAGE_PLACEHOLDER}
                        )
                    elif part_type == "tool_reference":
                        ref_name = block.get("name") or block.get("tool_name") or ""
                        structured_parts.append(
                            {"type": "tool_reference", "name": str(ref_name)}
                        )
                    else:
                        raise ValueError(
                            "DeepSeek V4.1 supports text and image content only; "
                            f"got {part_type!r}"
                        )
                message["content"] = structured_parts
            else:
                parts = []
                for block in content:
                    if not isinstance(block, Mapping):
                        raise ValueError(
                            "DeepSeek V4.1 supports text and image content only; "
                            f"got {type(block).__name__!r}"
                        )
                    part_type = block.get("type")
                    if part_type in ("text", "input_text", "output_text"):
                        parts.append(block.get("text") or "")
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


def _resolve_cached_pieces(
    pieces: list[str],
    cache: _BoundedPieceCache,
    encode_batch: Callable[[list[str]], list[list[int]]],
    encode_single: Callable[[str], list[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Resolve token IDs for unique string pieces using `cache` and parallel batching."""
    resolved: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for piece in pieces:
        if piece in resolved:
            continue
        cached = cache.get(piece)
        if cached is not None:
            cache.touch(piece)
            resolved[piece] = cached
        else:
            resolved[piece] = ()
            missing.append(piece)

    if not missing:
        return resolved

    work_items: list[str] = []
    work_index: dict[str, int] = {}
    piece_subchunks: list[tuple[str, list[str]]] = []
    sub_cached: dict[str, tuple[int, ...]] = {}
    total_chars = 0
    for piece in missing:
        subchunks = _split_parallel_newline_chunks(piece)
        piece_subchunks.append((piece, subchunks))
        if len(subchunks) == 1:
            sub = subchunks[0]
            if sub not in work_index:
                work_index[sub] = len(work_items)
                work_items.append(sub)
                total_chars += len(sub)
        else:
            for sub in subchunks:
                if sub in sub_cached or sub in work_index:
                    continue
                hit = cache.get(sub)
                if hit is not None:
                    cache.touch(sub)
                    sub_cached[sub] = hit
                else:
                    work_index[sub] = len(work_items)
                    work_items.append(sub)
                    total_chars += len(sub)

    if not work_items:
        work_ids: list[list[int]] = []
    elif (
        encode_single is not None
        and len(work_items) == 1
        and total_chars < _PARALLEL_CHUNK_THRESHOLD
    ):
        work_ids = [encode_single(work_items[0])]
    elif (
        encode_single is not None
        and len(work_items) <= 8
        and total_chars < 4096
    ):
        work_ids = [encode_single(item) for item in work_items]
    else:
        work_ids = encode_batch(work_items)

    for piece, subchunks in piece_subchunks:
        if len(subchunks) == 1:
            ids_tuple = tuple(work_ids[work_index[subchunks[0]]])
            resolved[piece] = ids_tuple
            cache[piece] = ids_tuple
        else:
            joined: list[int] = []
            for sub in subchunks:
                hit = sub_cached.get(sub)
                if hit is None:
                    hit = tuple(work_ids[work_index[sub]])
                    sub_cached[sub] = hit
                    cache[sub] = hit
                joined.extend(hit)
            resolved[piece] = tuple(joined)
    return resolved


def get_deepseek_v41_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """Wrap an HF tokenizer with the V4.1 prompt encoder."""
    wrapped = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    # Every added-token spelling splits the rendered prompt; all but the image
    # placeholder are kept out of the vocabulary when they occur in request text.
    template_controls = _compile_trie_regex(list(added_vocab))
    text_spellings = _ControlSpellings.compile(
        [s for s in added_vocab if s != IMAGE_PLACEHOLDER]
    )
    plain_encoder: list[Callable[[list[str]], list[list[int]]]] = []
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    can_split_segments = (
        isinstance(tokenizer, TokenizersBackend)
        and backend_tokenizer is not None
        and eos_token in added_vocab
        and USER_SP_TOKEN in added_vocab
        and ASSISTANT_SP_TOKEN in added_vocab
        and thinking_end_token in added_vocab
        and not any("<" in s[1:] for s in added_vocab)
        and not any("\n" in s or "\r" in s for s in added_vocab)
        and not any(
            s != d and s.startswith(d)
            for s in added_vocab
            for d in (
                eos_token,
                USER_SP_TOKEN,
                ASSISTANT_SP_TOKEN,
                thinking_end_token,
            )
        )
    )
    bos_prefix: tuple[int, ...] = ()
    eos_suffix: tuple[int, ...] = ()
    can_fast_special = False
    if can_split_segments:
        canary_plain = tuple(tokenizer.encode("canary check", add_special_tokens=False))
        canary_spec = tuple(tokenizer.encode("canary check", add_special_tokens=True))
        n_plain = len(canary_plain)
        if n_plain > 0:
            for start in range(len(canary_spec) - n_plain + 1):
                if canary_spec[start : start + n_plain] == canary_plain:
                    bos_prefix = canary_spec[:start]
                    eos_suffix = canary_spec[start + n_plain :]
                    can_fast_special = True
                    break

    cache_key = (int(getattr(tokenizer, "vocab_size", 0)), len(added_vocab))
    segment_cache = _SEGMENT_CACHES.setdefault(cache_key, _BoundedPieceCache())
    plain_cache = _PLAIN_CACHES.setdefault(cache_key, _BoundedPieceCache())

    def _seg_single(text: str) -> list[int]:
        assert backend_tokenizer is not None
        return backend_tokenizer.encode(text, add_special_tokens=False).ids

    def _seg_batch(texts: list[str]) -> list[list[int]]:
        assert backend_tokenizer is not None
        return [
            enc.ids
            for enc in backend_tokenizer.encode_batch(
                texts, add_special_tokens=False
            )
        ]

    def encode_prompt_segments(tok_self: Any, prompt: str) -> list[int]:
        assert backend_tokenizer is not None
        if getattr(tok_self, "split_special_tokens", False):
            return TokenizersBackend.encode(
                tok_self, prompt, add_special_tokens=False
            )
        if backend_tokenizer.truncation is not None:
            backend_tokenizer.no_truncation()
        if backend_tokenizer.padding is not None:
            backend_tokenizer.no_padding()
        if backend_tokenizer.encode_special_tokens:
            backend_tokenizer.encode_special_tokens = False

        chunks = _split_prompt_segments(prompt)
        if not chunks:
            return []

        resolved = _resolve_cached_pieces(
            chunks, segment_cache, _seg_batch, _seg_single
        )
        ids: list[int] = []
        for chunk in chunks:
            ids.extend(resolved[chunk])
        return ids

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
        text_pieces = [p for p in pieces if isinstance(p, str)]
        resolved = _resolve_cached_pieces(
            text_pieces, plain_cache, plain_encoder[0]
        )
        ids: list[int] = []
        for piece in pieces:
            if isinstance(piece, str):
                ids.extend(resolved[piece])
            else:
                ids.append(piece)
        return ids

    class _DeepseekV41Tokenizer(tokenizer.__class__):  # type: ignore
        supports_inline_system_messages = True

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
                    if can_split_segments and not tokenizer_kwargs:
                        return encode_prompt_segments(self, prompt)
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

        def _encode_fast_str(
            self, text: str, add_special_tokens: bool
        ) -> list[int] | None:
            if (
                not can_split_segments
                or len(text) < _MIN_FAST_PATH_CHARS
                or getattr(self, "split_special_tokens", False)
                or (add_special_tokens and not can_fast_special)
            ):
                return None
            ids = encode_prompt_segments(self, text)
            if add_special_tokens and (bos_prefix or eos_suffix):
                return [*bos_prefix, *ids, *eos_suffix]
            return ids

        _ENCODE_FAST_KEYS = frozenset(
            {
                "add_special_tokens",
                "padding",
                "truncation",
                "max_length",
                "text_pair",
                "return_tensors",
            }
        )
        _CALL_FAST_KEYS = frozenset(
            {
                "add_special_tokens",
                "padding",
                "truncation",
                "max_length",
                "return_token_type_ids",
                "return_attention_mask",
                "return_tensors",
                "verbose",
            }
        )

        def encode(self, text: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                not args
                and isinstance(text, str)
                and self._ENCODE_FAST_KEYS.issuperset(kwargs)
                and kwargs.get("text_pair") is None
                and kwargs.get("padding", False) is False
                and not kwargs.get("truncation")
                and kwargs.get("max_length") is None
                and kwargs.get("return_tensors") is None
            ):
                ids = self._encode_fast_str(
                    text, kwargs.get("add_special_tokens", True)
                )
                if ids is not None:
                    return ids
            return super().encode(text, *args, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> BatchEncoding:
            if len(args) == 1 and "text" not in kwargs:
                text = args[0]
                key_ok = self._CALL_FAST_KEYS.issuperset(kwargs)
            elif not args and "text" in kwargs:
                text = kwargs["text"]
                key_ok = (self._CALL_FAST_KEYS | {"text"}).issuperset(kwargs)
            else:
                text = None
                key_ok = False
            if (
                key_ok
                and isinstance(text, str)
                and kwargs.get("padding", False) is False
                and not kwargs.get("truncation")
                and kwargs.get("max_length") is None
                and kwargs.get("return_tensors") is None
            ):
                ids = self._encode_fast_str(
                    text, kwargs.get("add_special_tokens", True)
                )
                if ids is not None:
                    data: dict[str, Any] = {"input_ids": ids}
                    input_names = getattr(
                        self,
                        "model_input_names",
                        ("input_ids", "attention_mask"),
                    )
                    ret_type_ids = kwargs.get("return_token_type_ids")
                    ret_attn_mask = kwargs.get("return_attention_mask")
                    if ret_type_ids or (
                        ret_type_ids is None and "token_type_ids" in input_names
                    ):
                        data["token_type_ids"] = [0] * len(ids)
                    if ret_attn_mask or (
                        ret_attn_mask is None
                        and "attention_mask" in input_names
                    ):
                        data["attention_mask"] = [1] * len(ids)
                    return BatchEncoding(data)
            return super().__call__(*args, **kwargs)

        def num_special_tokens_to_add(self) -> int:
            return len(self.encode(""))

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v41_tokenizer, (tokenizer,)

    wrapped._segment_cache = segment_cache
    wrapped._plain_cache = plain_cache
    _DeepseekV41Tokenizer.__name__ = f"DSV41{tokenizer.__class__.__name__}"
    wrapped.__class__ = _DeepseekV41Tokenizer
    return wrapped


class DeepseekV41Tokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> HfTokenizer:
        tokenizer = TokenizersBackend.from_pretrained(*args, **kwargs)
        return get_cached_tokenizer(get_deepseek_v41_tokenizer(tokenizer))
