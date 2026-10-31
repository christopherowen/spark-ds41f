# Preserve Mid-Conversation System Messages in Anthropic Messages API (`#63992`)

## Upstream Source

- Upstream commit: `vllm-project/vllm@7f68ef1999` (`fix(anthropic): keep mid-conversation system messages in place (#63992)`)

## Problem

When serving DeepSeek-V4.1 without an explicit `--chat-template` override (using `DeepseekV41Tokenizer` / `DeepseekV41Renderer`), the Anthropic `/v1/messages` endpoint previously fell back to `merge_inline_system=True`, merging every mid-conversation `system` message (such as Claude Code / agentic harness system reminders) into the leading `system` block at index `0`. Each new mid-conversation reminder therefore mutated the very beginning of the prompt and invalidated ~96% of the KV prefix cache across turns, while also losing the reminder's position in the conversation.

## Changes

- **`vllm/entrypoints/anthropic/serving.py` & `vllm/entrypoints/scale_out/render/serving.py`**:
  - Replace the boolean `_detect_merge_inline_system` with `InlineSystemMode` (`KEEP`, `AS_USER`, `MERGE`) via `detect_inline_system_mode(chat_template, tokenizer)`.
  - Probe the served tokenizer (`tokenizer.supports_inline_system_messages` and `tokenizer.apply_chat_template`) when `chat_template` is `None`.
- **`vllm/tokenizers/deepseek_v41.py`, `vllm/tokenizers/deepseek_v4.py`, `vllm/tokenizers/deepseek_v32.py`**:
  - Set `supports_inline_system_messages = True` on `_DeepseekV41Tokenizer` (whose format natively encodes mid-conversation `<｜System｜>` messages in place) and `supports_inline_system_messages = False` on `_DeepseekV4Tokenizer` and `_DeepseekV32Tokenizer` (so they use `AS_USER` in place rather than `MERGE`).

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/entrypoints/anthropic/test_anthropic_messages_conversion.py -k "InlineSystem"` (`16 passed`).
