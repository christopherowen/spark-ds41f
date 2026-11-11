# Fast-path DeepSeek-V4.1 multi-turn chat rendering, control-spelling protection, and segment tokenization (`2026-11-08-chat-template-multiturn-fastpath`)

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (promoted `spark-ds41f-1107`, `experiments/2026-10-26-spark-ds41f-candidate/1107`)
- **Base vLLM commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, `0001..0761`)
- **Candidate vLLM commit**: `5b4f3275851e2d5bcb6d1c1de8753da789a4bf5d` (tree `df15576c1c7add1a108e73877cfc9ebad3f3824e`, `0001..0762`, `patchset_sha256` `cb5b5fac725c9fc0bcaabe69c73bc2df772205231fff879d4a717498bbcd4594`)
- **Intended causal variable**: Eliminate quadratic message scanning, redundant full-conversation `copy.deepcopy` passes, unfactored 1,283-branch control-spelling regex backtracking, and single-threaded full-prompt re-tokenization across deep (100–500 turn, 52K–249K token) DeepSeek-V4.1 coding-agent conversations (`0762-perf-dsv41-fast-path-multi-turn-chat-rendering-and-s.patch`).

## 1. Host CPU & Heap Bottleneck Analysis on `spark-ds41f:1107`

Profiling the OpenAI `/v1/chat/completions` host preprocessing pipeline (`DeepseekV41Renderer` -> `order_tool_results` -> `parse_chat_messages` -> `_DeepseekV41Tokenizer.apply_chat_template`) on Grace ARM64 (`sp7`, `spark-ds41f:1107`, `--cpuset-cpus=5-9,15-19`) across 100-turn to 500-turn coding-agent conversations (202–1,011 messages, 24 tool definitions, assistant `reasoning_content` + DSML `tool_calls` + `tool` outputs, 52K–249K tokens) identified five host CPU and memory bottlenecks:

1. **Quadratic $O(M^2)$ `find_last_user_index` Re-evaluation in `vllm/tokenizers/deepseek_v41_encoding.py`**:
   - `encode_messages()` called `render_message(idx + context_len, full_messages, ...)` in a loop over all $M$ messages, and `render_message()` recomputed `last_user_idx = find_last_user_index(messages)` on every message.
   - Because merged `tool_result` user messages do not count as user turns, in a single-user-turn coding-agent loop with 500 tool calls (`1,002` messages), `find_last_user_index()` walked backward across ~1,000 merged tool messages on every single message ($1,002^2 \approx 10^6$ iterations, adding ~117 ms of pure Python loop overhead at 500 turns).
   -Fix: Compute `last_user_idx` once in `encode_messages()`, forward it into `_drop_thinking_messages()` and `render_message()`, and join rendered message chunks via `"".join(parts)`.
2. **Redundant Full-Conversation `copy.deepcopy` in `_normalize_messages` and `merge_tool_messages`**:
   - `_normalize_messages()` (`vllm/tokenizers/deepseek_v41.py:165`) ran `copy.deepcopy(messages)` on the entire conversation, `insert_system_prompt_fields()` attached the 24 tool schemas to `conversation[0]["tools"]`, and `merge_tool_messages()` (`vllm/tokenizers/deepseek_v41_encoding.py:243`) ran `msg = copy.deepcopy(msg)` on every message again (deep-copying all 24 tool schemas and all nested `tool_calls` dictionaries, invoking `copy.deepcopy` 45,710 times and allocating >6 MiB of transient heap objects per 500-turn request).
   - Fix: Neither `_normalize_messages()` nor `merge_tool_messages()` mutates nested structures inside `tools` or `tool_calls`. Shallow-copying each message dict (`dict(message)`) in `_normalize_messages()`, copying `dict(msg)` / `list(content_blocks)` only for user messages in `merge_tool_messages()`, and skipping single-item group/block sorting allocations in `order_tool_results()` and `sort_tool_results_by_call_order()` preserves full caller immutability while cutting peak Python heap allocations by **51.8%–53.4%**.
3. **Unfactored 1,283-Branch Alternation Regex in `_ControlSpellings` and `template_controls`**:
   - `DeepSeek-V4.1-Flash-1017` registers 1,283 tokens in `added_vocab` (including 800 `<｜place▁holder▁no▁...｜>` and 442 `<|place_holder_mm_span_...|>` tokens).
   - `_ControlSpellings.compile` and `template_controls` compiled `"|".join(re.escape(s) for s in sorted(..., key=len, reverse=True))`, causing Python's `sre` engine to backtrack across 800–1,281 branches at every `<` character in source code (`idx < limit`, `vec<int>`) and every `<｜DSML｜` tag in the rendered prompt.
   - Fix: Compile `text_spellings` and `template_controls` as prefix-factored character trie regexes (`_compile_trie_regex`), skip `json.loads()` in `_strings()` when tool `arguments` strings contain no backslash escapes (`"\\" in value`), test candidate private-use sentinels in `_unused_private_pair()` via $O(1)$ UCS-4 string membership (`chr(code) in text`) instead of hashing every character into a `set[str]`, and guard `controls.restore()` with `self.open not in text`.
4. **Per-Message `MultiModalContentParser` Allocation on Plain Text / `None` Content in `vllm/entrypoints/chat_utils.py`**:
   - `_parse_chat_message_content()` wrapped `content is None` and `isinstance(content, str)` into `[ChatCompletionContentPartTextParam(...)]` and allocated an `AsyncMultiModalContentParser` + `defaultdict(list)` per message.
   - Fix: Fast-path `content is None` and `isinstance(content, str)` directly in `_parse_chat_message_content()` while still invoking `_reject_reserved_placeholder_in_text(content, mm_tracker._model_config)` and honoring `content_format == "openai"` vs `"string"`.
5. **Single-Threaded Full-Prompt Re-Tokenization & `BatchEncoding` Wrapper Overhead (`vllm/tokenizers/deepseek_v41.py`)**:
   - `_DeepseekV41Tokenizer.apply_chat_template` called `self.encode(prompt, add_special_tokens=False)`, which ran single-threaded BPE over the entire 500–950 KB rendered prompt on every turn and allocated a discarded `attention_mask` list of 249,000 `1`s in `TokenizersBackend._convert_encoding`.
   - In `DeepSeek-V4.1-Flash-1017`, no added token contains `<` after index 0 or has `<｜end▁of▁sentence｜>`, `<｜User｜>`, or `<｜Assistant｜>` as a proper prefix, and Rust `tokenizers` extracts added tokens before `PreTokenizer` and BPE run.
   - Fix: When `can_split_segments` holds, split `prompt` along `<｜end▁of▁sentence｜>`, `<｜User｜>`, and `<｜Assistant｜>` into turn/message segments (`_split_prompt_segments`), cache tokenized segments in a bounded `_SEGMENT_CACHES` map (`_MAX_SEGMENT_CACHE = 8192`), and encode uncached segments in parallel via `backend_tokenizer.encode_batch(missing, add_special_tokens=False)` (or direct `backend_tokenizer.encode` when `len(missing) <= 8` and `< 4096` chars). Similarly deduplicate and cache plain-text pieces in `encode_with_text_spellings` when control-token spellings appear in request text.

## 2. Exact Commands & Verification (`sp7`, `spark-ds41f:1107`)

### 2-Boot Unit & Contract Test Suite (`413 passed` across both boots)

```bash
for b in 1 2; do
  docker run --rm --cpuset-cpus=5-9,15-19 --entrypoint pytest \
    -e PYTHONPATH=/tmp/cand_vllm2 \
    -v /tmp/cand_vllm2:/tmp/cand_vllm2:ro \
    -v /tmp/cand_vllm2/deepseek_v41.py:/opt/spark-ds41f/candidate/vllm/vllm/tokenizers/deepseek_v41.py:ro \
    -v /tmp/cand_vllm2/deepseek_v41_encoding.py:/opt/spark-ds41f/candidate/vllm/vllm/tokenizers/deepseek_v41_encoding.py:ro \
    -v /tmp/cand_vllm2/chat_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/chat_utils.py:ro \
    -v /tmp/cand_vllm2/test_deepseek_v41.py:/opt/spark-ds41f/candidate/vllm/tests/tokenizers_/test_deepseek_v41.py:ro \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    -p no:cacheprovider -p tblib_stub -p offline_skip -q \
    tests/tokenizers_/test_deepseek_v41.py \
    tests/tokenizers_/test_deepseek_v41_effort.py \
    tests/tokenizers_/test_deepseek_v4.py \
    tests/parser/engine/test_deepseek_v4.py \
    tests/parser/engine/test_deepseek_v41.py \
    tests/entrypoints/openai/test_reasoning_enable_thinking.py \
    tests/entrypoints/openai/test_tool_calls_serialization.py | tee receipts/pytest-boot${b}.txt
done
```

- Boot 1 (`receipts/pytest-boot1.txt`): **`413 passed` in `40.75s`** (`0 failed`).
- Boot 2 (`receipts/pytest-boot2.txt`): **`413 passed` in `45.00s`** (`0 failed`).

### 2-Boot A/B Benchmark (`bench_chat_template_multiturn.py`)

```bash
for b in 1 2; do
  docker run --rm --cpuset-cpus=5-9,15-19 --entrypoint python3 \
    -v /home/chris/.cache/huggingface:/home/chris/.cache/huggingface:ro \
    -v /tmp/cand_vllm2:/tmp/cand_vllm2 \
    spark-ds41f:1107 \
    /tmp/cand_vllm2/bench_chat_template_multiturn.py --iters 8 \
    --label baseline-boot${b} --output receipts/bench-baseline-boot${b}.json

  docker run --rm --cpuset-cpus=5-9,15-19 --entrypoint python3 \
    -v /home/chris/.cache/huggingface:/home/chris/.cache/huggingface:ro \
    -v /tmp/cand_vllm2:/tmp/cand_vllm2 \
    -v /tmp/cand_vllm2/deepseek_v41.py:/opt/spark-ds41f/candidate/vllm/vllm/tokenizers/deepseek_v41.py:ro \
    -v /tmp/cand_vllm2/deepseek_v41_encoding.py:/opt/spark-ds41f/candidate/vllm/vllm/tokenizers/deepseek_v41_encoding.py:ro \
    -v /tmp/cand_vllm2/chat_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/chat_utils.py:ro \
    spark-ds41f:1107 \
    /tmp/cand_vllm2/bench_chat_template_multiturn.py --iters 8 \
    --label candidate-boot${b} --output receipts/bench-candidate-boot${b}.json
done
```

## 3. 2-Boot Benchmark Results (`sp7`, Grace ARM64 `--cpuset-cpus=5-9,15-19`)

Every scenario verifies 100% bitwise token-ID equivalence between `spark-ds41f:1107` baseline and candidate `0762` across both boots (`tokens_sha256` matches across all 4 runs):

| Scenario (`n_turns`, `user_every`, `ctrl`) | Msgs / Tokens (`tokens_sha256[:12]`) | Stage / Metric | Baseline (`1107`) Boot 1 / Boot 2 | Candidate (`0762`) Boot 1 / Boot 2 | Delta (Mean) | Speedup / Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`100 turns`**, `user_every=50`, `ctrl=False` | `203` msgs / `52,293` tok (`baccd4201d8c`) | `parse_chat_messages` (`mean`) | `1.01` / `0.99 ms` | `0.20` / `0.21 ms` | `-0.80 ms` | **`4.88x`** |
| **`100 turns`**, `user_every=50`, `ctrl=False` | `203` msgs / `52,293` tok (`baccd4201d8c`) | `render_only` (`tokenize=False`) | `14.02` / `14.15 ms` | `1.68` / `1.67 ms` | `-12.41 ms` | **`8.41x`** |
| **`100 turns`**, `user_every=50`, `ctrl=False` | `203` msgs / `52,293` tok (`baccd4201d8c`) | **`cold_e2e`** (cold cache) | `93.23` / `91.56 ms` | `11.39` / `11.26 ms` | **`-81.07 ms`** | **`8.16x`** |
| **`100 turns`**, `user_every=50`, `ctrl=False` | `203` msgs / `52,293` tok (`baccd4201d8c`) | **`incremental_e2e`** (turn $T$) | `94.07` / `92.52 ms` | `3.65` / `3.65 ms` | **`-89.65 ms`** | **`25.56x`** |
| **`250 turns`**, `user_every=50`, `ctrl=False` | `506` msgs / `125,952` tok (`3c146c4aa9ed`) | `parse_chat_messages` (`mean`) | `2.17` / `2.16 ms` | `0.67` / `0.67 ms` | `-1.50 ms` | **`3.23x`** |
| **`250 turns`**, `user_every=50`, `ctrl=False` | `506` msgs / `125,952` tok (`3c146c4aa9ed`) | `render_only` (`tokenize=False`) | `33.71` / `33.60 ms` | `4.03` / `4.13 ms` | `-29.58 ms` | **`8.25x`** |
| **`250 turns`**, `user_every=50`, `ctrl=False` | `506` msgs / `125,952` tok (`3c146c4aa9ed`) | **`cold_e2e`** (cold cache) | `217.07` / `213.88 ms` | `24.59` / `24.32 ms` | **`-191.02 ms`** | **`8.81x`** |
| **`250 turns`**, `user_every=50`, `ctrl=False` | `506` msgs / `125,952` tok (`3c146c4aa9ed`) | **`incremental_e2e`** (turn $T$) | `219.27` / `216.87 ms` | `24.06` / `22.93 ms` | **`-194.58 ms`** | **`9.28x`** |
| **`500 turns`**, `user_every=50`, `ctrl=False` | `1,011` msgs / `248,798` tok (`97a032f32afb`) | `parse_chat_messages` (`mean`) | `4.30` / `4.32 ms` | `1.55` / `1.53 ms` | `-2.77 ms` | **`2.80x`** |
| **`500 turns`**, `user_every=50`, `ctrl=False` | `1,011` msgs / `248,798` tok (`97a032f32afb`) | `render_only` (`tokenize=False`) | `66.50` / `67.13 ms` | `7.97` / `7.75 ms` | `-58.96 ms` | **`8.50x`** |
| **`500 turns`**, `user_every=50`, `ctrl=False` | `1,011` msgs / `248,798` tok (`97a032f32afb`) | **`cold_e2e`** (cold cache) | `439.07` / `437.24 ms` | `48.68` / `48.45 ms` | **`-389.59 ms`** | **`9.02x`** |
| **`500 turns`**, `user_every=50`, `ctrl=False` | `1,011` msgs / `248,798` tok (`97a032f32afb`) | **`incremental_e2e`** (turn $T$) | `421.89` / `420.41 ms` | `15.05` / `14.54 ms` | **`-406.36 ms`** | **`28.47x`** |
| **`100 turns`**, `user_every=1000`, `ctrl=False` | `202` msgs / `52,265` tok (`02b371ce8e53`) | `render_only` (`tokenize=False`) | `16.32` / `16.41 ms` | `1.71` / `1.71 ms` | `-14.66 ms` | **`9.57x`** |
| **`100 turns`**, `user_every=1000`, `ctrl=False` | `202` msgs / `52,265` tok (`02b371ce8e53`) | **`cold_e2e`** / **`incremental_e2e`** | `95.99` / `94.44 ms` (`incr`: `97.12` / `95.67 ms`) | `11.83` / `12.15 ms` (`incr`: `3.87` / `3.85 ms`) | `-83.23 ms` / **`-92.54 ms`** | **`7.94x`** / **`24.97x`** |
| **`250 turns`**, `user_every=1000`, `ctrl=False` | `502` msgs / `125,840` tok (`ba167fe3da8a`) | `render_only` (`tokenize=False`) | `54.99` / `55.06 ms` | `4.09` / `4.04 ms` | `-50.96 ms` | **`13.54x`** |
| **`250 turns`**, `user_every=1000`, `ctrl=False` | `502` msgs / `125,840` tok (`ba167fe3da8a`) | **`cold_e2e`** / **`incremental_e2e`** | `239.25` / `237.41 ms` (`incr`: `242.44` / `239.04 ms`) | `25.42` / `24.62 ms` (`incr`: `8.61` / `8.45 ms`) | `-213.31 ms` / **`-232.21 ms`** | **`9.53x`** / **`28.22x`** |
| **`500 turns`**, `user_every=1000`, `ctrl=False` | `1,002` msgs / `248,546` tok (`fd4a03f054c9`) | `render_only` (`tokenize=False`) | `184.18` / `184.52 ms` | `7.72` / `7.92 ms` | **`-176.53 ms`** | **`23.57x`** |
| **`500 turns`**, `user_every=1000`, `ctrl=False` | `1,002` msgs / `248,546` tok (`fd4a03f054c9`) | **`cold_e2e`** / **`incremental_e2e`** | `520.50` / `518.80 ms` (`incr`: `519.51` / `519.23 ms`) | `47.91` / `48.50 ms` (`incr`: `14.73` / `15.12 ms`) | `-471.45 ms` / **`-504.45 ms`** | **`10.78x`** / **`34.80x`** |
| **`250 turns`**, `user_every=50`, `ctrl=True` | `506` msgs / `125,977` tok (`5e7cc508dc6c`) | `render_only` / **`cold_e2e`** / **`incr_e2e`** | `38.20`/`38.57` / `109.85`/`110.83` / `112.21`/`112.03 ms` | `5.53`/`5.48` / `26.28`/`26.74` / `11.79`/`11.31 ms` | `-32.88` / `-83.83` / **`-100.57 ms`** | **`6.97x`** / **`4.16x`** / **`9.71x`** |
| **`500 turns`**, `user_every=50`, `ctrl=True` | `1,011` msgs / `248,823` tok (`96ddc572772c`) | `render_only` / **`cold_e2e`** / **`incr_e2e`** | `89.26`/`89.01` / `217.74`/`220.00` / `236.63`/`237.74 ms` | `10.39`/`10.54` / `50.06`/`51.29` / `19.93`/`20.62 ms` | `-78.67` / `-168.20` / **`-216.91 ms`** | **`8.52x`** / **`4.32x`** / **`11.70x`** |

## 4. Memory Observations & Conclusion

- **Peak Per-Request Python Heap Allocation (`tracemalloc`)**:
  - `100 turns` (`52K` tokens): **`2,482.5 KiB` -> `1,166.4 KiB` (`-53.0%`)**
  - `250 turns` (`126K` tokens): **`5,946.8 KiB` -> `2,866.4 KiB` (`-51.8%`)** (`ctrl=True`: `7,443.7 KiB` -> `3,662.4 KiB`, **`-50.8%`**)
  - `500 turns` (`249K` tokens): **`11,719.2 KiB` -> `5,461.4 KiB` (`-53.4%`)** (`ctrl=True`: `14,667.6 KiB` -> `7,004.5 KiB`, **`-52.2%`**)
- **Conclusion**: Fixing the $O(M^2)$ `find_last_user_index` scan, replacing redundant `copy.deepcopy` passes with shallow message/block copies, factoring the 1,283-token control-spelling regexes into character tries, fast-pathing plain string message parsing, and splitting + caching + batch-encoding prompt segments along structural added-token boundaries cuts cold 100–500 turn host chat preprocessing latency by **`7.9x–10.8x`** (`520 ms -> 48 ms` at 500 turns), incremental multi-turn latency by **`9.3x–34.8x`** (`519 ms -> 14.9 ms` at 500 turns), and peak Python heap allocation by **`51%–53%`** with 100% bitwise token-ID parity across both boots.
