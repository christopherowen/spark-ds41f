# 2026-11-22: Long-Context OpenCode DSML Control-Token Reproducer & Fixes

## Base Identity

- Deployment commit: `3fa0d0a812` (`candidate-67`)
- Promoted image: `spark-ds41f:1120` (`sha256:5c12472565a708518ec0566b6653b5bd324e06881a34b90f69d6b8210a386d08`)
- Candidate base for patches & overlay: `spark-ds41f:candidate-67` (`sha256:22725ab3a3242b16ea23eb59c5d0ee9031b5c7c6559b7837cf3d7228a8a583d4`, vLLM `0001..0850`, `patch_head = 629d215f7b97d9909eaa31df542a599666fb6188`, `expected_tree = d2fefd95b6bfb59cf3a3d2874b919002b472604d`)
- Model / tokenizer: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`tokenizer_mode="deepseek_v41"`, `max_model_len=1048576`)

## Observed Symptom (`597.8K` OpenCode Session)

In a multi-turn OpenCode `Build` session (`wispy-oasis`) at `597.8K (57%)` context (`reasoning_effort="high"`, `7 subagents · goal`), two consecutive `bash` tool turns exhibit:

1. `<think>...</think>` parsed cleanly into `reasoning` (`+ 1 thought` in UI).
2. One-sentence assistant prose emitted in `content`, followed directly in `content` (with `tool_calls == []`) by a **headless** DSML parameter/closer block missing both `<｜DSML｜ calls>` and `<｜DSML｜ invoke name="bash">`, and missing `string="true"` / `string="false"` attributes:
   ```text
   <｜DSML｜ parameter name="command">cd /Users/herlon/dev/woncadigital/workspace/multitracking && echo "typecheck: $(bun run typecheck 2>&1 | grep -cE 'error TS')"; bun test src/core/mint.test.ts scripts/bench-5rps.test.ts 2>&1 | tail -3</｜DSML｜ parameter>
   <｜DSML｜ parameter name="timeout">400000</｜DSML｜ parameter>
   </｜DSML｜ invoke>
   </｜DSML｜ calls>
   ```

## Part 1: Deterministic Parser & Prompt-Encoding Reproduction

1. **Unpatched Failure Characterization (`test_dsml_screenshot_repro.py` — `43/43` Passed on `1120`):**
   - `ParserState.CONTENT` defines tool-entry transitions only on `TOOL_START` (`<｜DSML｜ calls>`) and `INVOKE_PREFIX` (`<｜DSML｜ invoke name="`), so `PARAM_START`, `PARAM_CLOSE`, `INVOKE_END`, and `TOOL_END` leak verbatim into `assistant.content`.
   - `ParserState.TOOL_PREAMBLE` has `content_events[TOOL_PREAMBLE] is None` and no exit on `PARAM_START`, `INVOKE_END`, or `TOOL_END`, silently dropping the rest of the turn when `<｜DSML｜ invoke>` is omitted or unspaced.
   - Replaying leaked `assistant.content` strips token `128825` via `_ControlSpellings`, while non-empty `tools` forces `effective_drop_thinking = False` and injects `<｜Assistant｜><think></think>` on every completed assistant turn lacking `reasoning_content`, and streamed `assistant.content` ending in `\n\n` doubles to `\n\n\n\n<｜DSML｜ calls>` on replay.
2. **Target 7-Contract Suite (`test_dsml_fixes_contract.py` — `63` Cases):**
   - **Unpatched `spark-ds41f:candidate-67`:** `58 failed, 5 passed` (all `5` non-regression invariants pass; all `58` target contract checks fail).
   - **Patched `spark-ds41f:candidate-67` (`0851` + `0852`):** `63 passed, 0 failed`.

## Part 2: Live Unpatched `600K` GPU Sweeps & Incremental Warm-vs-Cold Parity (`153` Requests)

Summary artifacts in `runs/`:
- `runs/c64-temp1-summary.jsonl` & `runs/c64-temp0-summary.jsonl` (`spark-ds41f:candidate-64`, `80` single-shot requests across `4K`, `131K`, `600K`)
- `runs/1120-sparse-summary.jsonl` (`spark-ds41f:1120`, `32` single-shot requests at `4K` & `602K`)
- `runs/1120-warm-vs-cold-summary.jsonl` (`spark-ds41f:1120`, `41` incremental multi-turn requests from `755` → `597,198` tokens)

| Milestone (`1120` Incremental) | Prompt Tokens | Warm Cached Tokens | Cold Cached Tokens | Greedy Tokens (Warm / Cold) | Exact Token Parity | Max Abs Logprob Diff |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: |
| `65K` | `70,112` | `48,128` | `0` | `673 / 673` | `True` (`0` diffs) | `0.000000` |
| `262K` | `267,191` | `245,248` | `0` | `491 / 491` | `True` (`0` diffs) | `0.000000` |
| `580K` | `597,198` | `574,464` | `0` | `39 / 39` | `True` (`0` diffs) | `0.000000` |

- **Zero KV / SWA / SparseMLA Prefix-Cache Drift:** Warm cached generation (`96.2%` cache hit at `597,198` tokens) is bit-exact (`0.000000` max logprob diff) against cold full recompute.
- **Multi-Turn `<think></think>` + `\n\n\n\n` Imitation Confirmed Live:** `5/5` tool-calling turns streamed `assistant.content` ending in `\n\n`, and accumulating `<｜Assistant｜><think></think>` across turns with omitted `reasoning_content` drove immediate `</think>` zero-reasoning skips (`reasoning_chars == 0`) from `1/80` (`1.2%`) in single-shot prefill to `5/27` (`18.5%`) live turns, `1/1` greedy parity turns at `597,198` tokens, and **`5/8` (`62.5%`)** sampled frontier trials.

## Part 3: Patch Series Rebased onto `candidate-67` (`vllm/` & `tp3-c67-dsml.json`)

`vllm/series` layers two commits (`0851..0852`, `725` total patches) onto `candidate-67` (`629d215f7b97d9909eaa31df542a599666fb6188`):

1. `vllm/0851-fix-dsv41-avoid-empty-historical-think-blocks-and-do.patch` (`vllm/tokenizers/deepseek_v41_encoding.py`, `vllm/tokenizers/deepseek_v41.py`, `tests/tokenizers_/test_deepseek_v41.py`):
   - Renders completed (`not wo_eos`) assistant turns lacking `reasoning_content` as `<｜Assistant｜></think>` rather than `<｜Assistant｜><think></think>`, preserving non-empty `reasoning_content` and `prefix: True` (`wo_eos`) continuations.
   - Trims one trailing `\n\n` on `assistant.content` before `\n\n<｜DSML｜ calls>`.
   - Prevents an explicit `"reasoning": None` key from clobbering populated `"reasoning_content"` during message normalization.
2. `vllm/0852-fix-dsv41-recover-headless-DSML-parameters-spacing-v.patch` (`vllm/parser/deepseek_v41.py`, `docs/features/tool_calling.md`, `tests/parser/test_deepseek_v41_structural_closers.py`):
   - Exits `ParserState.TOOL_PREAMBLE` cleanly on `TOOL_END` or non-whitespace plain text.
   - Accepts `128825`-gated DSML spacing/dialect variants (`<｜DSML｜calls>`, `<｜DSML｜ call>`, `<｜DSML｜ tool_calls>`, V4 unspaced `<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="`, `<｜DSML｜parameter`, `function_calls`/`function`, optional whitespace before `>`, and closers) while keeping BPE-spelled markup as literal `content` on `tool_choice="auto"`.
   - Closes `<think>` implicitly when a `128825`-gated `<｜DSML｜ invoke name="` appears directly inside `ParserState.REASONING`.
   - Buffers genuine-`128825` headless `<｜DSML｜ parameter ...>` blocks outside `TOOL_ARGS` (supporting unique schema recovery, `parameter`-substituted-for-`invoke` headers, parallel headless calls, in-`<think>` headless calls, mid-value EOS truncation, and fail-closed quarantine on ambiguous/unknown parameters).
   - Under forced/named `tool_choice`, keeps `128825` provenance enforced inside `<think>` while accepting post-`</think>` DSML markers spelled with ordinary tokens so string-based structural tags always extract as tool calls.

## CPU Verification Receipts (`candidate-67`)

- Fresh prepared tree `git am` + `git diff --check`: `cd8c95b3e2249ae0ebb5fce5c1eca1e0122407b1` (tree `740f884e927d790cdc48e5b6fe60056a6a0925c1`, `CLEAN`).
- Multi-boot CPU test matrix on `spark-ds41f:candidate-67` (`test_dsml_fixes_contract.py` `63` cases + all DeepSeek V4/V4.1 tokenizer, renderer-pool, parser, provenance, structural-closer, tool-parser, and custom-tool suites = **`926` tests per boot**):
  - `spark-ds41f:candidate-67` boot 1: **`926 passed` (`19.12s`)**
  - `spark-ds41f:candidate-67` boot 2: **`926 passed` (`18.67s`)**

## Live `tp3-triangle` Overlay Verification (`tp3-c67-dsml.json`)

Measured across two clean cluster boots on `dgx1`–`dgx3` inside `spark-lab` window `dgx1-1795363437` (`MemAvailable = 117.8–118.3 GiB`, `config/cluster.json` restored and verified via `doctor --live` on close):

1. **2-Boot Live `/v1/chat/completions` DSML Matrix (`verify_live_dsml_matrix.py`, `13` cases per boot):**
   - `runs/c67-dsml-boot1-matrix.jsonl`: **`13/13 passed` (`all_passed: true`, `0` DSML leaks)**
   - `runs/c67-dsml-boot2-matrix.jsonl`: **`13/13 passed` (`all_passed: true`, `0` DSML leaks)** — deterministic greedy outputs, `reasoning_chars` (`183`, `0`, `0`, `375`, `2602`), and loop counts (`8/8`, `8/8`, `8/8` tool calls; `0/8` zero-reasoning turns on active-turn `reasoning_content` replay) matched identically across both boots:
     - `auto_bash_{whole,stream}`: valid `bash` call (`timeout=400000`), `reasoning_chars=183`, `0` leaks.
     - `required_tool_choice_{whole,stream}` (`xgrammar` structural tag): valid `bash` + `read` calls, `0` leaks.
     - `named_tool_choice_{whole,stream}` (`xgrammar` structural tag): valid `read` call on `src/carriers/royalmail/index.ts`, `0` leaks.
     - `quoted_dsml_in_write_arg_{whole,stream}`: literal `</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\nEOF\n` preserved intact inside `write.arguments`, `0` leaks.
     - `tool_choice_none_prose_{whole,stream}`: `calls=[]`, `reasoning_chars=2602`, markdown DSML example preserved as plain text.
     - `intraturn_active_reasoning_loop` (`8` turns): `8/8` tool calls, **`0/8` zero-reasoning turns**, `0/8` DSML leaks.
     - `multiturn_user_omitted_reasoning_loop` & `intraturn_omitted_reasoning_resilience` (`8` turns each): `8/8` tool calls, `0/8` DSML leaks.
2. **Incremental `600K` Warm-vs-Cold Parity & Sampled Frontier Sweep (`runs/c67-dsml-warm-vs-cold-summary.jsonl`, `41` requests from `755` → `598,261` tokens):**

| Milestone (`tp3-c67-dsml`) | Prompt Tokens | Warm Cached Tokens | Cold Cached Tokens | Greedy Tokens (Warm / Cold) | Exact Token Parity | Max Abs Logprob Diff | Reasoning Chars |
| :--- | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| `65K` | `70,244` | `48,128` | `0` | `462 / 462` | `True` (`0` diffs) | `0.000000` | `1,340` |
| `262K` | `266,823` | `244,992` | `0` | `412 / 412` | `True` (`0` diffs) | `0.000000` | `743` |
| `580K` | `598,261` | `575,488` | `0` | `157 / 157` | `True` (`0` diffs) | `0.000000` | `147` (vs `0` on `1120`) |

   - **Bit-Exact `600K` Warm-vs-Cold Parity Maintained:** `3/3` checkpoints (`70,244`, `266,823`, `598,261` tokens) matched `100%` token-for-token with `0.000000` max abs logprob diff (`5.160s` warm vs `80.620s` cold at `598,261` tokens).
   - **Zero DSML Failures Across All `41` Long-Context Requests:** `0/27` live incremental turns, `0/6` parity turns, and `0/8` sampled frontier turns leaked DSML markup.
   - **Substantial Reduction in Omitted-Reasoning Zero-Think Skips at `598K`:** Even with `reasoning_content` omitted across all `27` historical turns, `598K` greedy parity recovered non-empty reasoning (`147` chars vs `0` on `1120`), live-turn zero-think skips fell from `5/27` (`18.5%`) to `3/27` (`11.1%`), and sampled `598K` frontier zero-think skips dropped from **`5/8` (`62.5%`)** on unpatched `1120` to **`1/8` (`12.5%`)** (`-80%` relative reduction).
