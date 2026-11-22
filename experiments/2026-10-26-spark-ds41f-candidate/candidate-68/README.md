# Candidate `candidate-68`

Date: 2026-11-22.
Base: `candidate-67` (`vLLM 0001..0850`, tree `d2fefd95b6bfb59cf3a3d2874b919002b472604d`).

## Delta Over `candidate-67`

1. **vLLM patch `0851`** (`0851-fix-dsv41-avoid-empty-historical-think-blocks-and-do.patch`):
   - Renders completed (`not wo_eos`) assistant turns lacking `reasoning_content` as `<｜Assistant｜></think>` rather than `<｜Assistant｜><think></think>`, eliminating out-of-distribution empty `<think></think>` blocks across multi-turn agent histories while preserving non-empty `reasoning_content` and `prefix: True` (`wo_eos`) continuations.
   - Trims one trailing `\n\n` on `assistant.content` before `\n\n<｜DSML｜ calls>` so streamed tool-calling prose round-trips without `\n\n\n\n` blank-line doubling.
   - Prevents an explicit `"reasoning": None` key on message dicts from clobbering populated `"reasoning_content"` during message normalization.
2. **vLLM patch `0852`** (`0852-fix-dsv41-recover-headless-DSML-parameters-spacing-v.patch`):
   - Exits `ParserState.TOOL_PREAMBLE` cleanly on `TOOL_END` or non-whitespace plain text.
   - Accepts `128825`-gated DSML spacing and dialect variants (`<｜DSML｜calls>`, `<｜DSML｜ call>`, `<｜DSML｜ tool_calls>`, `<｜DSML｜tool_calls>`, `<｜DSML｜invoke name="`, `<｜DSML｜parameter`, `function_calls`/`function`, optional whitespace before `>`, and closers) while preserving literal plain-text emission for BPE-spelled markup under `tool_choice="auto"`.
   - Implicitly closes `<think>` when `128825`-gated `<｜DSML｜ invoke name="` appears inside `ParserState.REASONING`.
   - Buffers genuine-`128825` headless `<｜DSML｜ parameter ...>` blocks outside `TOOL_ARGS` (unique `request.tools` schema recovery, `parameter`-substituted-for-`invoke` headers, parallel headless calls, in-`<think>` headless calls, mid-value EOS truncation, and fail-closed quarantine on ambiguous/unknown parameters).
   - Under forced or named `tool_choice` (`request.tool_choice not in (None, "auto", "none")`), keeps `128825` provenance enforced inside `<think>` while accepting post-`</think>` DSML markers spelled with ordinary BPE tokens by string-level `xgrammar` structural tags.
3. **Live `tp3-triangle` overlay evidence** ([2026-11-22-dsml-long-context-repro](../../../2026-11-22-dsml-long-context-repro/README.md), `PR #1312`–`#1315`):
   - Two clean boots of `tp3-c67-dsml.json` passed **`13/13` live `/v1/chat/completions` DSML cases** (`auto`, `required`, named `tool_choice`, quoted DSML closers in `write`, `tool_choice="none"` prose, and multi-turn active/omitted reasoning loops) with `0` DSML leaks and deterministic greedy parity across boots.
   - Incremental `600K` (`755` -> `598,261` tokens) Warm-vs-Cold parity held bit-exact (`0.000000` max abs logprob diff across `65K`, `262K`, `580K`), `0/41` DSML failures, `598K` greedy reasoning `0 -> 147` chars, and sampled `598K` frontier zero-think skips dropped from `62.5%` (`5/8`) to `12.5%` (`1/8`).

## Build & Verification Status

Built `spark-ds41f:candidate-68` (`sha256:05ee1eb162faa8813818093f1a18aeb46c43caf4fc62a7dcf77ef7434976ef02`, vLLM `0001..0852`, `725` patches, `patch_head = cd8c95b3e2249ae0ebb5fce5c1eca1e0122407b1`, `expected_tree = 740f884e927d790cdc48e5b6fe60056a6a0925c1`, `patchset_sha256 = 1b0ae795595ec0abcd768e6261636fb60e151f2aebfd6ac0ca1cf5fb20a0a735`) in `493 s` on `dgx4` (lowest `MemAvailable = 82.1 GiB`).

- **Zero-overlay built-image unit tests on `dgx4` (`--gpus all`)**: **`739 passed, 0 failed`** across two independent runs (`23.07 s`, `22.71 s`) of the complete DSML contract, DeepSeek V4/V4.1 tokenizer, effort, renderer-pool, structural-closer, provenance, engine-parser, and tool-parser suites, plus **`78 passed, 1 skipped, 0 failed`** (`31.13 s`) across `test_deepseek_v41_postload_startup.py`, `test_jit_warmup_migration.py`, `test_adaptive_verification.py`, `test_deepseek_v41_renderer_pool.py`, and `test_responses_custom_tools.py` (`795` unique tests total).
- **Distribution**: Identical OCI digest `sha256:05ee1eb162faa8813818093f1a18aeb46c43caf4fc62a7dcf77ef7434976ef02` verified on all four nodes (`dgx1`, `dgx2`, `dgx3`, `dgx4`). Production `spark-ds41f:1120` on `dgx1..dgx3` remained untouched.
