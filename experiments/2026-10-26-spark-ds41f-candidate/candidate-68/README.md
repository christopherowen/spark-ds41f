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
