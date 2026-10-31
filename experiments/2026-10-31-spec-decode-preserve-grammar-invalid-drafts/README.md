# Preserve Grammar-Invalid Drafts for Rejection Sampling (`#61557`)

## Upstream Source

- Upstream commit: `vllm-project/vllm@c4f2bb4abb` (`[Bugfix][Spec Decode] Preserve grammar-invalid drafts for rejection sampling (#61557)`)
- Reconciled with `rc21` (`38449bdb9d98948b8500fbf6b96ab636c7c69671`), including `29f662259` (`Fix deferred grammar rejection in v2 speculative sampling`) and `2d832d876` (`Hold scheduler grammar state at committed tokens`).

## Problem

When structured output (grammar constraints) is active during speculative decoding at `temperature > 0`:
1. In `rejection_sampler_utils._resample_kernel`, when a draft token is pre-stripped and replaced with `-1` (`rejected_draft_token < 0`), the kernel sets `residual_logits = target_logits` instead of computing the exact residual distribution $\max(p_{\text{target}} - q_{\text{draft}}, 0)$.
2. Pre-stripping the first grammar-invalid draft token before GPU rejection sampling therefore shifts the sampling distribution away from the target model's constrained distribution.
3. Furthermore, `XGrammarGrammar.accept_tokens` and `GuidanceGrammar.accept_tokens` did not atomically roll back partial token sequences on rejection, and `XGrammarGrammar.accept_tokens` logged false-alarm `ERROR` messages when probing speculative draft tokens.

## Changes

- **`vllm/v1/structured_output/__init__.py` (`trim_spec_tokens`, `grammar_bitmask`)**:
  - When a draft token violates an active (non-terminated) grammar state, preserve the first grammar-invalid draft token (`spec_tokens[: constraint_start + len_validated + 1]`) so the active grammar bitmask row masks it to $-\infty$ on the GPU and rejection sampling sees the true `rejected_draft_token >= 0` for exact residual resampling, while invalidating any unreachable drafts beyond it (`num_invalid_spec_tokens`).
  - When a valid draft token is a stop/EOS token that terminates the grammar (`grammar.is_terminated()`), continue invalidating all trailing drafts after EOS so unconstrained post-termination bitmask rows (`_full_mask`) cannot be accepted on the GPU (`29f662259`).
- **`vllm/v1/structured_output/backend_xgrammar.py` & `backend_guidance.py`**:
  - Make `accept_tokens()` atomic (rolling back any tokens accepted before a failure in the same call) and remove the false-alarm `logger.error` in `XGrammarGrammar.accept_tokens`.

## Verification

Verified in the `spark-ds41f:19f2c20ed4d6-next-21` container on `dgx4`:
- `tests/v1/structured_output/test_grammar_backends.py`
- `tests/v1/structured_output/test_structured_output_manager.py`
- `tests/v1/engine/test_structured_output_draft_handoff.py`
- `tests/v1/core/test_sink_draft_zero.py`
- `tests/v1/spec_decode/test_mtp_structured_output.py`
(`147 passed, 1 xfailed`).
