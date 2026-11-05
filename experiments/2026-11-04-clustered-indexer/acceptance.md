# Current LSA acceptance criteria

Owner revision,2026-11-04: equivalence is the **generated token sequence**.
Selected keys, cluster membership, scores, intermediate values and attention
arithmetic may differ. Base checkpoint weights and tokenizer remain unchanged;
auxiliary selector maps may be fitted separately. Earlier exact-selection
experiments retain their original criteria and results.

For the same checkpoint, tokenizer, tokenized request and deterministic generation
settings, require the same ordered generated token IDs, count and termination.
Include reasoning and tool-call tokens when present. Report every mismatch and
the predeclared workload coverage. A finite suite establishes its tested scope,
not universal equivalence. Changed generated tokens still require the owner's
previously requested evidence of superior quality before acceptance.

Correctness comparisons run **out of band**. The timed candidate must not execute
the native reference, token/logit comparisons, exclusion-proof bounds, diagnostic
reductions, duplicate scoring for validation, or activation copies to the host.
Use separate validation runs and retain those receipts. Essential index
construction/update work belongs in the total cost; report it separately from
query cost and whole-request latency so it cannot disappear outside the timer.

Existing serving receipts compare response digests, completion counts and finish
reasons, but do not record raw generated token IDs. Do not retroactively call
them a token-ID equivalence gate or recover IDs by retokenizing decoded text.
Future changed-selection candidates need explicit token-ID capture/comparison,
deterministic free-running tests and separate matched performance runs. Require
zero token mismatches for the declared gate; similar aggregate accuracy alone
does not replace it.

The target remains substantial request-level acceleration, repeated candidate
boots and uncertainty. Released builds are not timed again. Runtime-window,
memory/admission, coordinated-restore, publication and promotion rules remain.

Diagnostic request timing may proceed after an explicitly recorded token mismatch.
Such timings describe the failed candidate and cannot establish an accepted
speedup. Input identity errors, missing raw IDs or malformed responses remain
errors. Fidelity and performance are reported independently; acceptance requires
both. Native repeat captures check whether the reference itself is stable.
