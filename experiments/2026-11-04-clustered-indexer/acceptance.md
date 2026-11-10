# Current LSA acceptance criteria

Owner revision, 2026-11-05: **thinking/reasoning may diverge; the completed
actual answer is the acceptance target.** Reasoning-token mismatches alone do
not reject a candidate or require proof of superior answer quality. Preserve
reasoning captures as diagnostics, and charge all reasoning time to performance.

A run must reach a completed final answer before it can qualify final-answer
behavior. An empty answer after a truncated reasoning prefix is inconclusive,
not equivalence. The 128-token long-context receipts below are reasoning-prefix
measurements; their historical all-generated-token failures are not evidence of
incorrect final answers. No final-answer quality loss is authorized by this
revision. Changes to actual answers remain subject to the existing requirement
to preserve them unless superior quality is established.

Task correctness can be scored with predeclared objective checks on completed
answers. Useful throughput is the number of correctly completed tasks divided
by workload wall time, including thinking, failures and retries. At concurrency,
use elapsed workload wall time rather than summing overlapping request times.
A proposed quality-based comparison uses a lower confidence bound on the paired
quality difference against a predeclared tolerance, alongside a confidence bound
on useful speedup. No nonzero quality-loss tolerance or new promotion threshold
is adopted here. Evaluate important task/context groups separately.

## Historical 2026-11-04 generated-token criterion

The following criterion governed earlier runs. Its requirement to match reasoning
tokens is superseded by the owner's revision above; retained receipts are unchanged.

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
