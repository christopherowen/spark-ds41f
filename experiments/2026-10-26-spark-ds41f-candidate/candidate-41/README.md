# Candidate-41: simpler copy preparation

One runtime variable on corrected default-enabled C40: remove optional C39
copy-preparation stream/event overlap and prepare on the current stream before
DSpark. Preserve fused copy kernels/policy, the safe envelope, default
enablement, older async allocation D2H delivery and PR878.

The focused rework review withdrew event reuse before measurement: it has no
defensible serving-gain bound. This candidate is declared cleanup/simplification
before image or GPU work. Neutral performance is acceptable; full correctness
and no material incremental performance regression versus BOTH C40 boots are
required. No speedup is claimed.

Qualification and all outcomes belong under
`experiments/2026-11-04-copy-overlap-rework`. Source replay is pending. No image,
GPU or serving run has started, and no release promotion is implied.
