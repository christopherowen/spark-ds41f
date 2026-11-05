# Recorded stream collector

The released TP3 reference used client commit
`19efdd6fe1f11a159c5f1989aaa3433682325ab2`. Its stream reader records an event
when decoded content or reasoning is nonempty. Later collection also records
token-ID-only events and continuous-usage progress with empty decoded text.
That changes first-event timing, stream gaps and the tokens assigned to a
measurement window, even when the server sends exactly the same output.

The isolated serving controllers select
`SPARK_DS41F_BENCH_STREAM_POLICY=recorded-1030` for every timed suite. This
restores the reference event-selection rule. Suite identity and the controller
receipt record the policy; a missing/different policy fails the native suite
gate. Default collection remains `current`. The mode changes no HTTP request,
model setting, tensor, precision, sampling seed or inference result.

`recorded-events.json` was produced offline by the unmodified historical
collector on synthetic streams with a controlled clock. It pins that source's
hash and every returned field plus the complete wire request. Tests require
exact matching records under the recorded policy and retain current empty-delta
handling under the default policy. Unknown policies fail before a request.
`compatibility-audit.json` also records identical default prompt identity,
actual greedy/explicitly seeded sampled payloads, unchanged model/engine/prefill
sources and unchanged stall request builders. Stall body construction moved
before thread startup; its request parameters remain identical.

The initial scale window reached its first primary suite before the handoff
request was queued. That atomic suite completed and all original observations
remain under `lossless-scales-release-tp3`; the controller then restored C36.
`expert-scales/first-primary-results.json` retains the native suite, tests,
operator errors, restoration and offline exact model comparison. They are an unmatched-collector
observation, not the matched promotion comparison. No performance values were
reviewed before this decision. The new cohort uses
`lossless-scales-recorded-release-tp3`, two fresh boots and the exact recorded
suite definitions. The released image is never measured again.

The guarded restart retains the original image and passed component gate from
the pre-serving round-2 receipt. It runs a distinct full gate with a separate
bundle/output directory; original full-gate inputs and reports remain intact.
This avoids adopting a serving attempt as a gate-only receipt. Neither the
runtime source/image nor the intended expert-scale variable changes.
