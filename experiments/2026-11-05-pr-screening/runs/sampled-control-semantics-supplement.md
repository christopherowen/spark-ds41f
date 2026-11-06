# Sampled-control semantics supplement

The original audit is consistent with its declared predicate. C40 is not
retroactively invalidated by the sampled collector's `identical:false` value.
This separate note corrects any broader “all diagnostics/seeded batching controls
passed” wording; it changes no original audit, raw verdict or runtime.

The small saved C38 standard/copy, C39, C40 and C41 audits all record the same
pattern on both boots: `identical:false`, C8 differing prompts 0–7, isolated
`again` differing none, and sampled rows equal across boots. Direct small C39/C40
JSONL reads verify 527 bytes and SHA256
`101c40d681f6693ede8f0a774dddf114d17ef03d91ec2e8322726024889a0a25` against
their source receipts. This is not a newly isolated C40/C41 or row-cost effect.

`sampled_digest.py` explicitly says it records and exits 0 either way. Its hash
combines returned chosen-token strings and logprobs. A difference does not tell
which component differs and does not establish token corruption or a cause.
The older collector doc describes seed/position batch invariance, but the
campaign's analyzer explicitly defines a narrower acceptance contract:
`again.differing` must be empty; C8 variation and `identical:false` remain a
recorded limitation. The cross-boot sampled-row equality flag is recorded, not
itself a separate enforced predicate. It is true in every reviewed saved audit.

Accurate reporting is: “Planned process outcomes and experiment-declared semantic
gates passed. The observational seeded-sampling probe recorded C8 differences
on both boots; isolated repeats matched. Seeded batch invariance was not
established.” C39's separate native runtime-collector failures remain failures,
with its explicit identity revalidation kept separate.

Retain C40 as the best available validated corrected TP3 performance reference
within its declared gates, with this diagnostic limitation visible. This is no
automatic promotion or new TP4/release qualification. Do not infer a runtime
regression or globally hard-fail an explicitly observational collector.

Future plans must name which semantic predicates are gates and which are
observations before execution. A required gate needs exit 0 plus complete,
well-formed output satisfying every declared predicate; false required verdicts
must stop progression even with exit 0. Observations retain false/differing values
and limitations without a passed-gate badge. Keep the present isolated-repeat
predicate; any stronger future batch-invariance claim needs an explicit contract
and appropriate evidence. Existing exactness gates and adverse records remain
unchanged. Budget owner handles the separate recording/validator implementation;
this review adds no sampler code, tests, live probes or reruns.

See the adjacent JSON supplement for exact audit/source hashes, historical
collector blobs, selected fields and raw-file receipts. Only small local reads
and text edits were used. No PR925 helper was created or executed.
