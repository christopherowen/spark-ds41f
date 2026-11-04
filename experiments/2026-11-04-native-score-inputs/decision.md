# Decision

Not ready: the equal-or-better performance gate is not passed. PR #895 is a
draft, stacked on PR #893. The owner requires fixing slower cases while
retaining the conversion removals.

The current source (`d9f9a007`, tree `796f702a`) consumes noncompact ids before
the large score reduction and stores native sampled ids before top-k selection.
It preserves the original generated kernel for compact int64 inputs. All paths
read their producer's existing storage without dtype/layout materialization.
FP32 score arithmetic, ordering, ranks and serialized Python ids are unchanged.

The v4 source passed 625 tests and 40 decoded IKET launches on each boot. Its
first controlled timing run left 74 of 120 cases unproved across the two boots;
the unchanged reference control also showed large timing variation. This run
is retained as diagnostic evidence, not an acceptance. A follow-up changes
only the timer: GPU-resident event nodes exclude host submission gaps and
measure eight cold-cache samples per replay. The same strict per-case gate
applies on both boots.

Earlier source variants and unfavorable runs remain unchanged. In particular,
v3's means increased in 13 of 120 cases, up to 5.95%; those cases are not passes.
The current source applies cleanly after candidate-38 and composes with the
three prior ATen replacement candidates.

Full-image/full-model qualification, including host enqueue effects, remains
before promotion. No promoted configuration, default mode, source lock or
immutable baseline is changed. No serving operation was performed.
