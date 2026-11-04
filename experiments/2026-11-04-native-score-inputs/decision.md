# Decision

GPU case gate passed; retain as an unpromoted source candidate for full-model
and host enqueue qualification. PR #895 remains stacked on PR #893.

The final source `bce3375da1b478deabd64fe92a1ac4ac449131b7` reads the producer's native int32/int64 ids and
strides. Small transposed selections use the qualified block-sized shared stage;
larger selections retain the qualified native-stride program. No conversion or
layout materialization is restored, and no extra GPU launch or global buffer is
added. FP32 arithmetic and order, token values, ranks and wire ids are preserved.

All 120 GPU cases on both fresh final boots exactly match per-case reference and
candidate CUDA, inputs and strides that qualified on both prior timing boots:
112 cases from v6 and eight from v7. No unmatched or previously unqualified
program is accepted. Measured programs use the five-SE upper ratio at most 1,
with zero regression allowance; identical prerequisite programs are proven by
code and aliased preparation. The single unqualified v6 case changes, and the
larger v7 programs that regressed are excluded by the one-warp dispatch bound.

Both final boots pass 673 numerical, boundary, graph, prompt, speculative,
serialization and cold startup-freeze checks. Both expanded IKET captures pass
with 56 decoded launches, explicitly including shared staging and aligned vector
loads. Minimum sampled available host memory is 111.08 GiB, with no guard stop.
`runs/native-final.json`, `paired-results-final.json` and `performance-gate.json`
record the final source and proof chain. Every rejected source and unfavorable
observation remains separate and unchanged.

The full ordered patch series applies in a fresh source tree. Source composition
with the prior three ATen replacement candidates passes too. Doctor, export hashes,
parsing and diff checks pass; local full build inputs are absent. Full-image,
full-model and host enqueue qualification remain before promotion. No promoted
configuration, mode, source lock or immutable baseline changed; no serving
operation was performed.
