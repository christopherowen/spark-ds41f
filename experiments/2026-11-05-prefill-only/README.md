# Compact prefill with native generation

Base `49971829`. One intended variable: use the existing native index plan for
`mode=decode`, including generation and speculative verification, while retaining
compact dense prefill and native source selection. This generic phase policy is
motivated by the measured single-query component penalty, reduced draft
acceptance and extra generation time. It does not assume that prefill captures
are representative decode operands, or that native generation guarantees better
answers after an approximate prefix.

Both plans are already declared and their scratch admitted before capture. The
new host branch selects a plan using the existing phase and a boot-time flag;
it adds no device comparison, reference work, host copy or model-aware grading.
Defaults and previous profiles are unchanged. Covered prefill and already-native
source/reindex paths keep their existing behavior.

Before any model use, apply the same ordered nine patches plus the one host
routing patch in a fresh named source tree. Require exact overlay equality,
CPU routing tests across1/8/64 rows, both modes, covered extents and existing
native paths. Native68 versus unwritten-prefix80 key/handoff equivalence has
already passed all12 actual batches under strict initialization. Existing
native fallback declarations/admission must remain present. Full model startup,
CUDA graph, independent answer, repeated-boot request and admission gates remain
required. This experiment changes the generation route only; neither the exact
fusion/maintenance changes nor a larger budget is added simultaneously.

Source `f5820da9` passes72 graph routes across all12 frozen captured batches,
rows1/8/64 and prefill/decode modes. Each graph replays its original visible
extent, an empty extent and the restored extent, matching the expected existing
plan's keys and source handoff exactly. Initialization checking reports zero
errors. These are captured prefill operands used to test dispatch/graph
correctness, not actual decode-query or request-speed evidence. Workspace is
preallocated outside capture; the test performs reference work only out of band.
The ten-patch fresh source application equals the host overlay. All evidence and
memory observations remain in `runs/graph-01`. The sp8 lease releases
at21:51:31 UTC; independent verification finds no reservations, containers or
compute processes. The model recipe remains unqualified and unpromoted.
