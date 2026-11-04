# Decision

Retain for review and integration into a release candidate. Finish the TileLang
implementation of raw and processed logits output rather than disabling the modes.

The source candidate passes 87 tests on each of two boots, 36 paired operation
timing cases on each of two boots, and two native IKET captures with eager and
CUDA-graph records. Captured replacement operations use one TileLang GPU kernel
and no ATen kernel. Output and startup preparation contracts pass.

No promoted recipe, default mode, source lock or immutable baseline is changed.
The default remains raw_logprobs. Full-image/full-model release qualification
is still required before promotion; these isolated results do not establish a
serving-throughput improvement or qualification of the four replacement patches
executed together.
