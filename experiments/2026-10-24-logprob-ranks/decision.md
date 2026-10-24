# Decision

Retain the selected-token vocabulary-rank port as a kernel-qualified candidate.
Both final 15-shape paired performance runs pass; the focused tests and CUDA
memcheck pass. Rank outputs preserve the reference's semantics exactly.

Use 256 threads for FP32 and 512 for FP16/BF16. Reject a universal 512-thread
configuration because the 96-row FP32 case measured slower. Keep every receipt,
including the rejected run and the legacy lab-parser false failure.

Do not infer an overall serving throughput gain from the kernel's speedup.
A release image and the full DS41F serving comparison have not been run.
The production recipe and promoted source lock are unchanged.
