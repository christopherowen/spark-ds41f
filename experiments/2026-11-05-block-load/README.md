# Shared mean loads

Base `399fa6d8`, C36 image and block8 summary geometry unchanged. One intended
variable is shared-memory initialization scheduling: one conditional assignment
replaces zero-initialization followed by a conditional overwrite. Both branches
produce the same BF16 operand; matrix multiply, ordered head sum, budgets,
summary storage and fine selection remain unchanged.

The generated CUDA of the preceding larger-tile probe contains a CTA barrier
inside each summary-row load iteration, between the zero store and overwrite.
That probe passes score bits but slows further, and is retained. The single-store
probe targets this synchronization without changing selected values. It is an
out-of-band substitution only; serving defaults remain unchanged until qualified.

Actual-input coarse/fine score bits, cache/graph/lifecycle checks and strict
unwritten-suffix initialization checks must pass before two fresh-process
component measurements. Full-model final-answer quality, request performance
and memory/admission gates remain required afterward. Native uses its real
68-byte/key layout. The candidate pays for all 100 bytes/key and maintenance.
Record every run, error, command, allocator observation and release. Raw captured
inputs remain node-local, outside Git. No quality or speed win is assumed.


The single-store probe passes all 12 actual batches, strict initialization with
zero sanitizer errors and six graph/cache/lifecycle cases. Generated CUDA confirms
zero barriers inside its mean-load loop (the previous code has one per iteration).
Two fresh-process timings nevertheless remain slower than native in all twelve
shapes: native/candidate approximately 0.53–0.81. Removing the barrier improves
the earlier implementation but does not resolve the performance gate. It does
not advance unchanged to model inference. All receipts are retained.

The next intended variable is page-wise contiguous copying of means, retaining
the single store, tile64, scores, budgets and representation. `copy_score.py`
passes a strided view of only the summary headers and copies each complete
physical page's contiguous means into its matrix region. Partial tile pages
are zeroed without loading an invalid table suffix. No new tensor allocation or
GPU work is added by the view. An actual-score check precedes two fresh-process
performance screens; initialization/lifecycle checks follow only if the screen
finds a useful gain. Serving requires every gate plus final-answer evaluation.
