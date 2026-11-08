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


The page-copy actual check passes all 12 captured batches. Both fresh timing
processes show dense64 native/candidate ratios 1.165–1.178 and source8/64
1.187–1.283, while dense8 is approximately neutral. Single-query shapes still
lose: dense0.66–0.69, source0.87–0.88. All samples and t intervals are in
`runs/copy-01/comparison.json`. Allocator peaks are approximately 349 MB allocated
and 373 MB reserved. Processes start at 52/54 C, end at 54/55 C, and report no
active hardware thermal slowdown at the end. This is a useful component signal
but not an all-shape, request-speed or answer-quality pass. The lease releases
at 15:03:28 UTC with no containers left; a later independent reader finds no
lease, request, container or GPU compute process.

The next exact-arithmetic experiment decodes each packed query once per row
into an explicitly admitted bounded workspace, then reuses it across all
coarse scoring CTAs. Summary representation and selected-score arithmetic remain
unchanged. This removes repeated computation rather than reducing quality or
excluding work from timers. No model startup occurs until safety and speed gates.
