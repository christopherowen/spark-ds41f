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
