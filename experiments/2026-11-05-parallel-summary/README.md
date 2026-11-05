# Cooperative summary construction

Base `3ad74ef5`. One intended variable: parallelize dimensions while constructing
the same eight-key means and MXFP4 encoding. Preserve sequential accumulation of
the eight keys, BF16 rounding, native scale floor/rounding and E2M1 encoding.
Coalesced loads and cooperative maximum reduction replace serial per-thread
32-dimension loops. No summary shape, score, candidate or quality policy changes.

This is out-of-band. Require full cache-byte equality on all four private frozen
captures, both page sizes, wide public synthetic quantized scales, zero/tie/sign
vectors, skipped/outside slots and recycled pages. Require zero initialization
and shared-memory race errors. If these pass, record two fresh maintenance timing
processes, including single-page, all-page and skipped-slot costs and memory.
The published preceding summary timings are comparison evidence; do not rewrite
them. Keep every error. No request or answer-quality claim follows. Any runtime
integration requires the full package/kernel/lifecycle and admission gates.
