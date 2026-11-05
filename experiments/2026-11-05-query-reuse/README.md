# Decode each query once

Base `ed811312`, C36 image, block8 means, candidate budgets and native fine
arithmetic unchanged. One intended variable: decode the native packed query to
BF16 once per row instead of once per coarse scoring CTA. The preceding page-copy
screen preserves all actual score bits and wins larger row-count components but
still loses single-query shapes. It is the component comparator, not a final
answer or request-speed result.

The decoded-query workspace is explicitly admitted: 8192 bytes per query row,
bounded at 256 rows (2 MiB). Each chunk writes it before scoring. Covered domains
skip its writes and reads. Source and dense kernels use the exact same decoding
and BF16 values as before. Query, summary and scoring tensors stay on the GPU.
There are no reference calculations, output comparisons or grading in the runtime.

First check all 12 actual input batches against independent coarse/fine score
bits, ordered candidates and native covered-domain outputs. Then screen two
fresh performance processes, with Native68 and original physical page IDs.
If promising, graph/cache/lifecycle and strict uninitialized-suffix checks are
required before model startup. All commands, failures, allocator observations
and releases remain recorded; raw inputs stay node-local. Full-model no-loss
answer gates and request timing remain mandatory. No speed or quality win is
assumed from removing duplicate calculations.
