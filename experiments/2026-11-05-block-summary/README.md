# Finer summary selection

Base `f28c8db65d0aabfbe13a1accbd495d1bd250383a`, C36 vLLM source
`6d4e49d1cdbb6513819e47145e0e52fdab327b6b`, image
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.

One intended variable is summary granularity: a mean for each native 8-key
block, replacing balanced 64-key groups. Both arms retain the same number of
native fine-scored blocks, partial-page policy, forced final block and native
score/selection arithmetic. Forced-final-block membership can differ by one
8-key block. The new representation removes iterative clustering but requires
32 bytes/key of BF16 means: 100 bytes/key including the native68-byte records.
Those are storage calculations, not measured model memory or speed results.

This is an untimed, out-of-band screen on the already replayed rank-0 captures.
It checks that the generalized scorer reproduces the original visible coarse
BF16 bits, and that every retained fine-score bit equals native full scoring.
The actual captures have three independent documents and adjacent prefill
queries; the reused diagnostic document is not a fresh held-out benchmark.
Raw inputs stay private on the node. Every result, error and memory observation
is retained. No model-generation quality or runtime performance is established
by selection recall.

Quality is a hard gate for subsequent serving work. Benchmark keys and executable
tests, rather than native wording or thinking, judge final answers. Any new
incorrect or incomplete answer where the matched native control succeeds rejects
a candidate. Frozen independent quality sets, temperature0, identical generation
allowances, two fresh candidate boots, and full request timing remain required.
All thinking and unsuccessful attempts count toward elapsed time. No diagnostic
reference scoring, host copies or grading may enter timed serving.
