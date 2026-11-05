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

The first screen (`runs/screen-01`, runtime `c555e5527dc2ac74667bb14de9f980650dc76e32`)
passes every original coarse-score bit check and every retained native fine-score
bit check on all 12 captured batches. On the reused diagnostic document:

| Interface | Cluster64 | Block8 | Difference |
|---|---:|---:|---:|
| Dense layer2, top512 keys | 62.06% | 77.67% | +15.62 points |
| Dense layer8, top512 keys | 71.58% | 86.76% | +15.19 points |
| Dense layer14, top512 keys | 79.01% | 89.15% | +10.14 points |
| Source layer20, top512 keys | 86.17% | 93.49% | +7.32 points |
| Source layer20, handoff blocks | 66.77% | 81.05% | +14.28 points |

All per-query and fitting-document observations are retained. These remain
selection proxies from three documents, including one reused diagnostic, and
are no evidence of final-answer quality or request speed. Torch allocator peaks
are 500,192,768 allocated bytes and 853,540,864 reserved bytes; full-model memory
is not measured. The lease is released at13:38:15 UTC with no containers left.
Independent verification confirms the lease is absent and the checkout clean.

The separate runtime preparation writes means only after a physical page is
complete, leaves partial pages directly scored, and uses the native fine scorer
with ordered block candidates. It performs no clustering, reference scoring,
host copies or grading. Both native and candidate profiles reserve the same
100 bytes/key. Mechanical, graph/lifecycle, initialization, actual replay and
component-speed gates are required before any all-node model startup.

`check_quality.py` in the answer-quality experiment enforces the final-answer
no-loss gate. Its 16 tests cover new wrong answers, truncation, unstable boots,
missing tasks, changed budgets, empty workloads and reasoning divergence. An
audit of existing receipts accepts the dense-only/native-source setting's
observed answer quality and rejects the source1024 setting's lost completion.
That acceptance is quality-only: the former setting still fails the speed goal.
