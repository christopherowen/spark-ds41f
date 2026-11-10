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
host copies or grading. The component diagnostic control reserves the same 100 bytes/key. The final
model comparison uses `native.json`, retaining the actual native 68-byte/key
layout, against the candidate 100-byte/key layout. Its cache and admission costs
therefore remain in the acceptance comparison. Mechanical, graph/lifecycle, initialization, actual replay and
component-speed gates are required before any all-node model startup.

`check_quality.py` in the answer-quality experiment enforces the final-answer
no-loss gate. Its 16 tests cover new wrong answers, truncation, unstable boots,
missing tasks, changed budgets, empty workloads and reasoning divergence. An
audit of existing receipts accepts the dense-only/native-source setting's
observed answer quality and rejects the source1024 setting's lost completion.
That acceptance is quality-only: the former setting still fails the speed goal.


Runtime gates use published source `66e012581beba809d6c5757fe622257efd42d5dd`.
The mechanics job passes all six cases: both page sizes, both indexer interfaces,
mixed owners, native-byte preservation, mean bits, graph replay, page reuse,
257-row chunking and a source plan of 1M. It takes 257.04 seconds including cold
compilation, which is not a performance measurement. Allocator peaks are
777,711,616 allocated and 1,044,381,696 reserved bytes. A foreign request arrives
during that job; the controller releases before the next job at 14:09:05 UTC.
The retained error receipt records this orderly yield, not a failed kernel.
The next window resumes actual replay and strict initialization checks without
repeating the completed mechanics gate.

`benchmark_runtime.py` is a separate performance-only process. It uses the four
already-diagnostic validation captures with original physical page IDs and
visible lengths, at 1, 8 and 64 captured queries. Native retains its 68-byte/key
layout; block8 uses 100. Native-before / candidate / native-after graph timing
brackets each case. Page-completion construction is measured separately, for
one page and all captured complete pages. Neither reference scoring nor answer
grading enters these timers. Two fresh processes remain required. These are
component timings on captured prefill queries, not decode-query or request
speed measurements, nor a fresh independent quality benchmark.

Full-model answer phases additionally collect server draft acceptance and
prefill/decode counters outside every request timer. Their purpose is to
separate selector cost from changes in speculative progress; no cause is
assumed before measurement. The first observed new wrong/incomplete answer
against a successful matched native answer ends the candidate cohort and
restores the service. Such a rejected partial cohort is not a two-boot speed
qualification.


The resumed window (`runs/runtime-gates-02`) passes all 12 actual batches,
including every mean, coarse score, retained fine score, candidate ordering and
covered-domain native-output check. It takes 47.12 seconds. Strict initialization
passes both interfaces and both page sizes, including empty/partial/pruned rows
and invalid table suffixes, in 124.46 seconds with zero sanitizer errors.
Actual-replay allocator peaks are 278,402,560 allocated and 285,212,672 reserved
bytes. Strict allocator counters report zero because caching is disabled; these
are not evidence of zero GPU memory use. The lease releases at 14:31:57 UTC with
no remaining containers, and the next reader observes the foreign holder's new
lease rather than any retained LSA reservation. These checks qualify mechanics,
not final-answer quality or speed.


Both component timing processes pass mechanically but **fail the speed gate**.
They take 11.95 and 11.61 seconds, starting at 47/51 C and ending at 51/52 C,
with no active hardware thermal slowdown at the end. All twelve measured shapes
are slower: native/candidate ratios are 0.50–0.71, or roughly 1.4–2.0x candidate
latency. Exact event samples, both processes, memory and small-sample t intervals
remain in `runs/component-01/comparison.json`. Allocator peaks are approximately
349 MB allocated and 373 MB reserved, not full-model memory. Page-completion
maintenance is measured separately and does not explain these query-only losses.
This representation will not advance unchanged to serving. The next diagnostic
profiles the stages after complete-plan timing; it changes no runtime arithmetic
or selection. Released at 14:34:32 UTC, independently observed under the next
foreign holder at 14:35.


A separate exact-arithmetic tuning probe changes only the coarse matrix tile
from 64 to 128 summaries per CTA. It tests whether reusing each decoded query
across twice as many summaries reduces cost. The runtime default is unchanged.
`coarse_tile_probe.py` substitutes the existing parameter only in the out-of-band
check/timing program. All 12 actual batches must retain the independent visible
coarse BF16 score bits, fine-score bits and ordering before two fresh timing
processes. Any failed bit check rejects this optimization. This does not change
summary representation, budgets, fine-score arithmetic or answer protocol.


The stage diagnostic (`runs/stages-01`) identifies the mean scorer as the main
cost: approximately 70–85% at 8/64 rows; assembly is a small share. The tile128
probe passes coarse bits on all 12 captured batches but is slower still in its
first timing process (native/candidate 0.31–0.46). It yields at 14:43:53 UTC on a
foreign request; a second process would not rescue this clear screen failure.
The single unfavorable process and orderly-yield error remain in
`runs/tile128-01`. Its generic controller labels the check verdict as performance
measured; that is a controller metadata error. The check's own result correctly
says `performance_measured: false`; its elapsed time is not a performance sample.
The generated CUDA exposes a CTA barrier inside every summary-row load iteration.
The next experiment changes only that double-write initialization pattern.
