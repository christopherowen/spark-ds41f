# Remaining V2 sampling kernels

Source base: vLLM `47176036f` on `tilelang-ports`, including the combined runner,
sampling, mHC, embedding, confidence, rank and resampled-token ports. Deployment
base: `0e70ba56`. The promoted baseline remains
`2026-10-22-spark-ds41f-1022`; this experiment changes no runtime configuration.

## Inventory boundary

`inventory.json` lists 31 remaining kernel entry points in the V2 sampler and its
verification paths, after excluding the existing TileLang replacements and
Triton helper functions that do not launch. This is a static inventory of
supported/optional features, not a claim that 31 kernels run on every token.
The default sampled speculative path, per-request controls, and optional
diagnostics/distributed sampler features have separate groups. An unexercised
request feature needs a representative trace before its performance matters.

The inventory does not count PyTorch/ATen launches, collectives, checkpoint
loading, the vision tower, or other model runners. Those require separate
source/trace inventories; counting every Triton decorator in vLLM would include
unrelated models and retired reference kernels.

## Qualification rule

Each entry remains on its existing implementation until all of these hold:

- The old and new public paths agree, including seeds, draft prefixes, request
  slot reuse, exact vocabulary bounds, strides and supported exceptional inputs.
- The owning object declares every specialization before the cache freezes;
  request batch sizes do not produce new specialization keys.
- Focused compute-sanitizer memcheck passes.
- Paired CUDA graph timings at DS41F's 129280-token vocabulary and serving row
  counts have no demonstrated slowdown. Raw observations and paired 95%
  intervals are retained; an interval spanning zero is inconclusive.
- Stateful/in-place benchmarks start each graph node from its own initialized
  state. Candidate/reference use the same addresses and alternate timing order.
- A combined serving candidate passes startup, greedy output equality and
  seeded sampled output equality, followed by generation throughput and draft
  acceptance checks. Kernel timings alone do not establish a serving gain.

## Prepared candidates

Ten entry points now have TileLang candidates. Their public owners still select
the original implementation. The final source is
`8decb3e8d5438c9c7126f7e0be03f43b61a0dfcc`; `qualified/source.json` and the input
hashes in `qualified/run/verdict.json` identify the bytes tested on dgx4.

| Entry points | Contract covered | Timing status |
| --- | --- | --- |
| penalties and histogram accumulation (2) | Repetition/frequency/presence order; preceding draft tokens only; packed prompt bits and output multiplicities | Measured screens pass |
| min-p (1) | Exact retained logits, zero bypass, reference NaN reduction and strict threshold | Slower at 96 rows; unqualified |
| logit bias (1) | Allowed list, sparse bias, minimum length and structured-output stop restoration | Pending |
| forbidden sequences (1) | Committed/draft boundary, request mapping and exact suffix match | Pending |
| thinking marker cache and budget enforcement (2) | Cold scan, incremental update, rollback, slot reuse and multi-token markers | Pending |
| prompt token IDs and custom logprob IDs (2) | Chunk offsets, dynamic output width and zero-top-k unused-pointer alias | Pending |
| NaN metric (1) | FP32/BF16 input, explicit NaN predicates and output layout | Pending |

Histogram accumulation preserves the original kernel's ownership boundary:
the caller zeros the selected state rows, then the candidate accumulates into
them. Tests cover zero and nonzero initial buffers; the candidate is not a
replacement for the public owner's complete initialize-and-accumulate sequence.

The remaining 21 entry points have work packets in `work-packets.json`, covering
stochastic verification, vocabulary-parallel drafting, log-softmax and support
compaction, top-k/top-p, and optional tracing, sharding and watermark features.
These packets define contracts and relevant existing tests; they are not
completed translations.

Tests are in `tests/v1/worker/test_gpu_sampling_controls.py` and the existing
thinking-budget and logit-bias suites parameterized over both implementations.
The benchmark is `benchmarks/kernels/benchmark_sampling_controls.py`.

## Final measured screen

`qualified/run/output.txt` retains the native output:

- 141 tests passed; 50 focused cases passed under compute-sanitizer memcheck,
  with zero reported errors. These exercise candidates directly; they do not
  qualify a serving integration.
- All 18 paired timing cases first compare initialized candidate/reference
  outputs. Eight distinct initialized buffers are used for graph nodes, with
  resets outside timing, alternating measurement order and paired 95% intervals.
- One performance gate fails: min-p at 96 rows. The bundle exits 1 for that
  slowdown despite passing correctness and memory checks. The generic lab
  verdict's last-line summary says zero memcheck errors; it must not be read as
  an overall pass.

Representative medians in microseconds (TileLang / existing implementation):

| Kernel | 1 row | 16 rows | 96 rows |
| --- | --- | --- | --- |
| penalties, vocabulary 129280 | 4.188 / 5.852 | 74.448 / 80.000 | 486.530 / 511.488 |
| min-p, vocabulary 129280 | 65.132 / 66.276 | 110.330 / 113.730 | 568.388 / 538.732 |

All three penalties intervals favor the candidate (median reductions about
28%, 7% and 5%). Min-p's 16-row interval spans zero; its 96-row interval is
4.343 to 40.083 microseconds slower. Keep min-p on the original path.

The 12 histogram cases cover one and 16 rows, lengths 128 / 8192 / 131072,
and uniform or repeated-token inputs. None demonstrates a slowdown; the
one-row repeated-token case at length 8192 is inconclusive. Other histogram
intervals favor the candidate. Full timings and observations are in
`results.json`; these are kernel measurements, not serving throughput claims.

## Retained screening history

Every snapshot preserves its source, command bundle and available native output.
Run bundles live under `~/spark-lab/prepared`, outside the clean deployment
checkout. The lab specifications name their exact location.

| Snapshot | Observation and follow-up |
| --- | --- |
| `screen` | Initial three candidates; deployment checkout guard prevented execution |
| `screen2` | 80 pass, 25 fail; boolean vector compilation and NaN diagnostic layout |
| `screen3` | 116 pass, 9 fail; remaining NaN diagnostic layout |
| `screen4` | 114 pass, 11 fail; strengthened NaN cases exposed reduction semantics and BF16 predicate compilation |
| `screen5` | 125 pass and 40 memcheck cases with zero errors; timing precheck found a one-ULP min-p threshold mismatch at 96 rows, seed 48294 |
| `screen6` | Isolated timing/diagnostic families; wider penalties blocks were slower; histogram cases passed; exact libdevice log fixed the min-p boundary |
| `screen7` | Prepared snapshot, not executed; superseded by explicit histogram accumulation ownership and eight additional tests |
| `qualified` | Final 141 / 50 tests and all 18 timings; one min-p speed gate fails |

The threshold fix uses the same libdevice log as the existing implementation;
an approximate log had incorrectly removed one vocabulary entry. Penalties use
2048-column blocks after the wider variant lost its speed comparison. No test
tolerance was relaxed to accept either failure.

The final window ran from 17:29:17 UTC on 2026-10-24. The promoted four-node
service was restored and `doctor --live` passed at 17:32:09 UTC; the hold was
removed. `qualified/window.txt` records the coordinated stop and restoration.

## Replay and outstanding integration

`prepared-source/vllm/series` is a complete 274-patch source recipe from pinned
upstream `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`, including the earlier combined
ports on which this work depends. A fresh application of the entire series
produced exactly the tested source tree
`561b6faa709ec1274de1726e4e17a4a4436e76ef`.
`prepared-source/identity.json` and `replay.txt` retain the proof. This recipe
does not select a deployment image or enable the new controls.

Next steps are timing the other seven candidates, improving or retaining min-p,
and wiring each public owner with complete startup declarations. A combined
image must then pass frozen-cache startup, full-model greedy and seeded-sampling
equality, throughput and draft-acceptance checks. None of those serving gates is
claimed here; the two favorable kernel screens are preparation for them.
