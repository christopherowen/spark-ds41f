# TileLang sampled/rejected-token counts

Port `input_batch._get_num_sampled_and_rejected_kernel`, which runs after both
ordinary sampling and speculative verification. The profiled DS41F candidate
launches it every decode step: 104 calls across four ranks in the one-stream
profile and 436 in the sixteen-stream profile. This is a migration of a small
bookkeeping kernel; its total cost is too small to imply a material serving gain.

The contract is exact integer output, including in-place sampled counts, request
state remapping, incomplete prefill, variable draft depths and dead-row cuts.
Keep the original Triton kernel as a test reference. Remove its runtime dispatcher
and warmup; the runner's owner declares both dead-row variants and both request
index types through the common TileLang preparation registry.

GPU tests exercise the public API at DS41F's one- and sixteen-request limits,
five drafts plus one target token, context lengths through 1,048,576, both index
types, partial blocks, full-capacity persistent buffers, empty batches, and graph
replay after input changes. An independent torch oracle accompanies the exact
Triton comparison. No distribution, draft-budget or verification policy changes.

Acceptance requires all correctness cases, zero compute-sanitizer memory errors,
and no paired statistically significant slowdown at any measured shape. Benchmark
on identical warmed inputs, alternate order, use eight fresh allocations per
case, and retain every candidate and native receipt under `runs/`.

## Results

All **32 focused GPU tests pass**, including the actual public API, both index
types and both dead-row variants. All 32 also pass under CUDA 13.0 memcheck with
**zero errors**. The final paired run passes all ten performance cases. Timings
below are microseconds per call; the production recipe admits up to 16 requests.

| Requests | Dead rows | Triton | TileLang | Time change |
| ---: | :---: | ---: | ---: | ---: |
| 1 | off | 1.057 | 0.948 | -10.2% |
| 8 | off | 1.059 | 0.956 | -9.8% |
| 16 | off | 1.058 | 0.953 | -9.7% |
| 1 | on | 1.053 | 0.953 | -9.4% |
| 8 | on | 1.064 | 0.955 | -10.1% |
| 16 | on | 1.055 | 0.956 | -9.7% |

The 128- and 257-request stress cases pass too. [results.json](results.json) keeps
every timing sample and paired interval, along with the image and source-file
fingerprints. This is a kernel improvement of about 0.1 microseconds; no overall
generation-throughput gain is claimed.

## What was rejected and what changed

Every intermediate layout passes the same 32 correctness cases:

- `runs/screen`: one lane per request in a shared block, with generated bounds
  checks, is 7–15% slower at serving sizes; rejected.
- `runs/explicit-bounds`: remove redundant generated bounds checks, retaining the
  explicit row guard and valid request-state indices required by the caller.
  Dead-row performance improves, but other cases still lose about 8%; rejected.
- `runs/request-block`: restore one block per request. It reduces the remaining
  slowdown to about 4%, but is still rejected.
- `runs/branchless`: replace prefill-dependent branches with an integer 0/1 mask.
  This lets independent loads overlap instead of delaying reads of cumulative
  logit offsets until after the prefill test. All four thread-count screens pass.
- `runs/qualified`: fix 128 threads, repeat with a fresh seed and allocation
  placements, run memcheck, and require the unchanged no-slowdown gate. Accepted
  for integration and further release qualification.

Each run includes its exact bundle and native receipts. The final runnable
[bundle](bundle/) uses the same commands as the qualification snapshot; only its
description removes an erroneous mention of sink tests. The 19 other tests in the
InputBatch file were deselected in this focused run, not reported as passes.

All 16 offline variants (live/dead, int32/int64 indices, 32/64/128/256 threads)
lower and compile for SM121a. Ruff and whitespace checks pass. The complete
223-patch [series](vllm/series) replays from pinned upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354` into a fresh tree identical to the tested
source: `f2a569e0b3f5046a3f6769d7ad7ae708200bfe57`, replay head
`79b32f7924418c18eb59900aa67d4f98d4221e42`.

The [source metadata](source.json), [source lock](upstreams.lock.json) and
[launch-disabled candidate profile](candidate-tp4.json) describe that source.
Default and candidate configuration checks pass. [Release-test classification](release-tests.json)
reports 128 selected files, 71 gates and zero problems; classification is not a
claim that the full release suite has run. The new cases extend the existing
InputBatch file, so no new test module is needed.

The lab window ran from 12:40:24 to 12:50:25 UTC on 24 October 2026. It restored
the promoted production profile, passed `doctor --live` and removed its hold.
No release image was built, and the candidate remains unpromoted pending full
release and serving validation.

The concurrent runner-input port changes the warmup API and conflicts with the
standalone patch's context. An [integration adjustment](integration/README.md)
is prepared against its pinned commit, preserving identical kernel code and
count semantics. Its combined tree still needs GPU and full release validation.

[Acceptance opportunities](acceptance-review.md) records the separately requested
review. They are hypotheses or earlier measured evidence, not changes in this port.
