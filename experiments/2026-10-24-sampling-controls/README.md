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

## First independent group

The first source commit prepares three candidates without changing dispatch:

| Kernel | Variable | Behavior to preserve |
| --- | --- | --- |
| `min_p` | Per-lane maxima accumulated before one block reduction | Exact retained logits, zero-min-p bypass, NaN propagation and strict threshold comparison |
| `penalties` | TileLang implementation over the existing logits/count layouts | Repetition order, frequency/presence arithmetic, only preceding draft tokens counted |
| `bincount` | TileLang prompt/output atomics | Packed prompt bits, output multiplicities, selected state slots only |

Tests: `tests/v1/worker/test_gpu_sampling_controls.py`. Benchmarks:
`benchmarks/kernels/benchmark_sampling_controls.py`. Nearby existing sampler
flag, bias, bad-word and reasoning tests remain integration checks when their
public call sites change.

The `screen/` snapshot records exact source identity, input files and the lab
spec. Bundles copied for the lab live under `~/spark-lab/prepared`, outside the
clean deployment checkout. Shared-cluster holds apply to every GPU run.

Results are pending. No speed or correctness claim is made for these prototypes.
