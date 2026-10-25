# MoE overlap: routed weights evict-first in L2; mHC finalize writes the MXFP8 cast

Owner, 2026-10-25: defer fusions and tuning until after spark-ds41f-1025, and bound each
idea's gain before building it. Two bounded speed-ups from the MoE-chain ideas, prepared
for next week's release as tested TileLang changes, each with its own kernel bundle and
serving A/B. Nothing here has run on a GPU yet (written 2026-10-25 without node access).

## Ceilings (c1 decode profile, inventory 5, rank 0)

The decode MoE chain of one layer: dispatch 3.2 µs, the tokens' `per_token_cast` 1.7 µs,
FC1 227 µs, SwiGLU-cast 3.0 µs, FC2 123 µs, combine 2.6 µs, with 0.2-0.5 µs between
kernels. FC1 and FC2 are 42% of c1 kernel time.

| Change | Ceiling at c1 | Basis |
| --- | --- | --- |
| Routed expert weights evict-first in L2 | about 2% (the brief's; the traces below bound it near 0 at c1, 0.25% at c8) | The routed experts stream about 95 MB a layer a rank through the 24 MB L2. The FFN prefetch window loads the shared expert's ~6 MB just before that stream, and the shared GEMM takes 26.6 µs, about its DRAM-bound time: the prefetched lines might be evicted. |
| Activation quantization written by its producer | about 1.3% for every `per_token_cast` (about 6 a layer: 1.02% of the window plus gaps); about 0.5% for the casts mHC finalize feeds | Three of a decode layer's casts read an mHC entry's output: the attention entry's for the fused Q-A/KV projection, and the FFN entry's twice, for the shared expert's gate/up and for the routed experts (the same rows). |

The evict-first ceiling is the brief's. The traces bound it far lower (next section).

## L2 cache priorities: what the traces bound

Owner, 2026-10-25 (scope addition): L2 priorities beyond evict-first, namely evict-last on
reusable activations and a persisting-L2 set-aside, as sweep variables, not production flags.

**The set-aside.** On this stack an evict-last line is protected only inside the
persisting-L2 set-aside, and the recipe sets none (`VLLM_GLM53_L2_PREFETCH_PERSIST_MB`,
default 0, read by the prefetcher when it starts). So the windows' evict-last prefetches
(`cp.async.bulk.prefetch.L2`) are probably plain prefetches today.

**The windows in the traces.** [l2_windows.py](l2_windows.py) reads request-inventory-5's
decode traces (TP4, rank 0, ports build):
- c1 greedy: 13 steps, 27.1 ms of GPU work a step;
- eight mixed requests: 35 steps, 44.4 ms a step.

It matches each consumer of a window to the latest prefetch before it. A consumer whose
window had finished prefetching runs near L2 speed; one that started while the prefetch
was still running runs near DRAM speed. Excess a step over each consumer's L2-speed time:

| | While the window's prefetch still runs | After it finished |
| --- | ---: | ---: |
| c1 | 1.67 ms (6.2%) | 1.06 ms (3.9%) |
| c8 mixed | 0.82 ms (1.8%) | 1.02 ms (2.3%) |

- **Prefetch timing:** the "still running" column is the largest L2 loss, and no cache
  policy recovers it: the lines are not there yet. Examples:
  - WO-B at c1: p50 8.1 µs with its window finished, 32.3 µs without. The WO prefetch
    takes ~100 µs at grid 2, and at short context sparse MLA is over long before then.
  - The mHC projection and fused Q-A/KV start before the NEXT window lands.
  - The indexer's Q-B (8.4 MB) is in no window: p50 39 µs against Q-B's 17 µs.

  That is prefetch scheduling, the PDL session's (`local_8a75a575`; sent there).
- **What a policy could keep:** only the "after it finished" column.
  - Its biggest item is the shared expert's down projection (0.53 ms at c1). It takes
    19.3 µs beside FC1 and 18.1 µs before it, so the routed stream is not what slows it
    at c1. At c8 it is 16.7 µs beside FC1 against 10.9 µs before it, about 0.1 ms a step.
  - The rest (WO-A/WO-B, fused Q-A/KV, Q-B, router) is under 0.5 ms a step at c1 and
    0.7 ms at c8. That includes their own variance and the indexer's misses, which no
    window holds.

**Bounds by option (short-context decode, TP4):**

| Option | Upper bound | Why |
| --- | --- | --- |
| Evict-first on routed weights (built, `moe-evict-first`) | about 0 at c1, about 0.25% at c8 | No window consumer reads its lines after the routed stream. The shared gate/up runs at L2 speed before FC1 starts, and the NEXT window is issued after the combine. At c8 only the shared down projection, beside FC1, loses time. |
| Evict-first plus a set-aside of S | at most about 1.5% (c1 and c8), and only if those misses are evictions and S covers a window (16-20 MB) | Protects the windows' evict-last lines from everything else. GB10's 24 MB L2 leaves 24 - S for default lines. A set-aside cost GLM-5.3 its 32K prefill and c12 throughput on a 128 MB L2. |
| Evict-last on reusable activations, plus a set-aside | about 0 | At decode the MoE input is 6 rows x 5120 (30 KB of E4M3, 61 KB of BF16), written microseconds before its readers and re-read continuously. Default loads already keep it, more so once the weights are evict-first. Prefill SP's gathered rows (16K x 5120 = 84 MB) exceed the L2. Not built: below the noise floor by orders of magnitude. |

Long context may change this: the indexer streams the compressed index cache between the
WO window's prefetch and WO-A/WO-B, and at 1M that cache is tens of MB a layer. A long-
context decode trace would bound it; these traces are short-context.

**What runs next week**, after the kernel bundles:
1. The [L2 policy probe](bundles/l2-policy) (dgx2, in [kernels.json](kernels.json)). On the
   node it measures GB10's maximum set-aside and whether a buffer survives a 96 MB stream,
   for each pairing of how the buffer was loaded and how the stream reads, at set-asides
   from 0 to the maximum. It also measures the set-aside's cost to default reuse with stale
   persisting lines. It settles what the hints do on SM121.
2. The [set-aside sweep](l2-setaside-sweep.json), only if the probe shows that a set-aside
   keeps the windows' evict-last lines through a default stream that otherwise evicts them.
   - Arms: evict-first (the reference) and evict-first with set-asides of 6 MB, 12 MB and
     the maximum ([evict-first-persist-*-tp4-dev.json](.)), on the `screen` and
     `single-stream` suites.
   - The variable is the existing `VLLM_GLM53_L2_PREFETCH_PERSIST_MB`; the prefetcher's
     log line reports the size the driver applied.
3. The evict-first serving A/B is unlikely to resolve a 0-0.25% effect. Run it only
   beside the sweep (it is the sweep's reference), not on its own.

## Changes (vLLM, on the release's tilelang-ports 26382f5cf)

Each change is one commit on a branch of its own, so each arm carries one variable.
`moe-overlap` stacks both, for the candidate after both pass.

| Change | Branch, commit | Files |
| --- | --- | --- |
| Routed weights evict-first | `moe-evict-first` d9849b2d4 | `tilelang/moe.py`, its test file, `benchmarks/kernels/benchmark_deepseek_v41_moe_evict_first.py` |
| mHC finalize writes the cast | `mhc-mxfp8-cast` a5a7c5325 | `tilelang/{mhc,linear,gemm,moe}.py`, `fused_moe/tilelang.py`, `nvidia/model.py`, `attention.py`, `kernels.py`, `b12x_layers.py`, six test files, `benchmarks/kernels/benchmark_deepseek_v41_mhc_cast.py` |
| Both | `moe-overlap` d9849b2d4, 1c458d618 | the two commits above (1c458d618 is a5a7c5325 on top of d9849b2d4) |

**Evict-first.** Every weight byte of the routed experts' grouped GEMMs loads with the
evict-first L2 policy: the TMA tiles through TileLang's `eviction_policy`, and the K tails
(TP4's 576-wide down rows) and scale codes through `cp.async` with an L2 cache-policy
operand (`createpolicy.fractional.L2::evict_first`), since TileLang's own `cp.async` takes
none. The gathered activation rows, which every N tile reads again, keep the default
policy. A cache hint only: the output bits are unchanged.

**MXFP8 cast in mHC finalize.** Finalize's normalizing warps hold `y` in registers; each
now also writes `y`'s E4M3 bytes and UE8M0 scales, `per_token_cast`'s bytes (the
arithmetic of `gemm._reference_cast`, verified byte-identical to `per_token_cast` in
experiments/2026-10-13-tilelang-wo). Four adjacent lanes hold a 32-column block; two lane
shuffles give them its amax. The entry returns the cast as `linear.MXFP8Rows`, padded to
whole decode tiles, and the decoder layer hands it to the sublayer's projections for the
call. The fed projections (fused Q-A/KV, shared gate/up, routed experts) cast nothing, and
fail rather than cast if no cast or another tensor's arrives. Rows regrouped after the
entry are cast once by `per_token_cast`: a CED boundary's compact rows, and prefill SP's
gathered rows (attention with a compressor, and the FFN, which the release cast twice).
The decoder layer declares that cast; the MoE plan no longer warms its own.

## Tests and benchmarks (in the vLLM commits)

- Evict-first: the generated source loads tiles, tails and codes evict-first and the
  activations by the default policy, at TP4 and TP3 gate/up and down shapes; every
  numeric MoE test unchanged. The benchmark compares the routed chain (gate/up, SwiGLU,
  down) and "shared after routed" (the FFN window's prefetch kernel on the shared
  expert's weights, the routed chain, then the shared expert's GEMMs) against the
  release's `moe.py`, bits first, at c1/c4/c16 decode rows (4, 24, 96) for TP4 and TP3.
  Gate: neither sequence slower.
- Cast: every mHC entry test also checks the cast against `per_token_cast` of the `y` it
  wrote. Dedicated tests cover both finalize widths, decode and prefill sizes, every
  scale regime (zero blocks, below the 1e-4 floor, above 448, powers of two), a NaN
  token, and the regrouped rows' cast. A fed projection gives the self-casting
  projection's bits at 1-300 rows and fails without its cast; a split-K projection cannot
  be fed. The MoE tests hand the plan its MXFP8 rows. The CED and SP model tests' stand-ins
  return the sixth output. The benchmark compares decode entries (finalize plus one cast
  at attention, two at the FFN) and a TP4 prefill SP layer against the release's finalize
  plus the casts it replaces, bits first. Gate: no case slower.

## Lab plan

The arms are candidate 6's dev arm (image `spark-ds41f:19f2c20ed4d6-next-6`, whose
package files are tilelang-ports 26382f5cf) with each change's package files mounted
([evict-first-tp4-dev.json](evict-first-tp4-dev.json),
[mhc-cast-tp4-dev.json](mhc-cast-tp4-dev.json)). If spark-ds41f-1025 is promoted first,
its image is candidate 6's and the arms stand as they are.

1. [kernels.json](kernels.json): the L2 policy probe on dgx2, the evict-first bundle on
   dgx3 and the cast bundle on dgx4 (tests, then the gated benchmark). The two change
   bundles must exit 0. This is the first compile of either change; read each node's
   `~/spark-lab/results/moe-overlap-kernels/*/output.txt`.
2. Only after both pass (a failed kernel step does not stop a job, and a boot failure
   closes the window): [mhc-cast-ab.json](mhc-cast-ab.json), and, if the probe supports a
   set-aside, [l2-setaside-sweep.json](l2-setaside-sweep.json) (whose reference arm is
   evict-first). [evict-first-ab.json](evict-first-ab.json) measures evict-first against
   candidate 6 alone; given its bound, run it only if the sweep is skipped and a reading
   is still wanted. Each A/B boots the change and candidate 6 for the determinism gate and
   distinct streams, then runs the `screen` and `single-stream` suites on the change,
   candidate 6 and the change again.
   - **Outputs:** identical. Every reference digest equals candidate 6's
     ([tables_arms.py](tables_arms.py)), and the screen's model step reads KL 0.
   - **Speed:** the screen's engine grid (step cost at 1-96 rows) faster beyond its
     interval, with no metric slower: c16, prefill 32K/256K, single stream. The repeat
     arm bounds drift.

```sh
python3 experiments/2026-10-25-moe-overlap/make_bundles.py ~/projects/vllm-ds41-tilelang-migration
scripts/lab.py run experiments/2026-10-25-moe-overlap/kernels.json --dry-run
```

`make_bundles.py` rewrites the bundles and arms from the two branches after a fix.

## Open

- **First compile on the nodes.** The riskiest parts are the `cp.async` helper (an
  extern call with TileLang access pointers, and the sync planner seeing its writes as
  ordinary writes) and finalize's FP8 store. A compile error shows in the kernel bundle's
  output, not the verdict.
- **A possible extra barrier (review, 2026-10-25).** TileLang's sync planner tags
  `tl.ptx_cp_async` writes as asynchronous but sees the hinted copies' extern calls as
  ordinary writes. In the TP4 down GEMM's tail variant (K=576) it may therefore place a
  `__syncthreads` between the TMA tile and the tail copy of each stage. That is
  warp-uniform, so it costs speed, not correctness; the evict-first benchmark's routed
  sequence shows whether it costs anything.
- **Other models on `--moe-backend tilelang`.** `TileLangExperts` now reads only
  handed-over MXFP8 rows, so a model whose producer writes none (the plain DeepSeek-V4
  model could select the backend) fails at its first forward. The backend is documented
  as DeepSeek V4.1's.
- **NaN blocks.** Finalize's amax drops a NaN (`fmaxf`); whether `per_token_cast`'s BF16
  reduction does too is unverified. They agree on a whole NaN row either way (the 1e-4
  floor's `fmaxf` absorbs a NaN amax), and a whole row is the only NaN finalize writes (an
  RMSNorm of a NaN token); the NaN-token test checks it. A block mixing NaN and finite
  values could differ, but no finalize output has one.
- **TP3.** The benchmarks include TP3's shapes; the serving A/Bs are TP4 only. Screen the
  triangle (`screen-tp3`) before a release that carries them.
- **Next producers, same pattern.** Attention's Q-B and the indexer's Q-B both cast
  `q_norm`'s output (the same rows twice), and the shared expert's down projection casts
  its SwiGLU's output: the rest of the ~1.3%.

## Results

None yet.
