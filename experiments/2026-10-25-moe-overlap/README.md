# MoE overlap: mHC finalize writes the MXFP8 cast (routed evict-first dropped)

Owner, 2026-10-25: defer fusions and tuning until after spark-ds41f-1025, and bound each
idea's gain before building it. Two bounded speed-ups from the MoE-chain ideas were
prepared for next week's release as TileLang changes, and the owner's L2 cache-priority
options were bounded beside them. The kernel bundles ran on dgx4 the same evening; the
serving A/Bs are queued specs.

**Status (2026-10-25, 21:15 CET):**
- **mHC finalize writes its output's MXFP8 cast:** passes on GB10. 257 tests passed, and
  the gated benchmark is faster in every case: 4.3 µs a decode layer at c1, 6.2 µs at c16,
  and 0.88 ms a TP4 prefill SP layer. Its serving A/Bs on the ring and the triangle are
  ready, not run.
- **Routed expert weights evict-first in L2:** fails on GB10: the TP4 down GEMM's hinted
  8-byte K-tail `cp.async` faults with "Illegal Instruction Parameter" (root cause under
  Results). Its ceiling is about 0 at c1 and 0.25% at c8. Dropped.
- **L2 priorities:** on GB10 evict-last lines already survive at set-aside 0, and a
  set-aside only takes L2 from default lines. No set-aside, evict-first or evict-last arm
  has a ceiling worth a window. The L2 loss the traces show is prefetch timing, which
  belongs to the PDL session.

## Results (dgx4, 2026-10-25)

Every bundle ran on dgx4 through `scripts/lab.py kernel-local` from the agent checkout
`~/spark-lab/agents/moe-overlap/spark-ds41f`, in image `spark-ds41f:19f2c20ed4d6-next-6`
(sha256:82b071b57485, candidate 6, which is spark-ds41f-1025):

```sh
python3 scripts/lab.py kernel-local experiments/2026-10-25-moe-overlap/bundles/<bundle> \
  --out ~/spark-lab/agents/moe-overlap/results/<bundle>
```

Output: `~/spark-lab/agents/moe-overlap/results/<bundle>/{output.txt,verdict.json}` on
dgx4. The branch was rebased onto main after the first five runs, so each commit is given
as run and as rebased.

| Bundle | Ran (CET) | Commit (rebased) | Verdict | Result |
| --- | --- | --- | --- | --- |
| [mhc-cast](bundles/mhc-cast) | 19:01-19:04 | 3e39c2f8 (5a102cd0) | **pass** (exit 0) | 257 tests passed in 168 s. The benchmark's gate passed, faster in every case (below). |
| [evict-first](bundles/evict-first) | 19:58-19:59 | 3e39c2f8 (5a102cd0) | **fail** (exit 1) | 57 failed, 4 passed: `CUDA_ERROR_ILLEGAL_INSTRUCTION` at the first grouped GEMM, which poisons the context for every later test. The benchmark exits 1 for the same reason. |
| [l2-policy](bundles/l2-policy) | 20:07 | 3e39c2f8 (5a102cd0) | **pass** (status) | What the hints and a set-aside do on GB10 (L2 section). |
| [evict-first-diag](bundles/evict-first-diag) | 20:30-20:31 | f62f09b7 (1f3dd138) | diagnostic | The release's GEMM and the TMA tiles' hint alone run, with equal bits. The `cp.async` helper alone, and both together, raise the illegal instruction (at the down shape, evict-first-cause showed). |
| [cp-async-diag](bundles/cp-async-diag) | 20:37 | c6b13129 (befd4733) | diagnostic | Every hinted `cp.async` form (16 and 8 bytes, `createpolicy` evict_first or evict_normal, TileLang's constant word, an explicit source size) runs alone in a kernel with the right bytes. GB10's `createpolicy` words: evict_normal 0x16f0000000000000, evict_first 0x12f0000000000000, evict_last 0x14f0000000000000. |
| [evict-first-cause](bundles/evict-first-cause) | 20:43-20:44 | f0a466cc | diagnostic | At the gate/up shape (512x512, no K tail) the change's helper, the helper without a hint and the helper with the policy as a C operand all run, with the release's bits. Output in `results/evict-first-cause-512`. |
| [evict-first-cause](bundles/evict-first-cause) | 20:46-20:49 | 4dfb4def | diagnostic | At TP4's down shape (384x576, a 64-wide K tail) the GEMM faults whenever the tail's 8-byte copy carries the hint, and runs with the release's bits without it, or with the hint on the scale codes only. |
| [evict-first-sass](bundles/evict-first-sass) | 20:51-20:52 | 904741dc | diagnostic, no result | It compiled all four kernels, but its disassembly step printed nothing. dcee615c reads the cached cubins and reports the disassembler's errors. |
| [evict-first-sass](bundles/evict-first-sass) | queued 20:52 | dcee615c | pending | Queued on dgx4's GPU lock behind other agents' kernel runs (still waiting at 21:13). Compile only; it adds the SASS, not the verdict. |

**The cast benchmark** (`benchmark_deepseek_v41_mhc_cast.py --gate`, bits compared
first). Finalize writing the cast, against the release's finalize plus the
`per_token_cast` calls it replaces (µs, with the difference's interval):

| Case | Rows | Release | Finalize + cast | Difference |
| --- | ---: | ---: | ---: | ---: |
| decode c1, attention entry | 4 | 6.06 | 4.63 | -1.42 (-1.48, -1.36) |
| decode c1, FFN entry | 4 | 7.51 | 4.63 | -2.83 (-2.97, -2.70) |
| decode c4, attention entry | 24 | 6.22 | 4.67 | -1.54 (-1.65, -1.43) |
| decode c4, FFN entry | 24 | 7.83 | 4.71 | -3.31 (-3.97, -2.66) |
| decode c16, attention entry | 96 | 7.00 | 5.10 | -2.10 (-2.61, -1.59) |
| decode c16, FFN entry | 96 | 9.14 | 5.03 | -4.12 (-4.25, -3.98) |
| TP4 prefill SP layer | 4096 | 4156 | 3278 | -877 (-888, -866) |

**Why evict-first is an illegal instruction.** GB10 rejects the hinted copy of the K tail
in TP4's down GEMM.
- **The fault.** Every fault (19:58, 20:30 and 20:48 CET) is the driver's Xid 13, "Graphics
  SM Warp Exception: Illegal Instruction Parameter", at one PC. The instruction decodes,
  but the SM rejects an operand at run time.
- **Which copy** (evict-first-cause):

  | Variant | 512x512 (no tail) | 384x576 (64-wide tail) |
  | --- | --- | --- |
  | release | runs | runs |
  | helper on the tail, codes and words (the change without the TMA hint) | runs, equal bits | **faults** |
  | the same, policy as a C operand | runs, equal bits | **faults** |
  | the same extern call without a hint | runs, equal bits | runs, equal bits |
  | helper on the tail only | runs, equal bits | **faults** |
  | helper on the scale codes only | runs, equal bits | runs, equal bits |

  So it is the tail's 8-byte `cp.async.ca.shared.global.L2::cache_hint`. The 16-byte
  `cp.async.cg` with the same hint runs, and so does the TMA tiles' hint
  (evict-first-diag).
- **Not the instruction itself.** The same 8-byte hinted `cp.async.ca` runs in a kernel
  of its own (cp-async-diag), so what GB10 rejects is the code ptxas generates for it in
  this kernel. [evict-first-sass](bundles/evict-first-sass) disassembles the
  three down GEMMs and the standalone kernel to show that code (Results).
- **Not the sync.** The tail's extern call makes TileLang's sync planner add a
  `__syncthreads` before the copies (the review's note). The passing no-hint variant has
  it too, so it is not the cause.
- **A fix and why not.** Loading the tail without a hint would keep the tiles and codes
  evict-first. It is not worth it: the ceiling is about 0 at c1 and 0.25% at c8, and on
  GB10 an evict-first stream lowers the prefetched windows' survival (L2 section).

## Ceilings (c1 decode profile, inventory 5, rank 0)

The decode MoE chain of one layer: dispatch 3.2 µs, the tokens' `per_token_cast` 1.7 µs,
FC1 227 µs, SwiGLU-cast 3.0 µs, FC2 123 µs, combine 2.6 µs, with 0.2-0.5 µs between
kernels. FC1 and FC2 are 42% of c1 kernel time.

| Change | Ceiling at c1 | Basis |
| --- | --- | --- |
| Routed expert weights evict-first in L2 (dropped) | the brief's about 2%; the traces bound it near 0 at c1 and 0.25% at c8 | The routed experts stream about 95 MB a layer a rank through the 24 MB L2. The FFN prefetch window loads the shared expert's ~6 MB just before that stream, and the shared GEMM takes 26.6 µs, about its DRAM-bound time, so the prefetched lines might have been evicted. |
| Activation quantization written by its producer | about 1.3% for every `per_token_cast` (about 6 a layer: 1.02% of the window plus gaps); about 0.5% for the casts mHC finalize feeds | Three of a decode layer's casts read an mHC entry's output: the attention entry's for the fused Q-A/KV projection, and the FFN entry's twice, for the shared expert's gate/up and for the routed experts (the same rows). |

## L2 cache priorities

Owner, 2026-10-25 (scope addition): L2 priorities beyond evict-first, namely evict-last on
reusable activations and a persisting-L2 set-aside, as sweep variables, not production
flags.

### What the traces bound

[l2_windows.py](l2_windows.py) reads request-inventory-5's decode traces (TP4, rank 0,
ports build):
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

  That is prefetch scheduling, which the PDL session (`local_8a75a575`) owns; the findings
  were sent there.
- **What a policy could keep:** only the "after it finished" column.
  - Its biggest item is the shared expert's down projection (0.53 ms at c1). It takes
    19.3 µs beside FC1 and 18.1 µs before it, so the routed stream is not what slows it
    at c1. At c8 it is 16.7 µs beside FC1 against 10.9 µs before it, about 0.1 ms a step.
  - The rest (WO-A/WO-B, fused Q-A/KV, Q-B, router) is under 0.5 ms a step at c1 and
    0.7 ms at c8. That includes their own variance and the indexer's misses, which no
    window holds.

### What the hints do on GB10 (the l2-policy probe)

The probe ([l2_policy_probe.py](bundles/l2-policy/l2_policy_probe.py)) loads a buffer P,
streams 96 MB through the L2, then times re-reading P. Survival is how much of P still
hits (1: all of it at L2 speed, 0: none). GB10's L2 is 24 MB and its maximum
persisting set-aside 18 MB.

At set-aside 0 (the recipe's), after a 96 MB stream:

| P loaded by | P 8 MB, default stream | P 8 MB, evict-first stream | P 16 MB, default stream | P 16 MB, evict-first stream |
| --- | ---: | ---: | ---: | ---: |
| an ordinary read | 0.02 | 0.00 | 0.01 | 0.00 |
| a read with evict_last | 1.00 | 1.00 | 0.99 | 1.00 |
| the windows' prefetch kernel | 0.91 | 0.81 | 0.96 | 0.88 |

The 4 MB rows at set-aside 0 were measured first, and their hot and cold reads took the
same time (38.5 and 41.0 µs), so they say nothing.

- **Evict-last needs no set-aside on GB10.** Lines loaded evict-last survive the stream
  at set-aside 0, and the windows' prefetches survive nearly as well. The earlier
  assumption here, that an evict-last line is protected only inside the set-aside, was
  wrong for this GPU.
- **An evict-first stream protects nothing that matters.** It does not save ordinary
  8-16 MB buffers (0.00), and it lowers the prefetched windows' survival slightly (0.91
  to 0.81 at 8 MB, 0.96 to 0.88 at 16 MB).
- **A set-aside only costs.** It adds no survival to evict-last lines, which already
  survive, and it takes L2 from default lines. A 16 MB default buffer re-read with
  nothing in between takes 17.6 µs at set-aside 0, 63 µs at 4.5 MB and 70-85 µs at
  9-18 MB. An 8 MB one takes 12.4 µs at 0 and 24-46 µs with a set-aside.
- **Stale evict-last lines crowd out default lines at every set-aside, 0 included.**
  With 20 MB of earlier evict-last lines resident, a 16 MB default re-read takes 75 µs
  (DRAM speed) at every size, against 17.7 µs without them. The windows' evict-last
  prefetches leave such lines behind them: something for the prefetch schedule to know.
  This was sent to the PDL session as well.
- The 96 MB stream itself: 490.8 µs default, 493.9 µs evict-first, 420.4 µs evict-last.

### Bounds by option (short-context decode, TP4)

| Option | Upper bound | Verdict |
| --- | --- | --- |
| Evict-first on routed weights | about 0 at c1, about 0.25% at c8 | Dropped. No window consumer reads its lines after the routed stream: the shared gate/up runs at L2 speed before FC1 starts, and the NEXT window is issued after the combine. On GB10 the evict-first stream also lowers the prefetched windows' survival. |
| Evict-first plus a set-aside (6 MB, 12 MB, maximum) | none | Not worth a window. Evict-last lines already survive at set-aside 0; a set-aside only takes L2 from default lines. |
| Evict-last on reusable activations, plus a set-aside | about 0 | Not built. At decode the MoE input is 6 rows x 5120 (30 KB of E4M3, 61 KB of BF16), written microseconds before its readers and re-read continuously; default loads keep it. Prefill SP's gathered rows (16K x 5120 = 84 MB) exceed the L2. On GB10 it would need no set-aside. |

So no L2 policy combination has a ceiling worth testing. The specs for them
(`evict-first-ab.json`, `l2-setaside-sweep.json`, and their arms) were removed.

Long context may change the picture: the indexer streams the compressed index cache
between the WO window's prefetch and WO-A/WO-B, and at 1M that cache is tens of MB a
layer. A long-context decode trace would bound it; these traces are short-context.

## The change (vLLM, on the release's tilelang-ports 26382f5cf)

| Change | Branch, commit | Files |
| --- | --- | --- |
| mHC finalize writes the cast | **`mhc-mxfp8-cast` a5a7c5325** | `tilelang/{mhc,linear,gemm,moe}.py`, `fused_moe/tilelang.py`, `nvidia/model.py`, `attention.py`, `kernels.py`, `b12x_layers.py`, six test files, `benchmarks/kernels/benchmark_deepseek_v41_mhc_cast.py` |
| Routed weights evict-first (dropped) | `moe-evict-first` d9849b2d4 | `tilelang/moe.py`, its test file, `benchmarks/kernels/benchmark_deepseek_v41_moe_evict_first.py` |

Take `mhc-mxfp8-cast`. `moe-overlap` (d9849b2d4, then 1c458d618, which is a5a7c5325 on
top) carries the dropped evict-first commit. The branches are local to
`~/projects/vllm-ds41-tilelang-migration`: pushing them to the fork would also publish
the frozen release branch's history.

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

**Tests.** Every mHC entry test also checks the cast against `per_token_cast` of the `y`
it wrote. Dedicated tests cover both finalize widths, decode and prefill sizes, every
scale regime (zero blocks, below the 1e-4 floor, above 448, powers of two), a NaN token,
and the regrouped rows' cast. A fed projection gives the self-casting projection's bits
at 1-300 rows and fails without its cast; a split-K projection cannot be fed. The MoE
tests hand the plan its MXFP8 rows. The CED and SP model tests' stand-ins return the
sixth output.

**Benchmark.** Decode entries (finalize plus one cast at attention, two at the FFN) and a
TP4 prefill SP layer against the release's finalize plus the casts it replaces, bits
first. Gate: no case slower. It passed (Results).

## Lab plan

The arms are candidate 6's dev arms with the change's package files mounted:
[mhc-cast-tp4-dev.json](mhc-cast-tp4-dev.json) on the ring (`rc6/rc-tp4-dev.json`) and
[mhc-cast-tp3-dev.json](mhc-cast-tp3-dev.json) on the triangle (`rc6/rc-tp3-dev.json`).
Candidate 6's image is spark-ds41f-1025's, so the arms compare against the release.

1. The kernel bundle has passed (Results); no kernel job is left to run.
   [kernels.json](kernels.json) is the cluster form of the bundles, kept as a record;
   the bundles ran through `kernel-local` on dgx4 instead.
2. [mhc-cast-ab.json](mhc-cast-ab.json) (ring) and
   [mhc-cast-ab-tp3.json](mhc-cast-ab-tp3.json) (triangle). Each boots the change and
   candidate 6 for the determinism gate (determinism, greedy, and distinct streams at
   c16/c8/c4 on the ring, c8/c4 on the triangle, whose limit is 8), then runs the suites
   on the change, candidate 6 and the change again: `screen` and `single-stream` on the
   ring, `screen-tp3` and `single-stream` on the triangle.
   - **Outputs:** identical. Every reference digest equals candidate 6's
     ([tables_arms.py](tables_arms.py)), and the screen's model step reads KL 0.
   - **Speed:** the screen's engine grid (step cost at 1-96 rows) faster beyond its
     interval, with no metric slower: c16 (c8 on the triangle), prefill 32K/256K, single
     stream. The repeat arm bounds drift.

```sh
python3 experiments/2026-10-25-moe-overlap/make_bundles.py ~/projects/vllm-ds41-tilelang-migration
python3 scripts/lab.py run experiments/2026-10-25-moe-overlap/mhc-cast-ab.json --dry-run
python3 scripts/lab.py run experiments/2026-10-25-moe-overlap/mhc-cast-ab-tp3.json --dry-run
```

`make_bundles.py` rewrites the cast bundle and both arms from `mhc-mxfp8-cast` after a
fix.

## Open

- **Other models on `--moe-backend tilelang`.** `TileLangExperts` now reads only
  handed-over MXFP8 rows, so a model whose producer writes none (the plain DeepSeek-V4
  model could select the backend) fails at its first forward. The backend is documented
  as DeepSeek V4.1's.
- **NaN blocks.** Finalize's amax drops a NaN (`fmaxf`); whether `per_token_cast`'s BF16
  reduction does too is unverified. They agree on a whole NaN row either way (the 1e-4
  floor's `fmaxf` absorbs a NaN amax), and a whole row is the only NaN finalize writes (an
  RMSNorm of a NaN token); the NaN-token test checks it. A block mixing NaN and finite
  values could differ, but no finalize output has one.
- **Next producers, same pattern.** Attention's Q-B and the indexer's Q-B both cast
  `q_norm`'s output (the same rows twice), and the shared expert's down projection casts
  its SwiGLU's output: the rest of the ~1.3%.
