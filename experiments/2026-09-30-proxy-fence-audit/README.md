# Proxy-fence audit of the other B12X TMA pipelines (2026-09-30)

0007 (`experiments/2026-09-29-determinism`, round 7) fixed the dense GEMM: it
released each TMA-filled stage while its last ldmatrix reads of that stage were
still in flight, and with another kernel's CTAs on the SM a delayed read saw the
refill. This audit asks which other B12X kernels combine async-proxy stage fills
(TMA, `cp.async.bulk`) with generic-proxy reads (ldmatrix, LDS) and a release
that permits the refill, without `fence.proxy.async` between them.

Base: r5m, B12X tree 35299956 (f8069b2c plus `patches/b12x/series`), image
`vllm-ds41f-kkref:04c30fa98e79-r5m`. No configuration change.

## Method

1. **Source.** Every B12X file that issues a TMA load or `cp.async.bulk` into
   shared memory (22 files): for each release that lets a producer refill a
   stage, is there a fence, and are the stage's loaded registers consumed
   before the release?
2. **SASS.** Consumption in source is not enough: the MMA that consumes a load
   is not a memory operation, and the compiler may schedule it after the
   release. The compiled r5m kernels come from dgx1's compile cache
   (`cache/kkref/jit/b12x/compile/<key>.o` beside `<key>.json`, which names
   the kernel class): `extract_fatbins.py` writes the embedded fatbin,
   `cuobjdump -sass` disassembles it, and `sass_pending.py` decodes each
   instruction's scoreboard control bits (high word: write barrier at bits
   46-48, wait mask at 52-57) and runs a dataflow over the control-flow graph
   of outstanding LDS/LDSM. Shared loads complete in order (ptxas gives a
   scoreboard only to the last load of a batch), so a wait on one load's
   scoreboard also completes the earlier ones. It reports the shared loads
   still outstanding at every mbarrier arrive and CTA barrier, and whether a
   `MEMBAR`/`FENCE.VIEW.ASYNC.S` lies between.
   - Validation: it flags every stage release of the pre-0007 dense GEMM
     builds (two to five LDSM outstanding) and none of the 0007 builds. The
     0007 fence compiles to `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S`, and the
     arrive waits on the fence's scoreboard.
3. **Serving inventory.** Of the 41 kernel classes (entry points) in the
   serving cache, sampled at up to two builds each, eight issue `UTMALDG` or
   `UBLKCP`; each is below.
4. **Stress.** Where the SASS shows loads outstanding at a release,
   `fence_race_stress.py` (adapted from `gemm_race_stress.py`) runs the kernel
   on a side stream beside the routed MoE and compares every output with the
   solo result, shipped and with the fence (`run_stress.sh`, dgx3, cluster
   stopped; its fenced arm and log are labelled `0008`, the name of the
   staging directory, not a patch number).

## Serving kernels (r5m SASS)

| Kernel | Fill | Loads outstanding at the stage release | Verdict |
|---|---|---|---|
| Dense GEMM (`_lib/dense_gemm.py`: block-FP8 linears, fused-quant-A) | TMA | 2-5 LDSM before 0007, none with it | fixed by 0007 |
| BF16 prefill projection (`gemm/bf16_gemv/_prefill.py`; K 5120 to N 384, 512, 1024; plans of at least 128 rows for N 1024, 256 otherwise) | TMA, 2 stages | 5 LDSM of the stage's last k block; the HMMAs that use them come after the arrive | hazard |
| mHC TF32 prefill projection (`norm/mhc/_kernels.py` `MHCPrefillTf32ProjectTmaKernel`; 64x24x64, 2 stages, split-K 4) | TMA | 4 LDS/LDS.U16 | hazard |
| Contiguous varlen attention (`attention/_shared/contiguous/forward.py`; vision tower) | TMA, K and V stages | 4 LDSM.MT88 at the V release | hazard |
| Sparse-MLA unified decode (`attention/_shared/mla/kernel.py`) | `cp.async.bulk` KV rows | none of the KV rows; one LDS at `+0x166e0`, above the mbarriers (`0x16100`), outside every bulk-filled region | safe |
| Sparse-MLA MG prefill (`attention/_shared/mla/prefill_mg.py`) | `cp.async.bulk` | none at the refill barrier (`BAR.SYNC 1`, all 384 threads) | safe |
| Routed MoE, W4A8-MX A8 (`moe/_shared/kernels/dynamic.py`) | `cp.async` only | no TMA or bulk copy in the W4A8 functions | not in scope |

In all three hazards the source consumes the loads before the release; the
compiler moved the consuming MMAs below it. The window is the one 0007 closed:
in the BF16 prefill projection, for example, the consumer loads all of a
stage's fragments, arrives on the empty barrier at `0x0b00`, and issues the
last k block's HMMAs afterwards. The corresponding source sites are
`_prefill.py:324`, `mhc/_kernels.py:2993` (and the BF16 TMA twin at 2514,
not in the serving cache) and `contiguous/forward.py:1098`/`1130`.

## Kernels DS4.1 does not run (source only)

- **Routed MoE, NVFP4 and W6A8 recipes** (`dynamic.py` 6938/6941, 7256,
  8153): the pre-0007 dense GEMM shape in source. The last k block's copies
  are issued in the previous iteration and the release comes before the MMA
  that uses them. W4A8 compiles these paths out.
- **Consumed before the release in source; SASS not checked**: paged forward
  and extend (`attention/paged/forward_paged.py`,
  `forward_extend_generic.py`), paged decode, dense MLA, the MTP feedback
  prefill GEMM, the GDN boundary kernel, the delta-prefill recurrence, and
  the mHC BF16 TMA projection (fenced by the patch below anyway). Given the
  three serving kernels, treat these as unverified.
- **Safe regardless of scheduling**: the loaded values feed a shared store
  before the `bar.sync` that permits the refill, and a store cannot move past
  the barrier (DSA indexer paged-logits K repack; FP8 extend expansion in
  `forward_paged.py`).
- **Plain write-after-read races, not the proxy bug**:
  `PagedFp8DecodeRawForwardKernel` (`forward_paged.py` 8827-8855) and the
  q32 variant of `PagedFp8ExtendRawForwardKernel` (11300-11342). Warp 0
  issues the TMA refill of the only stage before the barrier that would wait
  for the other warps' reads. Only
  `tests/attention/test_cute_migration_paged_corpus.py` builds these classes.
- **Asides**:
  - Hand-rolled kernels initialise mbarriers without an init fence before the
    first TMA (DSA indexer, delta prefill, raw paged kernels).
    `PipelineTmaAsync.create` fences.
  - `quantization/mxfp6/bf16_to_fp6_tma.py` fills a TMA stage that nothing
    reads.

## Stress (`run_stress.sh`, dgx3, 2026-09-30 08:11-08:17 UTC)

Setup:
- Cluster stopped. Fresh r5m containers.
- Routed-MoE co-runner from the det-slices overlay, as in `run17.sh`.
- `B12X_DENSE_SPLITK_TURBO=0`.
- The fenced arm mounts the three patched files over the image; the image's
  shipped files were checked by sha256 first.
- Each cell below is wrong outputs out of all calls, compared bitwise with the
  result computed alone. "Alone" (no co-runner) was 0 in every case, and every
  target repeated exactly alone.

| Target | Shape | Shipped: copy | Shipped: MoE | Fenced: copy | Fenced: MoE |
|---|---|---|---|---|---|
| dense GEMM (control, down projection) | rows 6 | 0/12000 | 0/12000 | - | - |
| dense GEMM (control) | rows 48 | 0/12000 | **12/12000** | - | - |
| BF16 prefill | N 384, rows 256 | 0/12000 | 0/12000 | 0/12000 | 0/12000 |
| BF16 prefill | N 384, rows 1024 | 0/12000 | **15/12000** | 0/12000 | 0/12000 |
| BF16 prefill | N 512, rows 256 | 0/12000 | 0/12000 | 0/12000 | 0/12000 |
| BF16 prefill | N 512, rows 1024 | 0/12000 | **45/12000** | 0/12000 | 0/12000 |
| BF16 prefill | N 1024, rows 256 | 0/12000 | **2/12000** | 0/12000 | 0/12000 |
| BF16 prefill | N 1024, rows 1024 | 0/12000 | **25/12000** | 0/12000 | 0/12000 |
| mHC TF32 projection | rows 256 | **29/12000** | 0/12000 | 0/12000 | 0/12000 |
| mHC TF32 projection | rows 1024 | **59/12000** | **609/12000** | 0/12000 | 0/12000 |
| varlen attention | 1024, 4096 tokens | 0/3000 | 0/3000 | 0/3000 | 0/3000 |
| varlen attention | 256, 512 tokens | 0/12000 | 0/12000 | - | - |

- **Control.** The dense GEMM control reproduces as in `run17.sh`, although
  this time only at rows 48.
- **BF16 prefill and mHC projections.** Both return wrong outputs beside
  co-resident work, as shipped, and none fenced. For the mHC projection,
  a bandwidth-bound copy on the other stream is enough.
- **Varlen attention.** The V-stage hazard is in its SASS, but I could not
  trigger it with this co-runner.
- **Timing (alone, µs/call, shipped vs fenced).** Single runs, not a screen:
  - BF16 prefill 45.1/45.4 up to 242.5/246.2.
  - mHC 50.0/49.1 and 214.1/218.0.
  - varlen 72.1/72.2 and 960.6/968.5.
- **SASS of the builds compiled in this run**
  (`runs/sass-stress-builds.txt`). The shipped arrives have four or five
  shared loads outstanding. The fenced arrives are fenced, or have none
  outstanding.

Receipts are in `runs/`:
- `stress-dgx3.txt` is the full log.
- `sass-r5m-serving-cache.txt` is the checker's output on the serving cache,
  with the dense GEMM builds from before and after 0007.

## Serving impact

- The mHC TF32 projection is the prefill projection of every layer's
  hyperconnection. The serving cache holds it at the tested geometry.
- The BF16 projection serves K 5120 linears on prefill-sized plans.
- Both go wrong only when another kernel's CTAs share the SM. Serving has
  such overlap: the side-stream shared expert, and the next kernel starting
  early.
- I have not measured the rate in serving. It would appear as occasional
  wrong prefill activations, the same class of symptom as the shared-expert
  columns 0007 fixed.

## Proposed fix

`b12x-fence-stage-reads-three-kernels.patch` (B12X tree
044564ec on top of r5m's 35299956) adds
`cute.arch.fence_proxy("async.shared", space="cta")` before each stage release
of the three hazards, as 0007 did for the dense GEMM. It also fences the mHC
BF16 TMA projection's release, the TF32 kernel's twin. It does not touch the
other kernels DS4.1 does not run, nor any file 0007 changes.

In the production series it would be `patches/b12x/0005`, after r5n's 0004
(0007, the dense GEMM fence). It has not been built into an image or screened
in serving. It is a candidate for the release after r5n. Before that it needs:
- a lean decode and prefill screen against the r5n reference;
- the r5n determinism probe with it.

Upstream follow-ups, not in this patch:
- the NVFP4/W6A8 MoE releases;
- a SASS pass over the kernels DS4.1 does not run;
- the two single-stage write-after-read races in the raw paged kernels;
- the mbarrier init fences.
