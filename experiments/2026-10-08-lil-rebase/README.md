# Rebase onto the Local Inference Lab heads (r6c candidate)

Base: r6b ([2026-10-08-ced-routing-mask](../2026-10-08-ced-routing-mask/)),
image `vllm-ds41f-kkref:04c30fa98e79-r6b`, on Local Inference Lab vLLM
`04c30fa9` and B12X `f8069b2c`. This candidate moves both to the current
heads and changes nothing else: TileLang, TileKernels, sparknet, NCCL, CUTLASS,
the CuTe DSL wheel set and the vLLM nightly base image keep r6b's pins.

| Source | r6b | r6c | Upstream commits |
| --- | --- | --- | ---: |
| vLLM `integration/karmic-kraken-beta` | `04c30fa9` (2026-09-26) | `19f2c20e` (2026-10-08) | 313 (147 first-parent) |
| B12X `integration/karmic-kraken-beta` | `f8069b2c` | `236ddff0` (2026-10-06) | 148 |

vLLM's canonical base is still `vllm-project/vllm` `0f8fa53a`, and no native
code (`csrc`, CMake) changed, so the nightly base image and the two rebuilt
stable extensions still match. B12X has to move with vLLM: the new vLLM imports
six B12X names the old pin lacks (CSF and EXL3 loaders, preparation
artifacts); against the rebased B12X every `b12x` import in vLLM resolves.

The B12X repository is now archived; its code continues in FlashInfer
(`flashinfer/experimental/b12x`), and the vLLM head asks for FlashInfer 0.7.1
to supply it. `236ddff0` is the archived repository's last commit. This
candidate keeps building B12X from source with our series and leaves the base
image's FlashInfer 0.6.18.post1 in place; no vLLM change since `04c30fa9`
imports a FlashInfer API the base image lacks. Moving to FlashInfer's B12X is a
separate step, best taken when the owner moves off B12X altogether.

## What the rebase brings to the promoted recipes

- **Adaptive verification no longer stops speculating.** `43210646f`: a refit
  that saw one confidently wrong draft could drive the predicted acceptance to
  about 1e-8, after which no draft was admitted or graded for the rest of the
  request. Refits are now bounded, and a request whose drafts go unverified for
  a while verifies one per step again. Both recipes run adaptive verification;
  long agent requests are exactly where this showed.
- **Shared PyNCCL communicators** (`92dd74f13`, off unless
  `VLLM_SHARE_PYNCCL_COMMS=1`), draft-resource validation (`e9c07125e`) and the
  rest of the branch's engine and scheduler fixes.
- Three of our patches are upstream now and leave the series: 0003 (DSML tool
  parameters, `8ee35eda3`), 0022 (decode metadata across CUDA graphs,
  `8720e64f2`) and 0023 (BF16 sparse-attention arithmetic, `3540d5200`). B12X
  0002 (CuTe DSL 4.7.1) is upstream too.

## Series changes

[vllm/series](vllm/series) holds r6b's patches regenerated on the new base,
except #15's 0045, which still applies unchanged and is used in place from
[2026-10-05-ced-decoder-routing-mask](../2026-10-05-ced-decoder-routing-mask/).
[b12x/series](b12x/series) does the same for the packed-head B12X series.

Four vLLM patches needed resolution; everything after 0029 applied cleanly:

- **0002, asynchronous disk Engram rows.** Upstream `4211209df` removed the
  base's own Engram overlap (its side-stream lookups could be starved behind
  polling kernels in the replayed graph) and now enqueues the gathers before
  replay. Our path stays: the main thread enqueues the side stream's gate,
  decode and flag write before it launches the graph, and both waits are
  stream memory operations (`cuStreamWaitValue32`), not polling kernels. Its
  check that the base overlap is off is gone with the overlap.
- **0012, vocabulary-parallel greedy drafts.** Upstream added its own sharded
  draft argmax for Kimi K3 (`supports_local_draft_argmax`) and an optional
  sharded Markov head (`VLLM_DSPARK_SHARD_MARKOV_HEAD`). Both paths are kept:
  a model's own local argmax takes precedence, the DS4.1 drafter uses ours, and
  the Markov projection is sharded when either asks for it.
- **0027, explicit collective policy.** Upstream binds the RoCE all-gather
  locally; the policy guard on the symmetric-memory path is unchanged.
- **0029, TileLang kernel backend.** Upstream edited the sparse-MLA planning
  block that 0029 moves into `_declare_mla`; the moved block already carried
  the same BF16 pinning and decode-row capacity, so only upstream's type
  annotation is added.

Two patches are new:

- **0046** gives the padded-step router test a forward context's
  `attn_metadata`. r6b's in-image run failed both `test_padding_rows_are_not_routed`
  cases with `AttributeError`: 0045 reads the field, which a real
  `ForwardContext` always has, but that fixture's bare context did not. Tests
  only.
- **0047, `VLLM_DS41_HEAD_DTYPE`.** `3540d5200` also makes FP32 the default
  output-head dtype for DS4.1 and its DSpark drafter. Under FP32 the logits
  processor skips the exact packed BF16 and TileLang vocabulary projections
  (they run only when the head dtype is the model dtype), and the packed head
  would fall back to its quant method and widen BF16 logits. A recipe cannot
  opt the drafter out, because a DSpark draft does not receive the target's
  dict `hf_overrides`. The variable sets the default for both; upstream's
  default stays `float32`.

## Recipe changes

[r6c-tp4.json](r6c-tp4.json) and [r6c-tp3.json](r6c-tp3.json) are r6b's with
the image `vllm-ds41f-kkref:19f2c20ed4d6-r6c`, its vLLM and B12X tree labels,
this lock, and two environment changes:

- `VLLM_DS41_HEAD_DTYPE=model` keeps r6b's BF16 heads, so the rebase is the
  only variable. FP32 logits are closer to DeepSeek's reference head and would
  be their own arm: the packed and TileLang heads already accumulate in FP32,
  so that arm writes FP32 logits from them instead of rounding once to BF16,
  at the cost of the wider logits traffic.
- `VLLM_DS41_ENGRAM_OVERLAP` is dropped; nothing reads it any more.

## Checks so far (CPU)

- `bin/spark --cluster-config experiments/2026-10-08-lil-rebase/r6c-tp4.json build prepare --only vllm`
  and `--only b12x` reproduce the recorded patch heads and trees (vLLM
  `125c404e` / `3135cc05`, B12X `69e5795e` / `1a647a31`) with `git diff --check`
  clean.
- Every Python file the series changes compiles; pyflakes reports nothing new
  against the upstream head.
- No environment variable the recipes set lost its reader in vLLM or B12X
  (TileLang and sparknet are unchanged).
- The repository's unit tests pass.

## Quality and safety gates

One lab window on TP4:

1. Build `-r6c` on dgx4 and load it on all four nodes with one image ID.
2. In the image, with `--noconftest`: the TileLang router tests (including
   0045's and the two padded-step cases 0046 repairs), adaptive verification,
   the acceptance estimator, dead rows and the DS4.1 Engram tests.
3. Boot r6c. The log must not carry the
   `head_dtype=...: a quantized LM head computes in its own format` warning.
4. The mixed-step reproduction, the quality suite, then r6's acceptance
   benchmark (decode prose and code at 1-16 streams, prefill 32K-1M) against
   r6b's TP4 run in the same protocol.

TP3 needs the triangle cabling; [r6c-tp3.json](r6c-tp3.json) is ready for it.

Rollback trigger: any test, quality or request failure, the head-dtype warning,
or a decode point slower than r6b with an interval that excludes zero.

## Acceptance criteria

Serving level with r6b or better at every point, quality 5/5, and no failed
requests. The rebase is accepted for its fixes and for keeping the series on
the maintained branch; it does not need to be faster.

## Status

Gate run on the TP4 ring, 2026-10-08 14:01-14:39 UTC (window
`dgx1-1791468092`), from main `83c5cb5`:

- **Build.** 9 minutes on dgx4; one image ID on all four nodes
  (`sha256:a59e7c93`).
- **Tests in the image.** The router (including 0045's tests and the two
  padded-step cases 0046 repairs), adaptive verification, acceptance estimator
  and dead-row tests pass. `tests/models/test_deepseek_v4_1_engram.py`, new to
  this gate, does not run under `--noconftest`: 15 tests need vLLM's
  `dist_init` fixture, and
  `test_disk_engram_model_allocates_hash_buffer_from_declared_caps` builds the
  model from a stub parallel config without `tensor_parallel_size`, which the
  sequence-parallel prefill patches have read since r5 (r6b's constructor
  makes the same call). Neither is a rebase regression; the disk Engram path is
  covered end to end by quality and the benchmarks below.
- **Boot** 108 s; the log carries no widened-head warning, so
  `VLLM_DS41_HEAD_DTYPE=model` keeps the packed and TileLang heads.
- **Mixed-step reproduction.** Every mixed prefill overlapped all eight
  decoding streams. r6c passed three rounds. r6a passed too: on this ring a
  mixed step alone does not trip the assertion #15 reported from TP3 behind a
  switch, so the fix rests on review and on 0045's GPU tests.
- **Quality** 5/5.
- **Against r6b's TP4 run, same protocol.** Eight decode points have step-time
  intervals including zero; prose-c4 (+0.4%, [+0.1, +0.6]) and code-c1 (+0.5%,
  [+0.2, +0.8]) are slower with intervals excluding zero, which meets the
  rollback trigger as written. Against r6a every decode interval includes zero,
  and prose-c4 is identical (55.68 and 55.69 ms): r6b's 55.49 ms was the fast
  run, so these look like between-boot variation, which three samples within a
  run do not capture. Prefill at 32K, 262K, 500K and 1M tokens: +0.1, -0.2,
  +0.1 and +0.1% against r6b, intervals including zero. No failed requests;
  minimum MemAvailable 20.0 GiB; no thermal slowdown.

Accepted by the owner on 2026-10-08, who reads the two decode points as
between-boot variation; no A/B/A was run.
