# Current state

Production is **spark-ds41f-1025** ([config/cluster.json](../config/cluster.json)), serving
DeepSeek-V4.1-Flash-1017 at 1,048,576 tokens. Since 2026-10-26 21:46 CET it runs the
`tp4-ring` shape on dgx1-dgx4 again, after re-cabling. It was qualified on that shape first and
served there from 2026-10-25 18:31, then on the `tp3-triangle` shape (dgx1-dgx3) from 22:18
until 2026-10-26 08:10. It was promoted
2026-10-25 as [`2026-10-25-spark-ds41f-1025`](../manifests/baselines/2026-10-25-spark-ds41f-1025.json)
([release record](../experiments/2026-10-25-spark-ds41f-candidate/README.md),
[release notes](../experiments/2026-10-25-spark-ds41f-candidate/release-notes.md)): image
`spark-ds41f:19f2c20ed4d6-next-6`, spark-ds41f-1022's sources with candidate 4's vLLM 0228-0265, the
TileLang completion (vLLM `tilelang-ports` through 26382f5cf, here 0300-0488) and two test fixes,
0489-0490 (patch head `11171760`, tree `6f8e062f`). Every kernel serving launches is TileLang except
NCCL and a few rare-path ATen copies; no B12X preparation runs (B12X keeps the checkpoint loader).
0228 draws fresh random numbers in each block-verification round (vllm-project/vllm#61161);
0229-0255 move the CED row selection to TileLang and skip its nonfinal rows; 0256-0257 are the
tokenizer's generation controls and the API's `thinking` switch (#447); GLP steering is available,
off unless configured (#401); `SPARK_DS41F_*` variables; the sink log states its scope (#489); a
`prompt_logprobs` request no longer kills the engine under sink rejection (0266) and
`logprob_token_ids` work under speculative decoding (0267); at most 32 top logprobs. The recipe
moves to the spark-ds41f names (container, image, labels, `/opt/spark-ds41f`, `/cache/spark-ds41f`,
`spark-ds41f-kernel-memory`), sets the ring's long-prefill budget to 16,384 (its startup line reads
6.21 full 1M contexts, from 8.18; the pool holds 11 at either budget,
[KV capacity](memory-profiles.md#kv-capacity)), pins glibc malloc (#439), reports prompt-cache hits (#469), turns the JIT
monitor off and drops 21 settings for kernels serving does not run. Temperature-0 text outputs are
identical to spark-ds41f-1022's; image answers differ and score the same.
Against spark-ds41f-1022 in candidate 6's release comparison (one session, 2026-10-25
14:46-18:20 CET, verdicts against the repeat): prefill +12.8% / +13.9% / +13.7% / +12.9% /
+11.8% at 32K / 131K / 262K / 524K / 1M (better), decode grid -2.38% (better); c16 +0.12%, c8
+0.88%, Poisson +0.01% and one stream +2.11% the same; sampled the same; greedy 256/256
identical, teacher-forced KL 0.
`tp4-ring`'s `benchmark_reference` is
[manifests/suites/2026-10-25-spark-ds41f-1025-tp4](../manifests/suites/2026-10-25-spark-ds41f-1025-tp4/)
(candidate 6's first arm of its release comparison). `tp3-triangle` ran an 8,192-token
prefill budget in spark-ds41f-1025 (spark-ds41f-1022: 4,096); the recipe now sets 6,144, for a
median step beside a long prompt 21% shorter at about 3% of prefill (owner, 2026-10-27), and
the next release qualifies it. Its `benchmark_reference` is
[manifests/suites/2026-10-25-spark-ds41f-1025-tp3](../manifests/suites/2026-10-25-spark-ds41f-1025-tp3/)
(candidate 6's arm of its two-arm TP3 comparison). Against spark-ds41f-1022 on the triangle:
prefill +25.5% to +28.2% from 32K to 500K and +22.5% at 1M; decode grid -1.65% (better); c8,
c4, Poisson and one stream the same; greedy 256/256 identical, KL 0. The 4 KiB-page triangle
is not qualified on this image. spark-ds41f-1022's recipe and lock are frozen in `experiments/2026-10-23-next-release/1022`
as the rollback.

Before spark-ds41f-1025 production was **spark-ds41f-1022**, serving DeepSeek-V4.1-Flash-1017 at 1,048,576 tokens on dgx1-dgx4 since
2026-10-23 (the `tp3-triangle` shape on dgx1-dgx3 from 2026-10-22), promoted
2026-10-22 as [`2026-10-22-spark-ds41f-1022`](../manifests/baselines/2026-10-22-spark-ds41f-1022.json)
([promotion record](../experiments/2026-10-22-spark-ds41f-1022/README.md)), the first release named
`spark-ds41f-MMDD`: image `vllm-ds41f-kkref:19f2c20ed4d6-1022-4`, r6x's sources with vLLM
0185 and 0221-0227 (patch head `8fc15c18`, tree `e68aa19c`) and sparknet 0.5.0. 0185 runs the TP
group's broadcast (DSpark's per-step confidences) on sparknet's one-shot all-gather, so serving
never touches the PyTorch NCCL process group; 0221 and 0223 detect the model's sink and reject a
held draft zero that lands on it, on by default for DeepSeek V4.1, so short prompts such as
`Say hi.` no longer collapse to an empty answer; 0222 and 0226 decide the next step at decoder layer
30 of the current one (late admission), so an arrival joins one step sooner, while a step that
continues a prompt chunk decides its successor at once; 0227 keeps control-token spellings in
request text as plain text (#449); sparknet 0.5.0 widens the one-shot read phase. Outputs are
identical to r6x's. Against r6x on the trimmed promotion comparison (one session, the owner's choice): decode
steps 1.1% faster, eight streams, one stream and prefill level once 0226 restored prefill (its A/B:
0.6-1.4% on every paired prompt); time to first token level at four and eight streams and one decode
step longer alone (0.127 to 0.159 s at p50). Release tests: 1,801 passed.
`tp3-triangle`'s `benchmark_reference` is
[manifests/suites/2026-10-22-spark-ds41f-1022-tp3](../manifests/suites/2026-10-22-spark-ds41f-1022-tp3/).
On the ring (2026-10-23, trimmed comparison against r6x in one session): decode steps 1.7%
faster, sixteen streams, one stream and prefill level, outputs identical to r6x's ring hashes;
`tp4-ring`'s `benchmark_reference` is
[manifests/suites/2026-10-22-spark-ds41f-1022-tp4](../manifests/suites/2026-10-22-spark-ds41f-1022-tp4/). r6x's recipe and lock are frozen in `experiments/2026-10-22-spark-ds41f-1022/r6x` as the
rollback.

Before spark-ds41f-1022 production was **r6x**, serving DeepSeek-V4.1-Flash-1017 on the recipe's
`tp4-ring` shape, promoted 2026-10-19 as
[`2026-10-19-karmic-kraken-r6x`](../manifests/baselines/2026-10-19-karmic-kraken-r6x.json)
([promotion record](../experiments/2026-10-18-r6x/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6x` on dgx1-dgx4, r6v's sources with vLLM 0181-0184 (patch head
`7316c7f9`, tree `4d6851c8`). 0181 runs the routed experts' SwiGLU and MXFP8 cast over the
dispatch layout's live tiles only; 0182-0183 keep the routed experts' weight scales as a row
base exponent and 4-bit codes (exact, half the scale bytes, about 2.1 GiB a rank freed); 0184
budgets DSpark's drafts by a rule of the inputs, a fixed linear step cost (row cost 0.035), so
nothing is timed at boot and every build drafts the same way. Outputs are identical to r6v's.
Against r6v on the `release` suite: decode steps 2.5% faster, 16 and 8 streams 3.6% and 3.8%
faster (sampled 3.0% and 4.5%), prefill level, one conversation at a time 74.1 tok/s.
`tp4-ring`'s `benchmark_reference` is
[manifests/suites/2026-10-19-karmic-kraken-r6x-tp4](../manifests/suites/2026-10-19-karmic-kraken-r6x-tp4/).
On the triangle, `tp3-triangle` serves r6x at **1,048,576 tokens** since 2026-10-19
([experiments/2026-10-19-r6x-tp3-1m](../experiments/2026-10-19-r6x-tp3-1m/README.md)): a
4.90625 GiB KV pool that holds five full 1M contexts at once (r6x's startup line at its 4,096
budget: 4,765,898 tokens, 4.55x; [KV capacity](memory-profiles.md#kv-capacity)), outputs identical to r6u's TP3
path; its `benchmark_reference` is
[manifests/suites/2026-10-19-karmic-kraken-r6x-tp3](../manifests/suites/2026-10-19-karmic-kraken-r6x-tp3/).
r6v's recipe and lock are frozen in `experiments/2026-10-18-r6x/r6v` as the rollback.

Before r6x the ring ran **r6v**, promoted 2026-10-17 as
[`2026-10-17-karmic-kraken-r6v`](../manifests/baselines/2026-10-17-karmic-kraken-r6v.json)
([promotion record](../experiments/2026-10-17-r6v/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6v`, r6u's sources with vLLM 0180 (patch head
`c001cc4e`, tree `d11e9836`). 0180 relays Engram's sequence-parallel row all-to-all through the
neighbours at four ranks: on the switchless ring ranks two apart share no cable, and r6u's
direct exchange (r6s's, released on the triangle only) timed out there in memory profiling.
Against r6n on the `release` suite: prefill 3.2-4.2% faster at 32K-1M, decode level, one
conversation at a time 71.1 tok/s. It scheduled on its build's own DSpark cost curves
(`build-2a95f7780992adb4`). The triangle ran the same image on r6u's TP3 path (below, its
reference r6u's run). r6u's recipe and lock are frozen in `experiments/2026-10-17-r6v/r6u`.

On the triangle, r6v's TP3 path is **r6u**'s, serving DeepSeek-V4.1-Flash-1017 (revision `1bae0e10`), promoted 2026-10-17 as
[`2026-10-17-karmic-kraken-r6u`](../manifests/baselines/2026-10-17-karmic-kraken-r6u.json)
([promotion record](../experiments/2026-10-17-r6u/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6u` on dgx1-dgx3, r6s's sources with vLLM 0156-0179
(patch head `67403b8d`, tree `53b007b5`) and TileLang 0004 (tree `222b895d`). r6u adds:
- **Kernel preparation** (0156-0172, TileLang 0004). Before memory profiling each rank
  compiles every declared TileLang call (186 at TP3), runs TileKernels' warm calls and
  the step's Triton kernels, then freezes TileLang's cache; profiling, warm-up, capture
  and serving compile nothing, and a kernel not prepared fails by name on every rank.
  The startup rendezvous waits on liveness (forum 384438#24: a cold rank used to stall a
  one-shot collective). One rank's TileLang cache empty: ready in 159 s, the others
  waiting on its heartbeat; every JIT cache empty: 180 s.
- **DeepSeek's current reference** for both V4.1 Flash checkpoints: WO-A's input through
  `act_quant` (0177), prepared once per row in the sparse MLA kernels' output store
  (0179); the 2026-10-16 encoding (0178).
- **The 1017 checkpoint** (the hub cache mounted whole, `models.v41-flash-1017`) with
  DeepSeek's sampling defaults (`generation_defaults`: temperature 1.0, top_p 1.0).
- **Release tests** on the promoted image ([testing](testing.md)): 58 DS4.1 Flash gate files.

Against r6s on 1017 (`release-tp3`): prefill level at 32K-500K, decode level (the step
time 1.0% faster, cross-session), outputs different by WO-A's input quantization.
`tp3-triangle`'s `benchmark_reference` is
[manifests/suites/2026-10-17-karmic-kraken-r6u-tp3](../manifests/suites/2026-10-17-karmic-kraken-r6u-tp3/)
(0179 on the image through 0178: the same files). It schedules on its build's cost curves
(`build-111f0abb94a12639`, profiled on the promoted image's first boot). The r6s recipe
and lock are frozen in `experiments/2026-10-17-r6u/r6s` as TP3's rollback.

TP3 ran **r6s** before it, promoted 2026-10-16 as
[`2026-10-16-karmic-kraken-r6s`](../manifests/baselines/2026-10-16-karmic-kraken-r6s.json)
([promotion record](../experiments/2026-10-16-r6s/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6s` (`sha256:2414b9d1`) on dgx1-dgx3, r6n's sources with
vLLM 0146-0155 (patch head `238b7ea3`, tree `71c6e1df`). r6s changes only how Engram's
rows reach a prefill:
- **Rows before the step.** The native reader reads the rows of the tokens the host
  already knows before the step starts (0148), so the first Engram layer waits less.
- **An all-to-all for the prefill product.** Under sequence parallelism each rank sends
  every other rank only that rank's real rows of its `wkv` column block (0150, 0152),
  and the TileLang gate reads the received blocks in place.
- **Interleaved hash columns.** A disk shard is whole hash columns, every tp-th (0153,
  0154): every rank reads 2-, 3- and 4-gram heads, the ranks' columns are joined by an
  all-gather, and the slowest rank's reads no longer set the step.
- **Checkpoint copies through anonymous memory** (0147), never pinned: a large copy
  from untouched file-mapped pages hangs on R580 with 64 KiB pages.

Against r6n on the `release-tp3` suite: prefill 4.8-5.6% faster at 32K-500K, decode,
8 and 4 streams and Poisson level, temperature-0 text and teacher-forced
distributions identical. `tp3-triangle`'s `benchmark_reference` is the release's r6s
run ([manifests/suites/2026-10-16-karmic-kraken-r6s-tp3](../manifests/suites/2026-10-16-karmic-kraken-r6s-tp3/)).
It schedules on its build's cost curves (`build-75cd5b96c4ec4942`, profiled on the
release's first r6s boot). The r6n recipe and lock are frozen in
`experiments/2026-10-16-r6s/r6n` as TP3's rollback
(`experiments/2026-10-15-r6n-tp3/r6n-tp3.json`, curves `build-77b3fa3c35f1ce6d`).

Before r6v, `tp4-ring` resolved to r6u but was not qualified on it (no
`benchmark_reference`); the four-node ring was last released on r6n, in the frozen
recipe's `tp4-ring` shape, described below.

The four-node ring's last release is the recipe's `tp4-ring` shape, promoted 2026-10-15 as
[`2026-10-15-karmic-kraken-r6n`](../manifests/baselines/2026-10-15-karmic-kraken-r6n.json)
([promotion record](../experiments/2026-10-15-r6n/README.md)): image
`vllm-ds41f-kkref:19f2c20ed4d6-r6n` (`sha256:5ec81bee`) on all four ranks, Local
Inference Lab vLLM `19f2c20e` with r6k's series plus vLLM 0134-0145 (patch head
`b1309b4a`, tree `0c60314d`), TileKernels with the routed SwiGLU NaN patch (tree
`cb2fa41c`), sparknet `timeout-stop` (`3bc8d10`). r6n keeps r6k's arithmetic and its
temperature-0 text exactly, and changes how the engine fails and how Engram rows
arrive:
- **Engram rows v2.** One native worker (no Python, no GIL) reads both tables in
  layer order; in the forward a one-thread TileLang kernel waits for each table's
  step, bounded by `SPARK3_ENGRAM_ROWS_TIMEOUT_MS` (10 s) and stopped at once by a
  failed read, and the TileLang gather loads only this rank's rows. Decode steps are
  2.2% shorter than r6k's; prefill is level.
- **Collectives that stop.** sparknet's waits time out by the GPU clock
  (`SPARKNET_ROCE_TIMEOUT_MS`, 10 s) and trap naming the peer and lane; the RoCE
  policy has no fallback to other backends (vLLM 0142).
- **Startup and the decode path.** Ranks meet on the CPU group before profiling and
  after the JIT warm-up (0143); the draft budget runs eagerly, so nothing compiles on
  the first decode step (0144), which removed the first-request crash's trigger.
- **NaN contracts.** The shared and routed SwiGLU clamps and the Engram gate floor
  keep a NaN instead of hiding it; ten kernel families have non-finite contract
  tests (r6l).

r6n was released on the servebench release suite. Its run
([manifests/suites/2026-10-15-karmic-kraken-r6n-tp4](../manifests/suites/2026-10-15-karmic-kraken-r6n-tp4/))
was `tp4-ring`'s `benchmark_reference` until r6v. It ran release version 1; r6v's release
recorded the first version 2 run (with the Poisson step).

Since 2026-10-14 the TP4 profile pins serving to the GB10's ten fast cores
(`--cpuset-cpus=5-9,15-19`; `experiments/2026-10-14-boot-spread`). Serving keeps three or
four threads busy per node, so the slow cores are never needed. `doctor --live` checks the
pin on every node.

Since r6x DSpark's verifier budgets drafts by a fixed linear step cost and times nothing,
on both shapes. Before it, the verifier scheduled on cost curves timed on a boot. A profile
without a pinned directory got its build's own (`bin/spark` derived `SPARK3_DSPARK_COST_DIR`
from the image's source trees, code overlays, environment and serve arguments, all but the
dev-mode RPC switch): a new build profiled itself on its first boot and reused its curves
after. `tp3-triangle` did so from 2026-10-15; `tp4-ring` pinned r6h's curves
(`ring4-r6h-headout-20261012`) through r6n and ran on its build's own from r6v.

TP4 keeps the embedding and head in regular memory
(`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0`, `experiments/2026-10-12-r6h-loss`); TP3 keeps them in
the display carve-out, which its KV budget needs. TP4's temperature-0 and greedy references
are r6v's (3eab5112 / db25fde3, c9cf9dbe / c6bb4f4e), which r6x reproduces; TP3's are r6u's
TP3 path (62313cb5 / 9742ac2c, 4d143977 / 12fd8257), which r6x at 1M reproduces.

Before r6s, the three-node shape `tp3-triangle` ran r6n, promoted 2026-10-16 on its release
against r6h ([experiments/2026-10-15-r6n-tp3](../experiments/2026-10-15-r6n-tp3/README.md),
reference [manifests/suites/2026-10-15-karmic-kraken-r6n-tp3](../manifests/suites/2026-10-15-karmic-kraken-r6n-tp3/)):
decode 5.5% faster at every shape, 8 and 4 streams faster, prefill level. It schedules
on its build's cost curves (`build-77b3fa3c35f1ce6d`, profiled on the release's first
r6n boot). `tp3-triangle-4k` (the 4 KiB-page kernel) is not qualified. r6h's archived
TP3 record (`experiments/2026-10-14-one-recipe/tp3-r6h-production.json`) stays as the
rollback. The TileLang kernel family runs the model;
B12X is the alternative backend in the same image.

| Setting | Active value |
|---|---:|
| Kernel backend | TileLang (`kernel_backend: tilelang`): TileLang 0.1.15 + SM120/SM121 block-scaled MMA patches, DeepSeek TileKernels 2.0.0, sparknet 0.4.0 one-shot RoCE collectives; TileLang also runs RoPE, the compressor and its projection, the KV cache writers, the Engram gate, hash and row gather, the sparse-MLA metadata, the WO projection and the drafter's context-KV projection; B12X runs the vision tower's kernels and the checkpoint loader |
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `19f2c20e` + the r6s series (`experiments/2026-10-16-r6s/vllm/series`, patch head `238b7ea3`, tree `71c6e1df`), B12X `236ddff0` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie, dense GEMM and prefill stage-fence patches, four-node relay and mesh transports, packed BF16 vocabulary projection, NCCL 2.30.7 + IB send-path fence and four-node ring patches |
| Image | `vllm-ds41f-kkref:19f2c20ed4d6-r6s` (`sha256:2414b9d1`), one digest on all ranks, built by `bin/spark build` |
| Decode tiles | block-32 FP8 decode rows on 16-, 32- or 64-row tiles by row count, TileLang's unspecialized pipeline, no K split; prefill rasterizes weights larger than L2 in panels (vLLM 0043) |
| Batch invariance | every kernel adds its reduction in a fixed order; collectives add every rank's partial in rank order with one FP32 rounding; prefill chunks leave room for every stream's decode and draft rows (threshold 4048); a mixed step runs sparse MLA per request, so decode rows keep the decode plan |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia-64k` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 524,288 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025) |
| B12X fixes (alternative backend) | indexer top-k ties by lowest logical position (B12X 0003); async-proxy fences before the TMA refill in the dense GEMM (0004) and in the BF16 and mHC prefill projections and contiguous attention (0005); W4A8 tiny decode off |
| Explicit KV memory | 3.5 GiB per rank |
| Reported KV capacity | 2,845,543 tokens (5.43x full 512K windows); the pool holds seven full 512K contexts at once ([KV capacity](memory-profiles.md#kv-capacity)) |
| Memory-saver | signed DKMS 0.4.1 on all nodes; loaded UVM leaf-table packing enabled |
| Page-size profiles | 64 KiB selected; 4 KiB retains 2.2 GiB KV and a 262,144-token limit |
| Output head | exact 12-bit packed BF16 (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`; B12X 0012, vLLM 0028), bit-identical to the checkpoint; FP32 logits (`VLLM_DS41_HEAD_DTYPE=float32`, the reference's `head_dtype`) |
| Reference precision | the mHC stream projection in FP32 (three exact TF32 parts per weight), the shared expert's SwiGLU in FP32 with one rounding, MoE partials rounded once per rank; WO-A's FP8 activations are the known deviation left |
| Display carve-out | TP3 profiles (TP4 keeps both in regular memory, above): embedding and packed output-head weights in the firmware scanout reserve, loaded there directly (vLLM 0030) (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`, the GPU's DRM card by PCI path, `/dev/dri/by-path/pci-000f:01:00.0-card`, as `/dev/dri/card0`); the DRM file closes after the import, so the text console keeps drawing |
| Sparse attention | TileLang sparse MLA with the MXFP4 indexer; BF16 attention (patch 0023) is available and off pending a fidelity test |
| Image input | vision tower loaded; up to 4 images per request, the recipe's chosen limit, not the model's ([below](#images-per-request)); no host preprocessing cache |
| DSpark | 5 draft tokens, draft TP 3, adaptive verification (r6u's TP3 path: cost scale 2.0; r6x: the row cost 0.035, nothing timed), dead verification rows below survival 0.2, block rejection; vocabulary-parallel greedy drafts over the checkpoint's BF16 drafter head (the target head's own tensor) and Markov projection; the drafter marks its padding rows for the MoE routers (vLLM 0042); cost curves profiled for this build with real rows (`build-75cd5b96c4ec4942`); greedy verification from vocabulary shards when no request needs the logits (vLLM 0112-0114) |
| Collectives | sparknet one-shot all-reduce (up to 2 MiB) and all-gather over the triangle (`oneshot-direct`), larger all-reduces summed in rank order in FP32; the requested policy is enforced and a declined or lost backend fails startup (vLLM 0027, 0038) |
| Profiling context | each KV cache group gets its own blocks, in every dummy layout (vLLM 0039, 0041) |
| CUDA graphs | full, capture sizes 1-48 |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | one native reader thread, rows of host-known tokens read before the step; disk shards on interleaved whole hash columns, joined by an all-gather; one bounded wait per table (`SPARK_DS41F_ENGRAM_ROWS_TIMEOUT_MS`, 10 s) |
| L2 weight prefetch | on (base default on SM121); TileLang scale words prefetched with each weight (vLLM 0044) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s), each killing on its first sample below; an optional steady warn line (`host.memguard_warn_gib`, off) only logs; hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Prefix caching | enabled, 256-token blocks; cache reuse (`--enable-prefix-caching`) and cache reporting (`--enable-prompt-tokens-details`, `usage.prompt_tokens_details.cached_tokens` in every response) are separate switches, and without the second the key is absent while the cache still works |
| Reasoning | enabled by default; a request's `thinking` or `enable_thinking` is honored |

The release comparison against r6d is in the
[r6f results](../experiments/2026-10-11-r6f-release-bench/README.md), and r6's
acceptance benchmark in its
[promotion record](../experiments/2026-10-05-tilelang-r6/README.md); the TP3
profile's long-context admission and retrieval evidence (measured on r5o, same
shape) is in the
[memory-saver deployment](../experiments/2026-10-02-memory-saver-capacity/README.md).
Against r5p, the kernels change family; the weights, the TP3 shape and the
memory guards are unchanged.

## Images per request

`--limit-mm-per-prompt {"image":4}` in the recipe's `serve_args` is the recipe's chosen
per-request limit, set when the vision tower was enabled
([experiments/2026-09-25-vision](../experiments/2026-09-25-vision/README.md)). It is not a
limit of the model. vLLM counts every image in a request, and a chat client resends its
history, so the images retained in earlier turns count too: once a conversation holds a
fifth image, every later turn fails with
`400: At most 4 image(s) may be provided in one prompt`.

A site profile raises the limit by writing another count in that entry (`{"image":8}`), or
omits both entries to take vLLM's default of 999 per prompt. `{"image":0}` (or
`--language-model-only`) skips the vision tower and its 0.9 GiB of weights on every rank.
A profile's `serve_args` is the complete command, so the change is made in the profile's
own list ([setup.md](setup.md#a-site-profile-replaces-serve_args-wholesale)).

The trade-off is memory per request, not at startup. The tower's weights load whenever
images are allowed at all, and vLLM encodes a request's images within a per-step budget
derived from the batched-token budget (6,144 tokens on the triangle, 16,384 on the ring), so
the encoder's working memory per step does not grow with the count. The request does: every
image becomes prompt tokens that hold blocks of the fixed KV pool (`kv_cache_memory_bytes`)
for the life of the request, beside every other stream's context, and lengthen its
prefill; and the head's API server decodes and preprocesses every image in host memory,
which on GB10 is the unified memory dgx1's steady guard watches. No count above four is
qualified here.

**Pixels per image: at most 16,777,216** (`VLLM_MAX_IMAGE_PIXELS` in the recipe, since
2026-10-25). vLLM decodes every image at its source size before DS4.1 resizes it to about
1K tokens, so a request's host memory follows the source pixels, not the tokens. Measured
on the 1025 image (experiments/2026-10-26-site-landmines):
- **Per image in flight:** about 19 MiB at 1,302 px, 61-74 MiB for a 12 MP photo, 310 MiB at
  48 MP, and 790 MiB at vLLM's default ceiling of 178.9 MP.
- **The danger at the default:** four images at that ceiling peaked at 5.2 GiB (JPEG) and
  8.4 GiB (PNG with alpha). dgx1 serves the triangle with about 4.4 GiB above its 3 GiB
  guard, so one such request could stop the service.

The model never sees more than about 1.7 MP, so a 16 MP limit changes nothing for ordinary
images. Larger photos, such as 48 MP phone originals, get a 400 before decoding: vLLM reads
the dimensions from the header and refuses the image before any pixel is decoded. A higher
per-request image count waits for its own measurement.

Externally reported, not qualified here: a three-node TP3 site with the limit removed (r6
image, 524,288-token context, eight sequences) completed a chat-history request with 13
images and 104,615 prompt tokens, cold and warm, and three overlapping replays of it
([#30](https://github.com/christopherowen/spark-ds41f/issues/30)). That shows more than four
working on that site, not that a given count is safe for every image size, workload or
shape.

The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [setup.md](setup.md).
