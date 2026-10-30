# Current state

Production is **spark-ds41f-1030** ([config/cluster.json](../config/cluster.json), frozen in
[releases/spark-ds41f-1030](../releases/spark-ds41f-1030/)), serving DeepSeek-V4.1-Flash-1017 at
1,048,576 tokens on the `tp4-ring` shape (dgx1-dgx4). It was promoted 2026-10-30 as
[`2026-10-30-spark-ds41f-1030`](../manifests/baselines/2026-10-30-spark-ds41f-1030.json)
([release record](../experiments/2026-10-26-spark-ds41f-candidate/README.md),
[release notes](../experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)): image
`spark-ds41f:1030` (built as `spark-ds41f:19f2c20ed4d6-next-14`, `sha256:7918fd6145b5`),
candidate 14, spark-ds41f-1025's sources with vLLM 0491-0567 (patch head `b6de07aa`, tree
`ea1281f8`). 0491 counts the startup KV figure as the pool holds it (11,554,122 tokens, 11.02
full 1M contexts at the ring's 16,384 budget); 0493-0496 make TileLang the only kernel family
and 0499 loads the checkpoint directly (`--load-format direct`, `VLLM_PLUGINS` empty); 0502-0503
drop per-allocation KV zeroing; 0520-0525 and 0543-0544 run the decode chain under PDL with the
attention front in one launch; 0526-0527 replay a prompt's last 128 CED rows wherever its chunks
end; 0537-0538 load the MXFP8 block-scale grid by row block and make the grouped MoE GEMM's stage
arrivals wait for every copy; 0539-0540 follow the DeepSeek API's effort aliases and prompt
semantics; 0541-0542 serve GLP steering per request under `deepseek-v4.1-flash-weightless`; 0545
is prefill yield (`--prefill-yield short`, 10 s); 0546-0562 and 0565 are IKET's debug mode, off;
0563-0564 and 0566 store 2-bit expert scale codes in the 4-bit codes' buffer; 0567 has the GPU
hash the Engram rows of a prefilling request that carries drafts (#677). Temperature-0 text is
identical to spark-ds41f-1025's except multi-chunk prompts whose final chunk has 1-128 rows
(0526). Against spark-ds41f-1025 in candidate 12's release comparison (one session, verdicts
against the repeat; candidates 13 and 14 keep its digests and kernels): decode grid -4.62%; c16
+2.05%, c8 +2.02%, one stream +3.61% (paired decode); prefill +8.4% / +8.8% / +9.9% / +11.4% /
+12.2% at 32K / 131K / 262K / 524K / 1M; 131K prefill beside 15 streams +8.2%, beside 3 and 11
the same; a short request's first token beside a 131K prefill 3.6-3.7x sooner; Poisson the same;
sampled c16 +1.87%, c8 +2.77%, c1 +2.68%; greedy 256/256 identical, teacher-forced KL 4.2e-8.
`tp4-ring`'s `benchmark_reference` is
[manifests/suites/2026-10-30-spark-ds41f-1030-tp4](../manifests/suites/2026-10-30-spark-ds41f-1030-tp4/)
(candidate 12's first release arm). On the triangle (qualified 2026-10-30 at the 6,144 budget,
run `candidate-14-release-tp3`): decode grid -5.99%; paired decode c8 +2.29%, c4 +2.68%, one
stream +3.76%; prefill +4.4% / +4.1% / +5.3% / +7.0% at 32K / 131K / 262K / 500K; 131K prefill
beside 7 streams +3.0%, beside 3 and 6 with short requests waiting -4.6% and -3.7% (the yield
and the 6,144 budget; those requests' first token p50 16.6 to 2.0 s and 18.3 to 9.6 s); sampled
c16 +2.76%, c8 +2.41%, c1 +2.94%; greedy 256/256 identical, KL 4.2e-8. `tp3-triangle`'s
`benchmark_reference` is
[manifests/suites/2026-10-30-spark-ds41f-1030-tp3](../manifests/suites/2026-10-30-spark-ds41f-1030-tp3/).
spark-ds41f-1025's recipe and lock, the rollback, are frozen in
[releases/spark-ds41f-1025](../releases/spark-ds41f-1025/) (image
`spark-ds41f:19f2c20ed4d6-next-6`, still on every node).

Before spark-ds41f-1030 production was **spark-ds41f-1025**, serving
DeepSeek-V4.1-Flash-1017 at 1,048,576 tokens. From 2026-10-26 21:46 CET it ran the
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
rollback.

## Active settings

<!-- BEGIN GENERATED: active-settings -->
| Setting | Active value |
| --- | --- |
| Promoted baseline | `2026-10-30-spark-ds41f-1030` (`spark-ds41f-1030`, frozen in `releases/spark-ds41f-1030/`); default shape `tp4-ring` |
| Served models | `deepseek-v4.1-flash` (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10`) and per-request GLP-steered `deepseek-v4.1-flash-weightless` (`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75`, revision `2715ffd5`) |
| Kernel backend | TileLang only (`kernel_backend: tilelang`, vLLM 0493-0496): TileLang 0.1.15 (`222b895d`), DeepSeek TileKernels 2.0.0 (`cb2fa41c`), sparknet 0.5.0 (`f997a594`) one-shot RoCE collectives; every serving kernel is TileLang including the vision tower, CED row selection, RoPE, compressor, KV writers, Engram gate/hash/gather, sparse MLA, WO projection, drafter context-KV, sampling and block verification |
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `19f2c20e` + `experiments/2026-10-26-spark-ds41f-candidate/rc14/vllm/series` (452 patches, patch head `b6de07aa`, tree `ea1281f8`), TileLang 0.1.15 (4 patches), TileKernels 2.0.0 (1 patch), sparknet 0.5.0, NCCL 2.30.7 (4 patches, tree `6af10aa7`); direct checkpoint loader (`--load-format direct`, vLLM 0499, `VLLM_PLUGINS` empty) |
| Image | `spark-ds41f:1030` (built as `spark-ds41f:19f2c20ed4d6-next-14`, `sha256:7918fd6145b5`), one digest on all ranks, built by `bin/spark build` |
| Decode tiles & PDL | block-32 FP8 decode rows on 16-, 32- or 64-row tiles by row count and split-K tiles up to 128 rows for 16-stream steps, programmatic dependent launch (PDL) along the decode chain (vLLM 0520-0525), attention front in one launch (0543-0544), 64-row prefill MoE tiles and panel rasterization past L2 (0043) |
| Batch invariance | temperature-0 outputs do not depend on the batch: rank-order FP32 reductions with one rounding, compressor projection one TileLang arithmetic at every row count, aligned prefill chunks (`--long-prefill-token-threshold` 16,288 on `tp4-ring`, 6,096 on `tp3-triangle`, 4,048 on `tp3-triangle-4k`), CED replays a prompt's last 128 rows wherever its chunks end (0526-0527), and mixed steps run sparse MLA per request |
| Hosts | DGX Spark 26.09.2 (dgx-spark-ota-update-meta), kernel `7.0.0-1019-nvidia-64k with kho=off`, driver 580.178.04, `multi-user.target (no desktop)`, `--cpuset-cpus=5-9,15-19`, glibc malloc pinned (`MALLOC_ARENA_MAX=2`, `MALLOC_MMAP_THRESHOLD_=1048576`) |
| Tensor parallel ranks & transport | `tp4-ring`: 4 ranks (`oneshot-ring4`); `tp3-triangle`: 3 ranks (`oneshot-direct`); `tp3-triangle-4k`: 3 ranks (`oneshot-direct`) |
| Maximum model length | `tp4-ring` and `tp3-triangle`: 1,048,576 tokens; `tp3-triangle-4k`: 262,144 tokens |
| Maximum sequences | `tp4-ring`: 16; `tp3-triangle` and `tp3-triangle-4k`: 8 |
| Maximum parallel prefills | 1 (`--max-parallel-prefills 1`) |
| Batched-token budget | `tp4-ring`: 16,384; `tp3-triangle`: 6,144 (qualified on `spark-ds41f-1030`); `tp3-triangle-4k`: 4,096 |
| Prefill yield & late admission | prefill yield enabled (`--prefill-yield short`, `--prefill-yield-limit-s 10`, vLLM 0545) so waiting short requests take the next step from a long prefill; late admission at decoder layer 30 (`VLLM_LATE_ADMISSION_LAYER=30`, vLLM 0222, 0226) |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers with non-final row skip (vLLM 0014-0018, 0229-0255) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025), ties broken by lowest logical position in TileLang |
| Explicit KV memory | `tp4-ring`: 10.5 GiB (11,274,289,152 B) per rank; `tp3-triangle`: 4.9062 GiB (5,268,045,824 B) per rank; `tp3-triangle-4k`: 2.2 GiB (2,362,232,012 B) per rank; no per-allocation KV zeroing (vLLM 0502-0503) |
| KV capacity | `tp4-ring`: 11,554,122 startup tokens (11.02x; holds 11 full 1M contexts or 16 x 718,848); `tp3-triangle`: 5,488,353 startup tokens (5.23x; holds 5 full 1M contexts or 8 x 682,240); `tp3-triangle-4k`: holds 8 full 256K contexts ([KV capacity](memory-profiles.md#kv-capacity)) |
| Expert scale codes | routed experts' UE8M0 weight scales as a per-row base exponent and 4-bit deltas (vLLM 0182-0183), with 2-bit deltas for narrow experts stored in the 4-bit codes' buffer (0563-0564, 0566; model loading 71.67 GiB a rank at TP4) |
| Memory-saver | signed DKMS 0.4.1 (`dgx-spark-memory-saver`) on all nodes; loaded UVM leaf-table packing enabled (`uvm_leaf_packing: Y`) |
| Page-size profiles | 64 KiB selected (`tp4-ring`, `tp3-triangle`); 4 KiB (`tp3-triangle-4k`) retains 2.2 GiB KV and a 262,144-token limit |
| Output head & display carve-out | exact 12-bit packed BF16 (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`), bit-identical to the checkpoint, with FP32 logits (`VLLM_DS41_HEAD_DTYPE=float32`); kept in regular memory on `tp4-ring` (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0`) and in the GB10 display carve-out on `tp3-triangle` (`=1`) |
| Reference precision | DeepSeek's `inference/model.py` term for term: FP32 logits, exact FP32 mHC stream projection (three TF32 parts per weight), shared expert SwiGLU in FP32 with one rounding, MoE partials rounded once per rank, and WO-A's input through `act_quant` (E4M3, UE8M0 per 32, vLLM 0177, 0179) |
| Image input | TileLang vision tower loaded (`--limit-mm-per-prompt '{"image":4}'`, `VLLM_MAX_IMAGE_PIXELS=16777216`, `--mm-processor-cache-gb 0`; [below](#images-per-request)) |
| DSpark speculative decoding | 5 draft tokens, draft TP = node count, fixed linear step cost budget (`adaptive_verification_row_cost: 0.035`, vLLM 0184, nothing timed at boot), dead verification rows below survival 0.2 (`0504`), block rejection with fresh draws per round (`0228`), sink draft-zero rejection on by default (`0221, 0223`), vocabulary-parallel greedy drafts and shard argmax verification (`0112-0114`) |
| Collectives | sparknet 0.5.0 one-shot RoCE all-reduce (up to 2 MiB) and all-gather (`oneshot-ring4` on TP4, `oneshot-direct` on TP3), TP broadcast on one-shot all-gather (`0185`), GPU-clock timeout 10 s (`SPARKNET_ROCE_TIMEOUT_MS=10000`), no fallback (`0142`) |
| CUDA graphs & warmup | `FULL_AND_PIECEWISE`, capture sizes 1-96 on `tp4-ring` (1-48 on `tp3-triangle`); all declared TileLang calls prepared and cache frozen before memory profiling (`--jit-monitor-mode off`) |
| Engram projection & rows | sharded across ranks (`projection_tp`); one native reader thread reads both tables in layer order from interleaved whole hash columns; host hashes prefill tokens a step early except drafted-prefill requests, which the GPU hashes (vLLM 0145-0154, 0180, 0567) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s), each killing on its first sample below; hosts set `vm.watermark_boost_factor=0` |
| Prefix caching & API semantics | 256-token blocks (`--enable-prefix-caching`, `--enable-prompt-tokens-details`); thinking enabled by default, top-level `thinking` switch, effort aliases, `response_format`, `tool_choice: none`, assistant prefix, and generation defaults `temperature: 1.0, top_p: 1.0` |
<!-- END GENERATED: active-settings -->

The historical release comparison of r6f against r6d is in the
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
