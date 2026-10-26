# spark-ds41f-1025: release notes (draft)

spark-ds41f-1025 is a release of this deployment, named after the day it is promoted. It is not
a model version: the model is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint, with native
weights. (A forum reader found the date names confusing next to DeepSeek's own; the release
name is the project and its date, the checkpoint keeps DeepSeek's name.)

It is candidate 6: spark-ds41f-1022 with candidate 4's changes (the spark-ds41f-1024 draft,
#503, never promoted) and the TileLang completion (candidate 5), plus two test fixes that
candidate 5's release tests asked for (vLLM 0489-0490; package files unchanged). Image
`spark-ds41f:19f2c20ed4d6-next-6`, vLLM patch head `11171760`, tree `6f8e062f`. Four-node ring
(tp4-ring) only.

## Results (2026-10-25, the ring)

| Check | Result |
| --- | --- |
| Release tests | candidate 5: 5,724 passed, 160 skipped (each with its reason), 6 failed in two stale test files; candidate 6 fixes them (its chunk ahead of the image: 336 passed, 0 failed). Candidate 6's own image: 5,730 passed, 160 skipped, 0 failed. |
| Temperature-0 outputs | identical to spark-ds41f-1022's under concurrency, staggered and mixed traffic, prefix caching and repeats: `3eab5112` / `db25fde3` with log probabilities, `c9cf9dbe` / `c6bb4f4e` as text |
| Generation controls | no failure; logprob ids and prompt logprobs succeed alone and at eight streams; control spellings stay text in all eight places |
| Images | answers differ from spark-ds41f-1022's tower and score the same on 2,283 public benchmark items (+0.43 points for 1022, interval [-0.14, +1.02]) |
| Sink probes | `Say hi.`, `Thanks!`, `ok` corrected, every rank agreeing; zero flags on ordinary text |
| Startup | API ready in 87-89 s (spark-ds41f-1022: 87-94 s); 247 TileLang calls prepared in 11 s a rank; nothing compiles after; no B12X preparation |
| Memory | lowest MemAvailable 20.8 GiB (dgx1), against 18.9 for candidate 4 at the same budget |
| Seeded sampled requests | reproducible alone; under concurrency with adaptive verification they depend on the batch, as in spark-ds41f-1022 (an existing limitation; see below) |
| KV capacity | vLLM's startup figure: 6,512,700 tokens, 6.21 full 1M contexts (the 16,384 budget's; 8.18 at 8,192). The figure charges every request the unsettled batches; the corrected count (#586, next release) is 11.02 full 1M contexts at 16,384 |
| Speed against spark-ds41f-1022 (one session, verdicts against the repeat) | prefill +11.8% to +13.9% from 32K to 1M; decode grid -2.38% (better); c16, c8, Poisson and one stream the same (+0.1%, +0.9%, +0.0%, +2.1%); sampled the same; greedy 256/256 identical, KL 0 |

## Results (2026-10-25, the triangle)

The owner re-cabled to the dgx1-dgx2-dgx3 triangle the same evening. tp3-triangle ships with an
8,192-token prefill budget (spark-ds41f-1022: 4,096); the fit check passed with dgx1 at 6.67 GiB
lowest through three cold 1M prefills. Details are in the
[release record](README.md#the-triangles-results-2026-10-25).

| Check | Result |
| --- | --- |
| Temperature-0 outputs | identical to spark-ds41f-1022's on the triangle in all six scenarios: `62313cb5` / `9742ac2c` with log probabilities, `4d143977` / `12fd8257` as text |
| Controls, logprobs, images, sink probes | as on the ring: no failures, one digest alone and under load, `Say hi.` / `Thanks!` / `ok` corrected on every rank |
| Startup | API ready in 103-111 s (first boots, compiles included); 246 TileLang calls in 11.4 s a rank; no B12X or JIT lines |
| Memory serving | dgx1 / dgx2 / dgx3 at 7.36 / 8.52 / 8.52 GiB (1022: 6.54 / 7.93 / 8.14) |
| KV capacity | vLLM's startup figure: 4,009,281 tokens (3.82x at 1M); the pool still holds 5 full 1M contexts (#586's count) |
| Speed against spark-ds41f-1022 (one session, two arms) | prefill +26.1% / +28.2% / +26.8% / +25.5% at 32K / 131K / 262K / 500K, and +22.5% at 1M (3,952 against 3,226 tok/s, two runs the same evening); decode grid -1.65% (better); c8, c4, Poisson and one stream the same (+0.2%, +0.6%, level, +1.9%; 60.5 tok/s at one stream); greedy 256/256 identical, KL 0 |

## What changes for users

- **Fixed: one request with `prompt_logprobs` killed the engine.** Under sink rejection (on by
  default since spark-ds41f-1022) the scheduler's partial-prefill assertion took EngineCore
  down (HTTP 500, then every request failed). The prompt's logprobs now go out with the
  request (vLLM 0266, #546; experiments/2026-10-24-prompt-logprobs-sink). spark-ds41f-1022 is
  affected.
- **Fixed: `logprob_token_ids` returned HTTP 500** under speculative decoding: verified tokens
  came back without their logprobs (vLLM 0267, #551; experiments/2026-10-25-logprob-token-ids).
  spark-ds41f-1022 is affected; the engine kept serving.
- **Every serving kernel is TileLang** except NCCL and a few rare-path ATen copies. No B12X
  preparation runs; the runner warm-up compiles no Triton. B12X keeps the checkpoint loader;
  three load-time Triton kernels pack the BF16 LM head.
- **Faster prefill:** the CED row selection in TileLang and its nonfinal row skip, with the
  ring's long-prefill budget at 16,384.
- **Sampling fix:** fresh random draws in each block-verification round (vLLM 0228, a backport
  of vllm-project/vllm#61161). Temperature 0 is unchanged; sampled text changes by design.
- **Generation controls:** V4.1's tokenizer honours `add_generation_prompt` and
  `continue_final_message`, and the DeepSeek API's top-level `thinking` switch works (#447).
- **Prompt-cache report:** every response's `usage.prompt_tokens_details` carries
  `cached_tokens` (#455, #469).
- **Token ranks** exclude the vocabulary padding: a selected token with logit `-inf` now ranks
  129,280 at most, not 131,072 (#504). Finite ranks are unchanged.
- **GLP steering** (#401, msuiche's, reworked) is available and off unless configured; off, it
  launches nothing.
- **glibc malloc pinned** in every process (#439's memory growth on long soaks).
- **The sink log states its scope:** each rank's line counts flags in executed steps;
  `vllm:sink_*` counts committed outputs (#489).
- **The live names move to spark-ds41f:** container, image repository, labels,
  `/opt/spark-ds41f`, `/cache/spark-ds41f`, `SPARK_DS41F_*` variables and the
  `spark-ds41f-kernel-memory` service. Sites with their own overlays must rename their
  `SPARK3_*` settings: the new image reads only `SPARK_DS41F_*`.
- **The JIT monitor is off** (`--jit-monitor-mode off`): under TileLang it had nothing left to
  report, and the frozen TileLang cache refuses an unprepared kernel by name.
- **21 recipe settings for kernels serving does not run are removed** (B12X and CuTe DSL
  compiler settings, the FlashInfer sampler and workspace, torch.compile's AOT and Inductor,
  unread attention settings, NVFP4 head defaults, the CUDA-graph memory estimate).

## Known limitations

- **At most 32 top logprobs** (a compatibility change the owner accepted). The TileLang top-k
  ids kernel takes k up to 32, so `--max-logprobs` above 32, or -1 (the whole vocabulary), is
  refused at startup. Before, any k up to the vocabulary was served through `torch.topk`. The
  recipe runs vLLM's default of 20, and the OpenAI API caps `top_logprobs` at 20, so no
  request through the recipe changes.
- **The DSpark drafter's argmax trails Triton's** by about 0.05 µs at 1-16 rows (0.86 µs at one
  stream), once a decode step on the drafter's last step: about 0.0001% of a step. Accepted by
  the owner; the sampling benchmark reports it without gating it.
- **Image outputs differ from spark-ds41f-1022's.** The TileLang vision tower follows
  DeepSeek's reference arithmetic where B12X's tower did not, so answers about images are not
  bit-identical to the previous release (text-only outputs are). On 2,283 public benchmark
  items (OCRBench, DocVQA, ChartQA, TextVQA, RealWorldQA) the two score the same.
  The built image against spark-ds41f-1022 (run `vision-eval-candidate5`): 83.36 against 83.79
  overall, a difference of +0.43 points for 1022 with a 95% interval of [-0.14, +1.02]; no
  benchmark's interval excludes zero; 58 of 2,283 items score differently (32 / 26).
- **Seeded sampled requests are reproducible alone, not under concurrency** (as in
  spark-ds41f-1022). Adaptive verification sizes each step's drafts over the whole batch, so
  which requests share a request's steps decides how its positions fall into verified blocks,
  and block verification's sampled tokens depend on that split. The same seeded request can
  therefore return different text when others run beside it. Temperature 0 is unaffected. On
  spark-ds41f-1022, 6-7 of 8 seeded requests sent eight at once differed between runs; in
  this release, 6-8 of 8. This release's sampled text also differs from spark-ds41f-1022's by
  design (0228's fresh draws per block-verification round).
- **vLLM's startup figure for the ring reads 6.21 full 1M contexts** (6,512,700 KV tokens;
  8,580,566 at the 8,192 budget), from the same 10.5 GiB pool, which holds 11 full 1M contexts
  at either budget ([KV capacity](../../docs/memory-profiles.md#kv-capacity), #586). The owner
  keeps the 16,384 budget for its prefill (2026-10-25) and notes that the KV-capacity
  arithmetic in earlier records is wrong. The figure's drop comes from the ring's 16,384-token prefill budget (the
  owner's "do 16" of 2026-10-23, for the CED skip's +5-6% prefill), not the TileLang work:
  every saved boot at 16,384 reports 6,512,700, candidate 4's image included (run
  `tilelang-ports-controls`), and every boot at 8,192 reports 8,580,566. vLLM charges each
  request, at admission as well as in this figure, its sliding-window caches' worst case: the
  128-token window plus two whole prefill chunks in flight (`max_in_flight_tokens`), and the
  chunk doubled. The pool and its full-attention blocks are unchanged. (spark-ds41f-1024's
  draft recorded 8.18 for candidate 4; that memory record's log lines predate its image.)
- **The 4 KiB-page triangle is not qualified** on this image (tp3-triangle-4k keeps its 4,096
  budget); `cluster start` refuses it without `--unqualified-shape`. The ring and the triangle
  both are.
- **The triangle's comparison ran two arms** (the owner's shortcut; the same build's repeat was
  measured on the ring). Sampled c8 and c1 exist for this release only. The triangle's 1M figure
  comes from two runs the same evening, not one session; release-tp3 version 2 (#600) prefills
  to 1M from the next release on.
- **The triangle's stall:** at 8,192 the other streams' decode step is about 1.7 times as long
  while a long prompt prefills (the ring's 16,384: +78%).

## Reports triaged for this release

Every open issue and pull request of spark-ds41f and sparknet, the recently closed issues
and merged pull requests, their comments, and the NVIDIA developer forum thread (topic
384438, 66 posts), read on 2026-10-25. **In** means this release addresses it: already fixed
in candidate 5, or a fix this release must carry. **Not a blocker** says why. **Owner** needs
a decision only the owner can make.

| Report | Decision | Why, and the evidence |
| --- | --- | --- |
| The ring's KV capacity at the 16,384 budget (found in this release's checks) | **Decided** (owner, 2026-10-25): keep 16,384 | vLLM's startup figure for the ring reads 6.21 full 1M contexts at the 16,384 budget (8.18 at 8,192); the pool holds 11 at either budget (#586). The owner keeps 16,384 for its prefill and notes that the KV-capacity arithmetic used so far is wrong; the earlier 8.18 figure is not a capacity this release loses. Not a regression of the candidate's code. |
| Candidate 5's release tests: 6 failures in two stale test files (found in this release's checks) | **In** (fixed in candidate 6) | vLLM 0489-0490, test files only. |
| #455 `prompt_tokens_details` absent (JordiPosthumus) | **In** (fixed) | #469's `--enable-prompt-tokens-details` is in the candidate's recipe and replayed on the promotion branch. Live on candidate 5 (run `candidate-5-recheck-tp4`): a 2,710-token request reports `prompt_tokens_details` with `cached_tokens` 0 (2,560 created), its repeat `cached_tokens` 2,560. Close after the deploy. |
| #469 the prompt-cache report (PR) | **In** | Replayed on the promotion branch; supersedes the PR. Close after the deploy. |
| #447 top-level `thinking: {"type": "disabled"}` ignored (Acermax) | **In** (fixed) | vLLM 0256-0257 (candidate 4's) are in the series. Live on candidate 5 (run `candidate-5-tp4`, controls probe): top-level `thinking: {"type": "disabled"}` renders the 6-token chat prompt of `chat_template_kwargs.thinking=false`, alone and beside `reasoning_effort: high` (the default renders 32), and `{"type": "maybe"}` is refused with 400. Close after the deploy. |
| #489 sink metrics against the rank log (JordiPosthumus) | **In** (fixed) | The owner's answer stands (different scopes); vLLM 0264 (here 0336) makes the rank log line say so. Live on candidate 5: every rank logs "N flagged rows in executed steps (draft zero, drafter, prompt; vllm:sink_* counts committed outputs only)", the four ranks agreeing. Close after the deploy, as promised in the issue. |
| #401 GLP steering (msuiche) | **In**, off by default | vLLM 0258, 0261, 0263, 0265 (here 0330-0337): one TileLang projection a layer, launched only when a vector is configured. Off, the release's outputs equal spark-ds41f-1022's (the gate). On candidate 5 with a synthetic vector (run `candidate-5-glp-tp4`): every rank logs "GLP steering on: alpha 0.75, layers [1..39]" and prepares one more TileLang call (248), and the steered outputs repeat exactly in all six concurrency scenarios, with and without log probabilities. Close after the deploy, crediting msuiche. |
| #484 the rename to spark-ds41f (PR) | **In** | Replayed on the promotion branch (two of #549's lab tests adapted to the renamed recipe); the candidate's recipe and image carry the names. Close after the deploy. |
| #503 promote spark-ds41f-1024 (draft PR) | **Superseded** | Candidate 5 contains candidate 4 and replaces it. Close when this promotion merges. |
| #503 comments: TP3 triangle results on `-next-4` (JordiPosthumus) | **Not a blocker** | TP3 is out of this release's scope (the owner re-wires later); tp3-triangle stays unqualified on this image. Jordi's identical TP3 hashes and the malloc pinning's headroom are useful evidence for the TP3 release. |
| #503 comments: Jordi's open questions (is `rc-tp3.json` the arm for `-next-4` on TP3; what to watch on the triangle) and offers (a ports-only arm, the release-tests bundle) | **Deferred** (owner, 2026-10-25), now unblocked | No reply until TP3 is qualified on this release's image. It was qualified on 2026-10-25 at 8,192 (the triangle's results above), so the reply can go out with the release announcement. |
| #503 comments: `cluster start` refusal should mention `--unqualified-shape`; site patches make the checkout dirty | **Not a blocker**, follow-up | Tooling ergonomics in `bin/spark`, not the served image or recipe; the scope freeze excludes unrelated improvements. Worth a small PR after the release. |
| #503 comments: a 6,144-token chunk budget beats 4,096 on TP3 | **In**, at 8,192 | The triangle's qualification (2026-10-25) ships 8,192 (owner: "if 8k fits"); the fit check passed. |
| #509, #506, #504, #524 TileLang ports (draft PRs) | **In** (superseded) | Their patches are in candidate 5's series: 0345 (counts), 0351 and 0473 (verification insert), 0352-0353 (ranks, padding), 0381-0401 (controls). Close after the deploy. #509's "eight vocabulary-parallel draft-seed mismatches" do not reproduce: `tests/v1/spec_decode/test_dspark_vocab_parallel.py` passes in candidate 5's release tests. |
| #504's rank fix: a selected `-inf` token ranked 131,072 | **In** (fixed) | 0353 excludes the vocabulary padding (129,280 at most). A reporting change for forced or prompt scoring only. |
| Prompt logprobs kill the engine (#546; found in the lab) | **In** (fixed) | vLLM 0266 (here 0408). Production 1022 is affected. Candidate 5's prompt-logprobs scenario succeeds, alone and at eight streams alike. |
| `logprob_token_ids` HTTP 500 under speculative decoding (#551; found in the lab) | **In** (fixed) | vLLM 0267 (here 0437). Production 1022 is affected. Candidate 5's logprob-ids scenario succeeds, alone and at eight streams alike. |
| #439 the steady guard binds first at 1M on TP3 (docs PR, JordiPosthumus) | **Not a blocker** | Docs only; the release pins glibc malloc, the growth #439 measured. The memguards stay 5 GiB at startup and 3 GiB steady. The PR can merge on its own; it describes TP3. |
| #367 RoCE GIDs by address; sparknet #1 | **Not a blocker** | Needs its own qualification (sparknet API 3); the recipe still pins slot 3 and `doctor --live` checks it. |
| #574 AgentX (PR) | **Not a blocker** | Measurement tooling, off the critical path (owner). After this promotion its arm, which adds `--enable-prompt-tokens-details` to production, duplicates the recipe's flag and needs a rebase. |
| #488 image IDs on containerd (closed) | Done | Closed by #492; the image job compares layers (this build: one ID and 54 layers on dgx1-dgx4). |
| #446 control-token spellings (closed) | Carried | vLLM 0227 since spark-ds41f-1022. The control-spellings probe passes in all eight request locations on candidate 5. |
| #434 `max_num_seqs` capture mismatch (closed) | Done | Closed; no change here. |
| Forum #60, #66: release names look like model versions (stu.miller) | **In** (documentation) | These notes and the README say the release name is a date and the model is DeepSeek-V4.1-Flash-1017. The naming scheme is the owner's. |
| Forum #42, #46, #47: 74 tok/s not reproduced (nick398) | **Not a blocker** | Answered (#47): 74 tok/s is the ring's single-stream suite (temperature 0, thinking off, distinct prompts); this release publishes that suite's figure with its method. |
| Forum #49: how does it perform? (1teenarp) | **In** (documentation) | README Performance gets this release's comparison; raw runs under `manifests/suites/`. |
| Forum #16, #24, #25, #40: TP3 startup stalls on uneven JIT caches (huxiaofengtiger) | Fixed since r6u | Kernel preparation before memory profiling, a frozen TileLang cache and a liveness rendezvous. Candidate 5 prepares more kernels; every rank prepared all 247 calls in 11.0-11.1 s before profiling and nothing compiled after (no cache file written during the boot), so the mechanism holds for the new kernels. A new image's first boot compiles them once (24.4 s on the overlay). |
| Forum #48, #50, #52-#54: GB10 clock latch | **Not a blocker** | Platform firmware, not the release; `doctor --live` detects it. |
| Forum #57: disk-backed KV cache (vedcsolution) | **Not a blocker** | A feature request, not planned (#58). |
| Forum #19-#23: tool-eval-bench failures (vedcsolution) | **Not a blocker** | Model behaviour and evaluator misses; no engine fault in the traces. |
| Forum #10-#12: RoCEnante attribution (el8, lukea) | **In** (documentation) | sparknet's NOTICE credits Luke Alonso and the Local Inference Lab contributors; these notes keep the credit. |
| Forum #31, #33, #36: qualify tp3-triangle-4k (j0n) | **Not a blocker** | Not in this release's scope. |
| Forum #67: softmax rounding in the sampler (owner) | Ruled out | A few parts per million per step at most; nothing to change. |

## Thanks

- justincasey, for the CED decoder keeping one row on a prompt's non-final chunks (the CED
  skip) and the larger long-prefill budget, both from his four-Spark recipe. Together they are
  most of this release's prefill gain.
- msuiche, for GLP steering (#401), reworked into this release.
- JordiPosthumus, for:
  - the prompt-cache report (#455);
  - the sink scope (#489);
  - the memory growth behind the malloc pinning (#439);
  - the capture-size mismatch at other stream counts (#434);
  - the containerd image IDs (#488, fixed by #492);
  - the TP3 results on candidate 4, which shaped the triangle's 8,192 budget.
- @rob01 (forum), for the four-node results on spark-ds41f-1022.
- Acermax, for the `thinking` switch report (#447) and the control-token fix (#449).
- DeepSeek, for 1017, the reference implementation and TileKernels.
- Local Inference Lab, for the vLLM integration branch, B12X (still the checkpoint loader) and
  RoCEnante, from which sparknet's one-shot collectives derive (sparknet's NOTICE credits Luke
  Alonso and the Local Inference Lab contributors).
- The tile-ai team, for TileLang.
- Everyone posting results and reports in the forum thread.
