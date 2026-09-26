# vLLM patch stack

Base: `local-inference-lab/vllm@04c30fa98e7917fee0a24c739ea503ce1e22538d`
(`integration/karmic-kraken-beta`).

- `0001-engram-projection-tp-padding.patch` lets `engram_config.projection_tp`
  shard DeepSeek V4.1's Engram WKV projection when its 25600-row output does
  not divide by the TP size (TP3): the output is padded to whole 32-row
  block-FP8 scale blocks per rank, the last rank's missing checkpoint rows are
  zero-filled through `allow_tp_padding`, and the gathered output is sliced
  back. On the r2 base it yielded tree `90fdd043`. Evidence:
  `experiments/2026-09-24-improvement-leads/`. Upstream status: candidate for
  Local Inference Lab, not submitted.

- The former `0002-engram-async-disk-rows.patch` is not carried on this
  base: its Engram overlap (`VLLM_DS41_ENGRAM_OVERLAP`, on by default) reads
  both disk tables on a worker thread with B12X `run_lookups` and gates the
  Engram layers in the graph. The r4 series (base `01f1b874`) still carries
  0002; see `manifests/sources/2026-09-26-r4-candidate-source.json`.

- `0003-dsml-optional-string-attribute.patch` keeps DSML tool parameters
  that omit `string="true|false"`; the V4 and V4.1 parsers dropped them
  silently. A missing attribute is treated like `string="false"` (JSON,
  falling back to the literal). Port of vllm-project/vllm#56271 (merged as
  `39e33db7`), Python parts only. Quality: tool-call arguments only. Tests:
  the PR's parser tests (128 pass in the r3 image).

- `0004-mm-block-hash-window.patch` hashes every multimodal item that
  overlaps a block completed by generated tokens; the incremental hasher
  started at the last item only, so a block holding two images hashed as if
  it held one. That lost the next turn's prefix-cache hit and let two
  conversations that differ only in an earlier image of that block share
  its KV. Port of vllm-project/vllm#51694 (merged as `e6c07ea5`). Tests: the
  PR's test fails on r3 and passes with the patch.

- `0005-dspark-pinned-cost-curves.patch` (`SPARK3_DSPARK_COST_DIR`) saves TP
  rank 0's adaptive-verification cost curves for a shape set and reuses them
  on later boots with the same shapes, so verified draft counts no longer
  follow one boot's startup timing. `SPARK3_DSPARK_PROFILE_REPLAYS` raises
  the replay count of the pinned profile. Every rank still profiles, so
  collectives stay matched. Output unchanged. Upstream status: candidate
  for Local Inference Lab, not submitted.

- `0006-dspark-marginal-verification-rule.patch`
  (`SPARK3_DSPARK_VERIFY_RULE=marginal`) chooses the draft budget that
  maximizes expected tokens minus the achieved rate times modeled cost,
  instead of each step's expected tokens per millisecond. The rate is an
  exponential average per request count, floored at the no-draft rate, and
  derived identically on every rank. Output unchanged: the target verifies
  every kept draft. Upstream status: candidate, not submitted.

- `0007-dspark-v41-gathered-markov-bias.patch` admits the V4.1 drafter to
  the existing gathered Markov-bias path (`dspark_draft_topk` in the
  speculative config): the bias is added to the top-k base candidates only,
  instead of projecting onto the whole vocabulary at every draft position.
  Only drafts can change, so acceptance can move but output cannot. Off
  unless `dspark_draft_topk` is set. Upstream status: candidate, not
  submitted.

- `0008-dspark-draft-trace.patch` (`SPARK3_DSPARK_TRACE=<path>`) makes TP
  rank 0 append each step's sampled tokens, verified draft counts, drafts
  and raw confidences as JSON lines, for pricing verification policies
  offline at temperature 0. It copies tensors to the host every step: replay
  runs only. Output unchanged.

Applying the series to the base yields patch head `ba8193a7` and tree
`f8b044c5` (on the r4 base `01f1b874`, with 0002: `0ccd0f62`, tree
`0debe853`).
