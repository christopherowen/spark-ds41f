# Exact drafter savings: vocabulary-parallel greedy drafts and main_proj

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5e`). The arms
run its image with the runtime files of vLLM patches 0012 and 0013 (branch
`spark3/r5f`) mounted over the vLLM tree (`overlay.sh`); a promotion builds
the image.

## Question

A single-stream step spends several milliseconds in the drafter:

- The full-vocabulary draft logits are all-gathered.
- Every rank reads the replicated 66 MB Markov projection five times (about
  330 MB per step).
- Every rank reads the replicated 79 MB `main_proj`.

Patch 0012 keeps greedy drafting sharded by vocabulary and reduces only
(max, index) pairs across ranks. Patch 0013 column-shards `main_proj`. Both
keep the drafts and the output unchanged. How much of the step do they save?

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none (no overlay) |
| `vp` | overlay, `SPARK3_DSPARK_VOCAB_PARALLEL=1` |
| `vpmp` | overlay, `SPARK3_DSPARK_VOCAB_PARALLEL=1`, `SPARK3_DSPARK_MAIN_PROJ_TP=1` |
| `combo` | as `vpmp`, plus `VLLM_DS41_DRAFT_NVFP4_HEAD=1` and `VLLM_DS41_MARKOV_NVFP4=1` (vocabulary-parallel drafts over an NVFP4 Markov shard) |

## Workload and gates

`test_overlay.sh` runs the patch tests on dgx3 first. `sequence.sh` runs a
control, vp, vpmp, combo and a second control, each with the LRU gate and the lean decode screen
(reasoning and answer cases plus `explain`, one and eight streams, three
samples). Decide on single-stream step time and accepted drafts per step;
the drafts should be unchanged, so acceptance should match the control.

## Results

The first attempt (16:13 UTC) failed at startup: 0012 checked the draft
head's partition in `process_weights_after_loading`, before the loader binds
the shared head. 0012 now checks on first use, and also admits the NVFP4
Markov projection for the `combo` arm.

One session (2026-09-27, 16:20-17:10 UTC), lean screen, three samples per
point; the two `control` runs are pooled (six samples).

Single-stream step time (ms) and accepted drafts per step:

| Arm | prose | code | prose answer | code answer |
|---|---|---|---|---|
| control | 49.6 (1.35) | 55.5 (2.23) | 50.9 (1.50) | 58.2 (3.28) |
| vp | 46.0 (1.19) | 53.1 (2.03) | 49.6 (1.68) | 55.5 (3.11) |
| vpmp | 46.2 (1.11) | 53.2 (2.22) | 49.2 (1.67) | 56.3 (3.25) |
| combo | 45.8 (1.32) | 52.0 (2.21) | 47.7 (1.56) | 54.5 (3.26) |

Throughput change against the pooled control, `combo`:

| Streams | prose | code | prose answer | code answer | explain |
|---|---|---|---|---|---|
| 1 | +7.2% | +6.5% | +9.0% | +6.2% | +9.9% |
| 8 | +4.6% | +3.4% | +6.3% | +4.0% | +0.2% |

- Vocabulary-parallel drafts alone (0012, exact) shorten single-stream steps
  by 1.3-3.6 ms; the drafts are unchanged by construction.
- The column-sharded `main_proj` (0013) is within noise at three samples
  (its expected saving is about 0.25 ms).
- `combo` (0012 and 0013 over an NVFP4 Markov shard, with the NVFP4 drafter
  head) shortens single-stream steps by 3.2-3.8 ms, about 7%, and is
  positive at every point: +6-10% at one stream, +0-6% at eight.
- Accepted drafts per step stay within sample noise of the control.
- Quality gate LRU 5/5 in every arm; dgx1 minimum MemAvailable 6.21-6.41
  GiB.
