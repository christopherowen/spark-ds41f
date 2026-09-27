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

## Workload and gates

`test_overlay.sh` runs the patch tests on dgx3 first. `sequence.sh` runs vp,
vpmp and a control, each with the LRU gate and the lean decode screen
(reasoning and answer cases plus `explain`, one and eight streams, three
samples). Decide on single-stream step time and accepted drafts per step;
the drafts should be unchanged, so acceptance should match the control.

## Results

Pending.
