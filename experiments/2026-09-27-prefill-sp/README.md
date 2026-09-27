# Prefill sequence parallelism

Base: the promoted configuration. The `sp` arm mounts the files of two vLLM
commits (branch `spark3/r5f-sp`, on the r5f series) over the promoted
image's vLLM tree (`overlay.sh`).

## Mechanism

For prefill forwards of at least `SPARK3_DS41_PREFILL_SP_MIN_ROWS` tokens,
each decoder layer's all-reduces (after WO, after the MoE) become dim-0
reduce-scatters. Between them each rank runs the row-wise work (mHC mixes,
norms, the Engram gate, the residual) on its third of the rows. Attention
and the MoE all-gather the rows first. Weights, decode and CUDA-graph
decode are unchanged, and the threshold is raised above every decode token
count. Output differs from the all-reduce path only by the summation order
of the reduce-scatter (ULP-level).

## Question

A 4K real-text chunk spends about 150 ms in hyper-connection kernels that
every rank repeats on every row. A TP4 SGLang stack measured +15% real-text
prefill from sequence parallelism. How much does it gain at TP3, and is the
output still correct on long prompts?

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none |
| `sp` | overlay, `SPARK3_DS41_PREFILL_SP_MIN_ROWS=2048` |

## Workload and gates

`test_overlay.sh` runs the SP unit tests and a CPU simulation of the model
forward in the image on dgx3. `sequence.sh` runs control, sp and control
again. Each arm runs the LRU gate, single-stream decode (answers and
`explain`, to confirm decode is untouched), cold prefill of real text at
about 4K-64K tokens, and `sp_check.py`:

- a ~6K-token needle question, which must be answered correctly;
- prompt logprobs of a ~5K-token prompt, twice.

`compare_logprobs.py` compares prompt logprobs across arms against each
arm's own run-to-run spread.

## Results

- First attempt (18:26 UTC): no change in prefill. The profile showed SP
  never engaged: every DeepSeek V4.1 prefill uses CED compaction, which the
  first version excluded. With CED only the encoder layers before
  `ced_decoder_start` run on every row, so a third commit runs SP through
  them and gathers the carried rows before the boundary layer.
- CED-aware SP (19:03 UTC, same-session control): +7.8% to +8.4%.
  Per 4K chunk, hyper-connection time fell from 152 to 57 ms. The
  reduce-scatters and all-gathers each cost about half an all-reduce, so
  communication was neutral: kernel time went from 1134 to 1043 ms.
- With the local attention front (fourth commit, 19:20 UTC):

| Arm | 4K (3854) | 16K (14768) | 32K (28937) | 64K (57122) |
|---|---|---|---|---|
| control | 3585 | 3460 | 3439 | 3421 |
| sp | 3912 | 3799 | 3752 | 3744 |
| change | +9.1% | +9.8% | +9.1% | +9.4% |

- Correctness: the ~5.7K-token needle is answered correctly in every arm.
  Prompt logprobs of a ~4.2K-token prompt differ from the control by 0.011
  on average (median 0.00000), no more than the control differs from
  itself between identical runs (0.012-0.014).
- Single-stream decode step times are unchanged. LRU 5/5 in every arm;
  dgx1 minimum MemAvailable 6.25-6.62 GiB.
- The unit tests pass. The upstream CED model test fails two cases on the
  unmodified image too (its fake model lacks `_l2pf_ready`).
