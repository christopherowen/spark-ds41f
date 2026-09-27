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

Pending.
