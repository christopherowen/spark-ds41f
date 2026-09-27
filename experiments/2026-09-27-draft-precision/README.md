# Drafter precision: NVFP4 vocabulary head and Markov projection

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5e`).

## Question

A single-stream step spends about 1.9 ms in the drafter's bf16 vocabulary
head (it shares the target's 442 MB per rank) and about 1.6 ms reading the
replicated bf16 Markov projection (`markov_w2`, 66 MB) once per draft
position. Both only shape the drafts; the target verifies every committed
token, so output is unchanged. The base already carries NVFP4 versions of
both, off by default. Do they shorten steps without costing acceptance?

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | none |
| `dhead` | `VLLM_DS41_DRAFT_NVFP4_HEAD=1` (drafter-owned NVFP4 head, +124 MB per rank) |
| `markov` | `VLLM_DS41_MARKOV_NVFP4=1` |
| `both` | both switches |

Each arm has its own compile cache.

## Workload and gates

`sequence.sh`: control, dhead, markov, both, control again in one session,
each with the LRU gate and the lean decode screen (reasoning and answer
cases plus `explain`, at one and eight streams, three samples). Decide on
single-stream step time and accepted drafts per step. Memory guards
unchanged.

## Results

Pending.
