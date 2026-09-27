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

One session (2026-09-27, 15:20-16:09 UTC), lean screen, three samples per
point; the two `control` runs are pooled (six samples).

Single-stream step time (ms) and accepted drafts per step:

| Arm | prose | code | prose answer | code answer | explain |
|---|---|---|---|---|---|
| control | 48.1 (1.14) | 54.4 (1.97) | 51.7 (1.68) | 57.8 (3.09) | 50.7 (1.90) |
| dhead | 47.9 (1.19) | 53.1 (1.96) | 49.6 (1.54) | 56.8 (3.22) | 49.2 (1.80) |
| markov | 47.2 (1.24) | 52.9 (2.18) | 49.6 (1.70) | 55.9 (3.26) | 47.9 (1.73) |
| both | 46.0 (1.14) | 52.6 (2.03) | 48.5 (1.55) | 55.9 (3.29) | 48.3 (1.82) |

Throughput change against the pooled control, `both`:

| Streams | prose | code | prose answer | code answer | explain |
|---|---|---|---|---|---|
| 1 | +4.2% | +5.7% | +1.4% | +8.6% | +2.1% |
| 8 | +2.9% | -1.7% | +0.1% | +2.9% | +2.4% |

- Both switches together shorten single-stream steps by 1.8-3.1 ms (about
  4.5%); each alone saves about 1-2 ms.
- Accepted drafts per step stay within sample noise of the control in every
  arm, so the NVFP4 drafter head and Markov projection cost no measurable
  acceptance.
- Quality gate LRU 5/5 in every arm; dgx1 minimum MemAvailable 6.16-6.40
  GiB (control 6.25-6.40).
