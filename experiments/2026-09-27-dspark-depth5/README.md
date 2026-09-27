# Five drafts with verification priced on real rows

Base: the promoted configuration on candidate image
`vllm-ds41f-kkref:01f1b874c774-r4b`: the r4a series (vLLM `01f1b874`,
patches 0001-0008) plus patch 0009 (tree `ee3a0fd4`).

## Background

- Adaptive verification's startup profile runs dummy rows that are marked as
  padding, which the MoE routers skip, so its cost tables leave out routed-
  expert work: they price a verified row at about 0.4 ms, while single-stream
  steps grow by several milliseconds per verified row
  (`experiments/2026-09-26-dspark-policy`). Patch 0009 profiles on distinct
  tokens marked as real rows (`SPARK3_DSPARK_PROFILE_TOKENS=random`).
- Five drafts (the drafter's trained block, graphs to 48 rows) already help
  code answers (+14-18% at one stream on the rebased stack) and are roughly
  neutral on reasoning and prose (`experiments/2026-09-27-lil-head`).

## Hypothesis

With verification priced on real rows at cost scale 1, adaptive verification
trims weak drafts on prose and keeps them on code, so five drafts gain on
answers without losing on reasoning or prose, against the r4a base
(`experiments/2026-09-26-dspark-policy`, runs b1 and b2).

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `k3real` | r4b image, distinct-token profile, cost scale 1.0, three drafts |
| `k5real` | as `k3real` with five drafts and graphs to 48 rows |
| `k3realm`, `k5realm` | as `k3real`, `k5real` with the marginal verification rule (patch 0006); they reuse those arms' pinned curves |

Both drop `--default-chat-template-kwargs` like the other candidate arms.

## Workload and gates

`sequence_lean.sh`: one boot per arm, LRU gate, reasoning and answer cases
at one and eight streams, three samples. A winner gets the full matrix
before promotion.

## Results

**Correction (same-protocol control).** These comparisons used the r4a runs
of `experiments/2026-09-26-dspark-policy` (b1, b2: full five-case matrix,
the night before) as the reference. A lean-protocol r4a control run on the
same image and config the next morning (`dsp-s-r4a-now`) read 0.9-1.2 ms per
single-stream step slower and 3-7% lower at eight streams than b1+b2, so
lean arms must be compared with lean controls. Against that control the
rebase is not slower: r5 (base Engram overlap) 51.06/53.63 ms against r4a
50.68/53.16 ms (prose/code steps), and r5c (patch 0002 re-ported, patched
NCCL, two boots) 50.18/53.06 ms, even with r4a and r4c at eight streams;
`experiments/2026-09-27-r5-engram-async` has the pooled table.

Screening runs (`results/private/bench/d5-s-*`, three samples per point),
against the r4a base (policy runs b1+b2). Every arm passed LRU 5/5; dgx1's
lowest MemAvailable was 6.35-6.5 GiB.

Profiled on distinct real rows, the verify forward costs 25.1, 31.7, 38.8
and 44.7 ms at 1-4 rows and 150.9 ms at 32 rows, against 19.2-20.4 and
31.7 ms on padding rows: a verified row costs about 6.5 ms, and 4 rows plus
the drafter match the real single-stream step (about 50 ms). At 32 rows
random tokens reach more distinct experts than real batches and overstate
the cost.

Single-stream step time (ms) and throughput change against r4a:

| Arm | prose step | code step | prose-nothink step | code-nothink step | code-nothink c1 | prose-nothink c8 | code c8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| r4a base | 49.81 | 51.98 | 50.67 | 52.42 | 62.5 tok/s | 170.6 tok/s | 184.5 tok/s |
| `k3real` | 44.61 | 49.28 | 45.56 | 52.32 | +0.4 ±1.8% | -5.0 ±2.3% | -4.1 ±3.5% |
| `k3realm` | 44.82 | 49.10 | 45.96 | 51.01 | -1.8 ±3.7% | -3.6 ±4.1% | -4.7 ±2.0% |
| `k5real` | 45.73 | 52.57 | 47.84 | 56.75 | +12.2 ±4.1% | -6.4 ±5.3% | -5.5 ±1.9% |
| `k5realm` | 46.89 | 51.24 | 48.51 | 55.69 | +10.9 ±4.0% | -4.0 ±3.3% | -4.3 ±1.9% |

- Priced on real rows, adaptive verification verifies about one row fewer
  per step at three drafts (prose c1 1.72 against 2.77) and steps shorten
  by about 5 ms, but accepted drafts fall in proportion: prose breaks even
  and code loses 4-7%. The marginal rule behaves the same. Trimming does
  not pay: the confidences are calibrated on average but do not single out
  losing rows well enough, and a row costs about as much as it returns.
  The zero-cost tables of the base verify almost every draft, which is as
  good as any policy tried.
- Five drafts help code answers (+11-12% at one stream, 2.9-3.0 accepted
  drafts per step) and cost 2-6% on reasoning and prose, with either rule.
  The same pattern held on the rebased stack. Depth 5 is a workload choice,
  not a default.
- Patch 0009 is the correct profile but is not adopted; keep three drafts
  with the base's profile.
