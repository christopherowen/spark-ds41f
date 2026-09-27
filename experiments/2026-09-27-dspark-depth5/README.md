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

Both drop `--default-chat-template-kwargs` like the other candidate arms.

## Workload and gates

`sequence_lean.sh`: one boot per arm, LRU gate, reasoning and answer cases
at one and eight streams, three samples. A winner gets the full matrix
before promotion.

## Results

Pending.
