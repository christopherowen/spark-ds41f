# The rebased stack with patch 0002 re-ported (r5c candidate)

Base: the promoted configuration on candidate image
`vllm-ds41f-kkref:04c30fa98e79-r5c`: Local Inference Lab vLLM `04c30fa9`
with patches 0001-0009 (tree `f250542a`), where 0002 is re-ported onto the
base's new disk-Engram code; B12X `e39b437b` with the switchless patch
(tree `f77d175f`); NCCL 2.30.7 with `patches/nccl` (tree `47687d2a`).

## Hypothesis

The rebased stack lost 1.0-1.6 ms per single-stream step against r4a, and
its own Engram overlap saves only 0.4-1.0 ms against reading rows before the
launch, where patch 0002 saved about 1.85 ms (`experiments/2026-09-27-lil-
head`). With 0002 in place of the overlap, the rebased stack matches r4a,
and L2 weight prefetch (about 1.2 ms per prose step) puts it ahead.

## Arm

`cluster-r5c.json`: r5c image, `SPARK3_ENGRAM_ASYNC=1` (as promoted) with
`VLLM_DS41_ENGRAM_OVERLAP=0`, own compile cache, and
`--default-chat-template-kwargs` dropped like the other candidate arms.
Compared with the r4c verification run (`experiments/2026-09-27-nccl-
fence`) and the r4a base.

## Workload and gates

One boot, as the r4c verification: LRU 5/5, the lean decode screen, cold
prefill at 2K and 32K; memory guards unchanged.

## Results

Pending.
