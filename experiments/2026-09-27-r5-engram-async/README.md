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

`results/private/bench/ea-s-r5c`, one boot, three samples: LRU 5/5, no failed
requests; the asynchronous path logged "asynchronous disk Engram rows
enabled"; NCCL 2.30.7 (patched build) loaded; graphs 0.99 GiB (r4a about
0); dgx1's lowest MemAvailable 6.5 GiB. Cold prefill 3,572-3,854 tok/s at 2K
and 4,051-4,167 tok/s at 32K (on NCCL), as on r3.

| Single-stream step (ms) | prose | code | prose-nothink | code-nothink |
|---|---:|---:|---:|---:|
| r4a base | 49.81 | 51.98 | 50.67 | 52.42 |
| r5 base (overlap) | 51.06 | 53.63 | 51.90 | 53.64 |
| r5c (patch 0002) | 49.64 | 52.79 | 50.82 | 53.42 |

- Patch 0002 recovers 1.2-1.4 ms per step against the base overlap: prose
  steps now match r4a, code steps remain 0.8-1.0 ms slower. (Correction,
  2026-09-28: the L2 prefetch did not run on r5c; it failed to compile until
  r5i.)
- Against the r4a runs of the night before (full matrix) r5c looked 4.5-6.4%
  lower at eight streams, but a lean r4a control the same morning showed
  that gap is the protocol, not the stack.

Same-morning lean comparison (r4c and r5c two boots each, r4a one boot;
`ea-s-r5c*`, `nf-s-r4c*`, `dsp-s-r4a-now`), single-stream steps (ms):

| Arm | prose | code | prose-nothink | code-nothink |
|---|---:|---:|---:|---:|
| r4a control | 50.68 | 53.16 | 51.77 | 53.60 |
| r4c (r4b + NCCL) | 51.37 | 53.53 | 51.73 | 53.72 |
| r5c | 50.18 | 53.06 | 51.25 | 53.28 |

- r5c steps are 0.4-1.2 ms shorter than r4c (prose -1.19 ms, clearest) and
  0.1-0.5 ms shorter than r4a. At eight streams r5c is even with r4c
  (-3.6 to +0.2%, only prose-nothink -2.0 ±1.4% beyond noise).
- r5c is the better candidate: the Local Inference Lab heads with patch
  0002, L2 weight prefetch and the patched NCCL.
