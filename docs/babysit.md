# Upstream follow-ups

Issues and pull requests outside this repository that we opened, or that a
decision here depends on. Each row says what we are waiting for and what to do
when it changes. Add a row when we open or start relying on an upstream thread;
remove it once the change has landed here or no longer matters.

Check them all at once:

```sh
for ref in tile-ai/tilelang#3287 tile-ai/tilelang#3288 \
           NVIDIA/open-gpu-kernel-modules#1269 NVIDIA/cutlass#3038 \
           eugr/spark-vllm-docker#4 christopherowen/dgx-spark-power-control#1; do
  gh api "repos/${ref%#*}/issues/${ref#*#}" --jq "\"$ref [\(.state)] \(.updated_at[:10]) \(.title)\""
done
```

## Waiting on upstream

| Thread | What | Our stake | Status (2026-10-05) | When it changes |
| --- | --- | --- | --- | --- |
| [tile-ai/tilelang#3287](https://github.com/tile-ai/tilelang/issues/3287) | Warp-specialized pipeline releases a stage before a non-TMA shared copy is consumed. Reported on SM120 with the block-scaled GEMM and narrow scale rows | Probably the race our 16-row decode tiles hit ([decode-kernel experiment](../experiments/2026-10-05-tilelang-decode-kernels/README.md)), which is why decode runs with `TL_DISABLE_WARP_SPECIALIZED` (vLLM 0043) | Open, not ours. Our reproducer is not yet run against the fix | Run [race2.py](../experiments/2026-10-05-tilelang-decode-kernels/bundles/dump/race2.py) on a TileLang with #3288 in a kernel-lab window. If the failures stop, add our SM121 data to #3287; if not, file a separate issue with the reproducer |
| [tile-ai/tilelang#3288](https://github.com/tile-ai/tilelang/pull/3288) | Proposed fix for #3287 | Lets decode tiles use warp specialization again, if that is ever faster | Open | After it is released and our TileLang pin moves past it, re-run the decode sweep with warp specialization before changing 0043 |
| [NVIDIA/open-gpu-kernel-modules#1269](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269) | 64 KiB-page ARM64: DMA submap not 2 MiB-aligned, so large GPU mappings fault (Xid 31) | Blocks R610 on the 64 KiB profiles; production stays on 580.178.04 | Open, not ours; reported to NVIDIA in the developer forum. The qualification carries a one-line RM patch | When a driver release includes the fix, re-qualify R610 (memory saver 0.3.0, `NVreg_EnableSystemMemoryPools=0`) |

## Our pull requests

| Thread | What | Status (2026-10-05) | Next |
| --- | --- | --- | --- |
| [NVIDIA/cutlass#3038](https://github.com/NVIDIA/cutlass/pull/3038) | Enable the SM121-gated MXFP4 MoE kernel path | Open since 2026-02-16, mergeable, review required | Answer review comments; rebase if it stops being mergeable |
| [eugr/spark-vllm-docker#4](https://github.com/eugr/spark-vllm-docker/pull/4) | Cache encodings to shorten rebuilds | Open since 2025-12-19, no activity | Close it if it is no longer wanted |
| [christopherowen/dgx-spark-power-control#1](https://github.com/christopherowen/dgx-spark-power-control/pull/1) | Power diagnosis, passive tracing and firmware recovery analysis | Draft | Merge only after the hardware writes have been tested with the owner's approval |
