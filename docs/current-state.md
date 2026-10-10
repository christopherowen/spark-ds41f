# Current state

Production is the TP4 1M profile ([config/cluster-tp4.json](../config/cluster-tp4.json))
on the four-node ring, promoted 2026-10-10 as
[`2026-10-10-karmic-kraken-r6d`](../manifests/baselines/2026-10-10-karmic-kraken-r6d.json)
([promotion record](../experiments/2026-10-10-r6d-deterministic/README.md)):
image `vllm-ds41f-kkref:19f2c20ed4d6-r6d` (`sha256:795eb1c7`) on all four ranks,
Local Inference Lab vLLM `19f2c20e` with r6c's series and the first TileLang
migration ports (vLLM 0048-0080), B12X `236ddff0`. Its temperature-0 outputs do
not depend on the batch: the reduce-scatter adds in rank order (C3), the
compressor projection keeps one arithmetic at every row count (B5), and prefill
chunks align (threshold 8096). Recipe changes from r6: sparknet's one-shot
all-reduce up to 2 MiB, long-prefill threshold 8096.

The three-node profiles below (`config/cluster.json`, promoted 2026-10-05 as
[`2026-10-05-karmic-kraken-r6`](../manifests/baselines/2026-10-05-karmic-kraken-r6.json),
[promotion record](../experiments/2026-10-05-tilelang-r6/README.md)) stay on r6
and pin its lock until they are qualified on the triangle cabling.
The TileLang kernel family runs the model; B12X is the alternative backend in
the same image.

| Setting | Active value |
|---|---:|
| Kernel backend | TileLang (`kernel_backend: tilelang`): TileLang 0.1.15 + SM120/SM121 block-scaled MMA patches, DeepSeek TileKernels 2.0.0, sparknet 0.2.0 one-shot RoCE collectives; B12X runs WO, RoPE, KV cache writes, the compressor, BF16 GEMVs and the checkpoint loader |
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `04c30fa9` + patches 0001-0030 and 0032-0044 (0005-0009, 0011 and 0026 off by default; 0023 off in configuration), B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie, dense GEMM and prefill stage-fence patches, four-node relay and mesh transports, packed BF16 vocabulary projection, NCCL 2.30.7 + IB send-path fence and four-node ring patches |
| Image | `vllm-ds41f-kkref:04c30fa98e79-r6` (`sha256:b5225c98`), one digest on all ranks, built by `bin/spark build` |
| Decode tiles | block-32 FP8 decode rows on 16-, 32- or 64-row tiles by row count, TileLang's unspecialized pipeline, no K split; prefill rasterizes weights larger than L2 in panels (vLLM 0043) |
| Batch invariance | every kernel adds its reduction in a fixed order, so a row's result does not depend on its batch |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia-64k` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 524,288 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025) |
| B12X fixes (alternative backend) | indexer top-k ties by lowest logical position (B12X 0003); async-proxy fences before the TMA refill in the dense GEMM (0004) and in the BF16 and mHC prefill projections and contiguous attention (0005); W4A8 tiny decode off |
| Explicit KV memory | 3.5 GiB per rank |
| Reported KV capacity | 2,845,543 tokens (5.43x full 512K windows) |
| Memory-saver | signed DKMS 0.2.0 on all nodes; loaded UVM leaf-table packing enabled |
| Page-size profiles | 64 KiB selected; 4 KiB retains 2.2 GiB KV and a 262,144-token limit |
| Output head | exact 12-bit packed BF16 (`VLLM_DS41_PACKED_BF16_LM_HEAD=1`; B12X 0012, vLLM 0028), bit-identical to the checkpoint |
| Display carve-out | embedding and packed output-head weights in the firmware scanout reserve, loaded there directly (vLLM 0030) (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, the GPU's DRM card by PCI path, `/dev/dri/by-path/pci-000f:01:00.0-card`, as `/dev/dri/card0`); the DRM file closes after the import, so the text console keeps drawing |
| Sparse attention | TileLang sparse MLA with the MXFP4 indexer; BF16 attention (patch 0023) is available and off pending a fidelity test |
| Image input | vision tower loaded, up to 4 images per request, no host preprocessing cache |
| DSpark | 5 draft tokens, draft TP 3, adaptive verification (cost scale 2.0), dead verification rows below survival 0.2, block rejection; vocabulary-parallel greedy drafts over the checkpoint's BF16 drafter head (the target head's own tensor) and Markov projection; the drafter marks its padding rows for the MoE routers (vLLM 0042); cost curves profiled for this image (`tp3-r6-20261005`) |
| Collectives | sparknet one-shot all-reduce and all-gather over the triangle (`oneshot-direct`); the requested policy is enforced and a declined or lost backend fails startup (vLLM 0027, 0038) |
| Profiling context | each KV cache group gets its own blocks, in every dummy layout (vLLM 0039, 0041) |
| CUDA graphs | full, capture sizes 1-48 |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | read beside the forward launch (`SPARK3_ENGRAM_ASYNC=1`); base overlap off (`VLLM_DS41_ENGRAM_OVERLAP=0`) |
| L2 weight prefetch | on (base default on SM121); TileLang scale words prefetched with each weight (vLLM 0044) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s); hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Reasoning | enabled by default; a request's `thinking` or `enable_thinking` is honored |

The acceptance benchmark is in the
[promotion record](../experiments/2026-10-05-tilelang-r6/README.md); the TP3
profile's long-context admission and retrieval evidence (measured on r5o, same
shape) is in the
[memory-saver deployment](../experiments/2026-10-02-memory-saver-capacity/README.md).
Against r5p, the kernels change family; the weights, the TP3 shape and the
memory guards are unchanged.

The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [setup.md](setup.md).
