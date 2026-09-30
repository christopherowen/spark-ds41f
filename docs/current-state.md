# Current state

Promoted 2026-09-30 as
[`2026-09-30-karmic-kraken-r5n`](../manifests/baselines/2026-09-30-karmic-kraken-r5n.json)
and running on all three nodes from `config/cluster.json`.

| Setting | Active value |
|---|---:|
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `04c30fa9` + patches 0001-0026 (0005-0009, 0011 and 0026 off by default; 0023 off in configuration), B12X `f8069b2c` + switchless RoCEnante, CuTe DSL 4.7.1 pin, top-k position-tie and dense GEMM stage-fence patches, NCCL 2.30.7 + IB send-path fence |
| Image | `vllm-ds41f-kkref:04c30fa98e79-r5n`, one digest on all ranks, built by `bin/spark3 build` |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 262,144 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Indexer under sequence parallelism | each rank scores and selects its own rows and all-gathers the top-k positions (patch 0025) |
| Indexer top-k ties | lowest logical position (B12X 0003): selections repeat exactly |
| Dense GEMM stage release | async-proxy fence before the TMA refill (B12X 0004): the shared expert no longer returns wrong columns beside the routed MoE |
| Explicit KV memory | 2.2 GiB per rank |
| Reported KV capacity | 1,348,708 tokens (5.14x full 256K windows) |
| Display carve-out | embedding and output-head weights (842.5 MiB per rank) in the firmware scanout reserve (`SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, the GPU's DRM card by PCI path, `/dev/dri/by-path/pci-000f:01:00.0-card`, as `/dev/dri/card0`); the DRM file closes after the import, so the text console keeps drawing |
| Sparse-attention arithmetic | B12X's tuned choice (`VLLM_DS41_ATTENTION_COMPUTE=auto`); BF16 (patch 0023) is available and off pending a fidelity test |
| Image input | vision tower loaded, up to 4 images per request, no host preprocessing cache |
| DSpark | 5 draft tokens, draft TP 3, adaptive verification (cost scale 2.0), dead verification rows below survival 0.2, block rejection; vocabulary-parallel greedy drafts, NVFP4 drafter head and Markov projection |
| CUDA graphs | full, capture sizes 1-48 |
| B12X W4A8 tiny decode | disabled (`B12X_W4A8_TINY_DECODE=0`) |
| Engram projection | sharded across ranks (`projection_tp`) |
| Engram rows | read beside the forward launch (`SPARK3_ENGRAM_ASYNC=1`); base overlap off (`VLLM_DS41_ENGRAM_OVERLAP=0`) |
| L2 weight prefetch | on (base default on SM121; runs since r5i) |
| B12X autotune | disabled |
| FlashInfer autotune | disabled (`--no-enable-flashinfer-autotune`) |
| Memory guards | 5 GiB startup (0.25 s), 3 GiB steady (2 s); hosts set `vm.watermark_boost_factor=0` |
| Async scheduling | enabled |
| Reasoning | enabled by default; a request's `thinking` or `enable_thinking` is honored |

Measured on r5l, which differs only by the top-k tie rule and the dense GEMM stage fence (neither with a measured cost): LRU coherence gate 5/5; needle retrieval
3/3 at 152,914 tokens; single-stream prose/code about 50/60 tok/s with reasoning,
code answers 81 tok/s; code at eight streams 187 tok/s (233 tok/s for code
answers); cold prefill 2K 4.4k, 32K 4.9k, 64K 4.9k, 131K 4.7k tok/s
(4K 3.9k, 16K 3.8k, 32K 3.8k, 64K 3.8k, 131K 3.8k, 200K 3.8k on real text); dgx1 minimum
MemAvailable 6.38 GiB under load.
The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [replicate.md](replicate.md).
