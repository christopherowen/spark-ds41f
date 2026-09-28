# Current state

Promoted 2026-09-28 as
[`2026-09-28-karmic-kraken-r5j`](../manifests/baselines/2026-09-28-karmic-kraken-r5j.json)
and running on all three nodes from `config/cluster.json`.

| Setting | Active value |
|---|---:|
| Sources | Local Inference Lab `integration/karmic-kraken-beta` vLLM `04c30fa9` + patches 0001-0020 (0005-0009 and 0011 off by default), B12X `e39b437b` + switchless RoCEnante and CuTe DSL 4.7.1 pin patches, NCCL 2.30.7 + IB send-path fence |
| Image | `vllm-ds41f-kkref:04c30fa98e79-r5j`, one digest on all ranks, built by `bin/spark3 build` |
| Hosts | DGX Spark 26.09.2, kernel `7.0.0-1019-nvidia` with `kho=off`, driver 580.178.04, no desktop |
| Tensor parallel ranks | 3 |
| Maximum model length | 131,072 tokens |
| Maximum sequences | 8 |
| Maximum parallel prefills | 1 |
| Batched-token budget | 4,096 |
| Prefill sequence parallelism | from 205 tokens, where the reduce-scatter exceeds the one-shot RoCE all-reduce; CED encoder layers (patches 0014-0018) |
| Explicit KV memory | 1.4 GiB per rank |
| Reported KV capacity | 575,304 tokens (4.39x full 131K windows) |
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

Measured on this configuration: LRU coherence gate 5/5; single-stream
prose/code about 50/63 tok/s with reasoning, code answers 79 tok/s; code at
eight streams 188 tok/s (226 tok/s for code answers); cold prefill
2K 4.3k, 32K 4.8k, 64K 4.6k tok/s (4K 3.8k, 14K 3.8k, 36K 3.8k, 61K 3.8k on real text); dgx1 minimum MemAvailable 6.44 GiB under load.
The previous state
(2026-09-20, 498,145 KV tokens in 3 GiB, incoherent code output) is retained in
[`2026-09-20-live`](../manifests/baselines/2026-09-20-live.json).

To reproduce it elsewhere, follow [replicate.md](replicate.md).
