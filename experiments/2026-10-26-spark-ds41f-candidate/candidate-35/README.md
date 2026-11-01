# Candidate 34 (`spark-ds41f:candidate-35`)

Built directly on `candidate-33` (`0001..0663`, 548 patches) with 4 hardware-audited vLLM patches (`0664..0667`, 552 patches total; `patch_head` `635eab8b20c9`, `expected_tree` `9744f37b7ea6`, `patchset_sha256` `07b681514a53`):

| Patch | PRs | Summary | GB10 Hardware Audit & Microbenchmark |
| --- | --- | --- | --- |
| `0664` | `#740` | Cover odd `c2` and full-depth speculative batches in `dense_varlen_decode_shapes` | Adds `11` (`c2`, `K=5`) and `max_num_seqs * (1 + num_speculative_tokens)` (`48` at `c8`, `K=5`) to CUDA-graph capture ladder without changing `max_cudagraph_capture_size` |
| `0665` | `#764`, `#765` | Use 16-row MoE decode tiles below 1 tile/expert (`tokens * 8 < 256 -> M <= 31`) and 3-tile 2D prefill swizzle (`num_Swizzle_M = 48`) | `STACK:0`, `KL=0`; `grouped_fp8_fp4_gemm` up (`99.01 -> 91.55 us`, `1.081x`) and down (`59.17 -> 53.66 us`, `1.103x`) at `M=24` (`-738 us` across 58 MoE layers); `M=4096` prefill up (`1994 -> 1966 us`) and down (`1097 -> 1084 us`) |
| `0666` | `#744`, `#745`, `#776` + Hybrid TP/SP `fused_wqa_wkv` | Run `attention_front` at prefill (`M <= 8192`), share group metadata across layer-type groups, trim TP3 rank-2 group (`30 -> 20` layers), and use zero-copy hybrid TP-decode / SP-prefill `_PaddedMergedColumnParallelLinear` | `STACK:0`, `KL=0`; prefill `attention_front` `2.38x–3.61x` faster (`186.8 -> 78.3 us` at `M=4096`); hybrid `fused_wqa_wkv` keeps TP-sharded decode (`rows <= 56`, `3.11 MB` L2 prefetch at TP3) while eliminating `1.24 GB` of extra RoCE All-Gather traffic and 61 collective launches per `4,096`-token SP prefill chunk |
| `0667` | `#833` (Part A) | Fuse `mhc.post` into `engram_post_gate` and chain Engram gates on PDL | `STACK:0`, `KL=0`; `engram_post_gate` + `mhc_post_and_finalize_streams` `1.26x–1.47x` faster (`6.94 -> 4.74 us` at `M=1`, `20.51 -> 15.81 us` at `M=48`), saving 1 kernel launch and `[M, 5120]` BF16 DRAM round-trip per Engram layer |

## Audited & Rejected Candidates on GB10 (`dgx4`)

- **PR `#837`** (`exp/dsv41-packed-bf16-markov-and-embed`): **REJECTED** (`embed_tokens` `1.32x–1.86x` slower across `M=1..2048`; `Markov W2` `1.15x–1.60x` slower across 5-step draft loop + 1 ULP CSR drift).
- **PR `#838`** (`exp/dsv41-packed-bf16-router-and-compressor-gemm`): **REJECTED** (`router.gate` `2.25x–3.72x` slower L2-hot, `+472 us` across 61 layers; `1.07x–4.03x` slower DRAM-cold).
- **PR `#747`**: **SKIPPED** (already superseded by `0635` `perf(dsv41): retire owner and engram L2 prefetch self-evictions`).
- **PR `#833` Part B** (1-CTA `engram_rows_step_kernel`): **REJECTED** (`1.6x–58x` slower for `M >= 4`).
