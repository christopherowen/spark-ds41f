# DeepSeek V4.1 Flash fundamentals

The shapes, number formats and terms behind the kernel work: what M, N and K
mean, what one decoder layer computes, how tensor parallelism splits it across
Sparks, and why decode speed is mostly a question of bytes.

Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a`. Shapes are per rank
at TP4 unless marked.

- [Reading a GEMM: M, N and K](#reading-a-gemm-m-n-and-k)
- [GB10 numbers that matter](#gb10-numbers-that-matter)
- [The model at a glance](#the-model-at-a-glance)
- [Number formats](#number-formats)
- [One decoder layer](#one-decoder-layer)
- [Attention: window, compression, index](#attention-window-compression-index)
- [mHC and Engram](#mhc-and-engram)
- [DSpark and decode rows](#dspark-and-decode-rows)
- [TP3 and TP4](#tp3-and-tp4)
- [Glossary](#glossary)

## Reading a GEMM: M, N and K

Almost every projection in the model is one matrix multiply. The activations
*x* are a batch of rows, the weight *W* is stored output-major, and the product
has one row per input row.

```text
y[M, N] = x[M, K] · W[N, K]ᵀ
y[m, n] = Σ over k = 0 … K−1 of x[m, k] · W[n, k]
```

**K: the reduction dimension.** K is the number of input features, the length
of each dot product. Every output element is a sum of K products, so K sets the
work per output and the weight's width. The **order** in which those K products
are added is what fixes the result's bits: floating-point addition is not
associative.

**N: output features.** N is the number of output features, one weight row
each. Kernels split N across thread blocks (each takes `block_N` columns), and
tensor parallelism splits N across GPUs for column-parallel layers.

**M: rows.** M is the number of rows in the batch, one per token being
processed. In prefill M is thousands (a whole prompt chunk). In decode it is
small: one stream's step verifies six rows.

**Shapes in this repository.** Weights are written `(N, K)` per rank. Q-B at
TP4 is `(8192, 1280)`: 8,192 outputs, each a dot product over 1,280 inputs. In
FP8 that is 8192 × 1280 bytes = 10 MiB of weights.

### Why decode is about bytes

A GEMM does 2·M·N·K floating-point operations and must read at least N·K weight
bytes (FP8). Its arithmetic intensity is therefore about **2·M FLOP per weight
byte**. At six decode rows that is 12 FLOP per byte; GB10's tensor cores need
hundreds per byte to be the limit. So a decode projection's time is close to
*weight bytes ÷ memory bandwidth*: Q-B reads 10 MiB, about 38 µs from DRAM at
273 GB/s, or a fraction of that if the weight is already in L2. Prefill, with
thousands of rows, is compute-bound instead.

### Tiles, stages and splits

A GEMM kernel gives each thread block (CTA) one output tile of
`block_M × block_N`, and walks K in steps of `block_K`, keeping `num_stages`
steps in flight in shared memory so loads overlap the math. A 64-row tile
serving six rows still does 64 rows of tensor-core work, which is why decode
now uses 16-, 32- and 64-row tiles chosen by the row count. **Split-K** cuts K
into shards summed separately and then added; that changes the order of the
sum, so the prefill kernel must split the same way or decode and prefill
disagree in the last bits.

> Batch invariance means a row's output does not depend on how many other rows
> share the batch. It holds when every kernel that can serve a row adds the K
> products in the same order, whatever the tile, the padding or the row count.

## GB10 numbers that matter

| Property | Value | Why it matters here |
| --- | --- | --- |
| Streaming multiprocessors | 48 (SM121, Blackwell) | A decode GEMM needs enough CTAs to keep 48 SMs pulling bytes; 50 tiles on 48 SMs leaves a ragged second wave. |
| L2 cache | 24 MiB | One layer's Q-A/KV plus Q-B (18.75 MiB) fits; the DSpark main projection (37.5 MiB per rank) does not. |
| Memory | 128 GB LPDDR5X, ~273 GB/s | Unified CPU/GPU memory; sets the floor for every cold decode projection. |
| Shared memory per block | 99 KiB | Caps `num_stages × (block_M + block_N) × block_K` for FP8 tiles. |
| Block-scaled tensor-core MMA | m16n8k32, MXF8F6F4 | Applies one UE8M0 scale per 32 K-elements of each operand in hardware, so MXFP8 needs no software rescaling loop. |
| Interconnect | ConnectX-7 RoCE | Each layer has two all-reduces; at TP4 the Sparks form a four-node ring, at TP3 a triangle. |

## The model at a glance

DeepSeek V4.1 Flash is a 40-layer mixture-of-experts model with a compressed,
sparsely indexed attention, four-stream hyper-connections, two hashed n-gram
memory layers, and a built-in speculative drafter. Values from the checkpoint's
`config.json`:

| Field | Value | Meaning |
| --- | ---: | --- |
| `hidden_size` | 5,120 | Width of the residual stream; K of most projections that read it. |
| `num_hidden_layers` | 40 | Target decoder layers (plus 3 drafter layers, below). |
| `num_attention_heads` × `head_dim` | 64 × 512 | Query heads, each 512 wide; 64 of the 512 dimensions carry RoPE (`qk_rope_head_dim`). |
| `num_key_value_heads` | 1 | One shared 512-wide latent key/value per token (MLA-style). |
| `q_lora_rank` | 1,280 | The query is projected down to 1,280 (Q-A), normalized, then up to the heads (Q-B). |
| `o_groups` × `o_lora_rank` | 8 × 1,024 | The output projection is grouped: 8 groups of 8 heads, each reduced to 1,024 (WO-A), then 8,192 → 5,120 (WO-B). |
| `n_routed_experts` / per token | 384 / 6 | Routed MoE experts and how many each token uses; scoring `sqrtsoftplus`, routed scaling 1.5. |
| `n_shared_experts` | 1 | One always-on expert beside the routed ones. |
| `moe_intermediate_size` | 2,304 | Expert hidden width (SwiGLU, clamped at `swiglu_limit` 10). |
| `vocab_size` | 129,280 | Output vocabulary; the LM head is 129,280 × 5,120. |
| `max_position_embeddings` | 1,048,576 | 1M positions: YaRN factor 16 over 65,536. |
| `sliding_window` | 128 | Recent tokens every layer attends to exactly. |
| `index_n_heads` × `index_head_dim` | 32 × 128 | The indexer's scoring heads. |
| `index_topk` | 512 | Compressed entries each query selects beyond the window. |
| `hc_mult` | 4 | Residual streams in mHC; mixing uses 20 Sinkhorn iterations. |
| `engram_layer_ids` | 1, 14 | Layers with hashed n-gram memory (up to 4-grams, 8 heads × 256). |
| `num_nextn_predict_layers` | 3 | DSpark drafter layers; draft block of 5 tokens. |

## Number formats

**E4M3 (FP8).** One byte: sign, 4 exponent bits, 3 mantissa bits. Dense weights
and decode activations use it.

**E2M1 (FP4).** Four bits: sign, 2 exponent bits, 1 mantissa bit. The routed
experts' weights use it, two values per byte.

**UE8M0 scale.** An unsigned 8-bit exponent: a pure power of two, 2^(e−127).
Multiplying by it is exact, which is why the hardware can fold it into the MMA.

**MXFP8 and MXFP4.** Microscaling: one UE8M0 scale per block of 32 values.
Dense weights carry one scale per 32 × 32 block; activations are quantized per
row, one scale per 32 elements, when each projection runs.

Some small projections stay in BF16 (the router gate, the indexer's head
weights, the compressor). The KV cache is quantized too: a sliding-window
record is 528 bytes (512 FP8 values and 16 scale bytes), and an indexer key is
68 bytes (128 FP4 values and 4 scale bytes).

## One decoder layer

The order of work in one target layer during decode:

1. **mHC attention mix**: combine the four residual streams into the attention
   input.
2. **Attention norm** (RMSNorm), then **fused Q-A/KV**: one projection gives the
   1,280-wide query latent and the 512-wide KV latent.
3. Normalize both, then **Q-B** expands the query latent to this rank's heads;
   RoPE rotates 64 of each head's 512 dimensions.
4. **Cache writes**: the KV latent goes into the sliding-window cache and, on
   compressing layers, the compressed cache.
5. **Indexer** (index source layers): **indexer Q-B** projects 32 scoring heads,
   scores the compressed entries, keeps the top 512.
6. **Sparse MLA**: each query head attends to the 128-token window plus the 512
   selected entries.
7. **WO-A** (grouped) and **WO-B** project back to 5,120; **all-reduce** across
   ranks.
8. **mHC FFN mix** and **FFN norm**.
9. **Router**: score 384 experts, pick 6. **Routed experts** (MXFP4) and the
   **shared expert** (MXFP8) run; **all-reduce**.

Every weight in that list, per rank at TP4:

| Projection | Shape (N × K) | Split | MiB / rank | Layers | Notes |
| --- | --- | --- | ---: | --- | --- |
| Fused Q-A/KV | 1792 × 5120 | replicated | 8.75 | 40 | 1,280 query latent + 512 KV latent; MXFP8. |
| Q-B | 8192 × 1280 | column | 10.0 | 40 | 16 heads × 512 per rank. TP3: 12288 × 1280 (24 padded heads). |
| Indexer Q-B | 4096 × 1280 | replicated | 5.0 | 8 | 32 index heads × 128; index source layers only. |
| Indexer head weights | 32 × 5120 | replicated | 0.31 | 8 | BF16. |
| Compressor KV/gate | (512·r) × 5120 | replicated | 5–10 | 4 | BF16, ratio r = 1 or 2; KV source layers 2, 8, 14, 20. |
| WO-A | 2 × (1024 × 4096) | by group | 8.0 | 40 | Grouped: 2 of 8 groups per rank, each 8 heads × 512 → 1,024. |
| WO-B | 5120 × 2048 | row | 10.0 | 40 | Partial sums, then the attention all-reduce. |
| Router gate | 384 × 5120 | replicated | 3.75 | 40 | BF16; top-6 with bias-corrected (noaux) selection. |
| Shared expert gate/up | 1152 × 5120 | column | 5.6 | 40 | 2 × 576 (2,304 ÷ 4); MXFP8. |
| Shared expert down | 5120 × 576 | row | 2.8 | 40 | K = 576 takes 64- or 192-wide K blocks, not 128. |
| Routed experts | 384 × (1152×5120 + 5120×576) | by width | ~4.5 each | 40 | MXFP4 with scales; 6 of 384 read per token. |
| DSpark main projection | 6400 × 6144 | column | 37.5 | drafter | 25,600 outputs in total; larger than L2. |
| LM head | 32320 × 5120 | vocab | — | 1 | 129,280 ÷ 4 per rank; served from exact 12-bit packed BF16. |

MiB are FP8 weight bytes (N × K) unless noted; BF16 counts 2 bytes per value.

## Attention: window, compression, index

Each token keeps one 512-wide latent shared by all query heads. Three
mechanisms keep 1M-token attention affordable:

**Sliding window.** Every layer attends exactly to the last 128 tokens (plus the
drafted rows of the current step).

**Compression.** Per-layer `compress_ratios` (43 entries for 40 target and 3
drafter layers): 0 means window only (5 layers), 1 or 2 means one compressed
entry per token or per two tokens (20 and 18 layers). Layers 2, 8, 14 and 20
own the compressed caches; the others read the nearest owner below them.

**Indexer.** Layers 2, 8, 14, 20, 24, 28, 32 and 36 score the compressed entries
with 32 heads × 128 and select the top 512. Layer 20 first pre-selects 2,048
candidate blocks of 8. Other compressed layers reuse the latest selection.

**Sparse MLA.** Each query head then attends to the window plus the 512 selected
entries, so attention cost per token stays near constant as context grows to
1M positions.

## mHC and Engram

**mHC: hyper-connections.** Instead of one residual stream, the model carries
four (`hc_mult` 4). Before attention and before the FFN, learned mixing
weights, pushed toward a doubly stochastic matrix by 20 Sinkhorn iterations,
combine the streams into the sublayer input and distribute its output back. It
is many small, latency-bound kernels per layer.

**Engram: hashed n-gram memory.** Layers 1 and 14 look up embeddings keyed by
hashed n-grams of up to four tokens: about 384M rows per layer, 8 heads × 256
dims. The tables live on disk; rows are read on a host thread while the forward
graph launches.

## DSpark and decode rows

DSpark is the model's own speculative drafter: three extra layers fed by target
layers 37–39, with their own MoE (128 experts, 3 per token) and a rank-256
Markov bias. Each step drafts a block of up to five tokens, and the target
verifies them in one forward pass.

**Rows per step.** One stream verifies its current token plus five drafts:
**6 rows**. Two streams: 12. Four: 24. Eight: 48. This is why decode tiles are
sized 16, 32 and 64 rows.

**Step time and acceptance.** Tokens per second = accepted tokens per step ÷
step time. Step time measures the kernels; acceptance depends on the text and
the draft-length policy. A kernel change is judged on step time.

## TP3 and TP4

Tensor parallelism splits each layer across Sparks. Column-parallel layers
split N, row-parallel layers split K and add partial sums with an all-reduce.
Small or index-related projections are replicated so no extra collective is
needed.

| Split | Projections | TP4 per rank | TP3 per rank |
| --- | --- | --- | --- |
| Query heads | Q-B, WO-A | 16 heads, 2 groups | 24 heads, 3 groups (64 → 72 padded, 8 → 9 groups) |
| Expert width | Routed and shared experts | 576 of 2,304 | 768 of 2,304 |
| Row-parallel | WO-B, expert down | K ÷ 4, all-reduce | K ÷ 3, all-reduce |
| Replicated | Q-A/KV, indexer, router, compressor | Same shape on every rank | Same shape on every rank |

## Glossary

| Term | Meaning |
| --- | --- |
| K | Reduction dimension of a GEMM: input features, the length of each dot product. |
| N | Output features: one weight row each. |
| M, rows | Tokens in the batch. Decode: 6 per stream; prefill: the prompt chunk. |
| `block_M` / `block_N` / `block_K` | A CTA's output tile height and width, and how much of K it loads per step. |
| `num_stages` | K steps buffered in shared memory at once, so loading overlaps math. |
| CTA | Cooperative thread array (thread block): the unit scheduled onto one SM. |
| Split-K | Summing K in shards, then adding the shards; changes the sum order. |
| Swizzle / raster | The order tiles are launched in. Panels keep a too-large weight's columns in L2 while several row tiles reuse them. |
| TMA | Tensor Memory Accelerator: hardware bulk copies of tiles into shared memory. |
| cp.async | Asynchronous per-thread copies into shared memory; used for scale words. |
| Warp specialization | Producer warps load while consumer warps compute, handing off through barriers. |
| MMA | Tensor-core matrix multiply-accumulate instruction (m16n8k32 for MXFP8 on SM121). |
| MXFP8 / MXFP4 | FP8 or FP4 values with one power-of-two (UE8M0) scale per 32. |
| TP | Tensor parallelism: each layer's weights split across GPUs, joined by collectives. |
| All-reduce | Sum partial results across ranks; two per layer. sparknet does it in one shot over RoCE. |
| MLA | Multi-head latent attention: heads share one compressed latent key/value. |
| Indexer | Scores compressed cache entries and selects the top 512 per query. |
| mHC | Manifold-constrained hyper-connections: four residual streams mixed by Sinkhorn-normalized weights. |
| Engram | Hashed n-gram embedding memory on layers 1 and 14. |
| DSpark | The built-in speculative drafter and its verification scheme. |
| Acceptance | Drafted tokens the target confirms per step. |
| Step time | Wall time of one target verification step; the kernel metric. |
| Batch invariance | A row's result is independent of the batch it ran in. |
| CUDA graph | A recorded launch sequence replayed per decode step without CPU launch cost. |
| L2 prefetch | A side stream that pulls upcoming weights into L2 while latency-bound kernels run. |

Sources: the checkpoint config, the vLLM DeepSeek V4.1 model code in the
patched tree, and the 2026-10-05 TP4 serving traces. Sizes are derived from
those shapes.
