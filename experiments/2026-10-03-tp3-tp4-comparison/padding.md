# Padding with three and four ranks

Four ranks remove the model-dimension padding needed for TP3 attention,
Engram and the target vocabulary. They do not eliminate graph-capacity,
sequence-parallel or kernel-tile padding. No padding implementation was changed
during this comparison.

This audit uses the actual served SM121 implementation, not just the generic
DeepSeek model: vLLM tree `c108cd6d1fe8e2d3b91c065feefe818742159020` and B12X
tree `2077a53bfb2a993b385f1a1bd9799b0a89694976`. Paths below are relative to
the respective prepared source repository. The checkpoint has hidden size
5,120, 64 attention heads, eight output groups, vocabulary 129,280 and expert
intermediate size 2,304.

| Quantity | TP3 | TP4 | Consequence |
|---|---:|---:|---|
| Attention heads, global | 64 → 72 | 64 | Extra zero-weight heads disappear; 24 versus 16 heads per rank. |
| Attention output groups, global | 8 → 9 | 8 | Three versus two groups per rank. |
| Engram WKV projection, global output width | 25,600 → 25,632 | 25,600 | Per-rank widths 8,544 versus 6,400; TP4 skips the trailing-output slice/copy. |
| Target vocabulary, global padded rows | 129,280 → 129,408 | 129,280 | Target embedding/head shard rows 43,136 versus 32,320. |
| Routed expert intermediate width per rank | 768 | 576 | Both are exact logical shards; TP4 needs the compact 64-channel-tail kernel path. |
| Compact W4A8 micro intermediate workspace width | Not this compact path | 576 → 640 | Local scratch alignment remains; this is not padding the model's expert weights to 640. |

## Exact implementation

- `vllm/model_executor/models/config.py`,
  `DeepseekV41ForCausalLMConfig.update_model_config_for_parallelism`: on the
  SM120 family, round output groups to a multiple of TP, then preserve eight
  heads per group. With TP4 the early return leaves the original dimensions.
  The served `vllm/models/deepseek_v4_1/attention.py` divides these dimensions
  by TP and permits padded weight loading only when originals differ.
- `vllm/models/deepseek_v4_1/common/engram.py`,
  `_PaddedColumnParallelLinear`: round output width to a multiple of
  `TP * 32` for FP8 scale blocks. Zero-fill missing checkpoint rows and slice
  gathered output only if that changes the width. TP4 takes the unchanged
  width branch. Keep the helper for TP3 support; it does not manufacture
  extra columns for TP4.
- `vllm/model_executor/layers/vocab_parallel_embedding.py`: global vocabulary
  alignment is `lcm(padding_size, TP)`; default padding size is 64. The target
  `ParallelLMHead` construction in `vllm/models/deepseek_v41/nvidia/model.py`
  does not override it, and the served SM121 class inherits that constructor.
  Thus global alignment is 192 for TP3 and 64 for TP4. This is the target
  BF16 head; the drafter's quantized head has its own packing requirements.
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`,
  `mxfp4_round_up_hidden_size_and_intermediate_size`: B12X returns the exact
  model dimensions. `fused_moe/b12x.py` accepts intermediate widths divisible
  by 32, which includes 576. B12X
  `b12x/moe/fused_moe/_impl.py` selects `n64_repack` when the intermediate
  width modulo 128 is 64. The preparation packs the logical weights in place.
  Larger compact launches select grouped M16 routing; this is a real kernel
  path difference from TP3, not just a smaller copy of its 768-channel kernel.
  `b12x/moe/_shared/kernels/w4a8_compact_micro.py::_layout` allocates the
  intermediate scratch at `ceil(N / 128) * 128`, hence 640 for N=576.

## Padding and replication that remain

- The configured target CUDA graphs have capacities
  `1,2,3,4,6,8,12,16,20,24,28,32,40,48`. Live rows still use supported captured
  capacities. The DSpark auxiliary context graphs have their own bounded
  power-of-two capacities (`deepseek_v4_1/nvidia/dspark.py`).
- Sequence-parallel prefill needs equal row counts for reduce-scatter. In
  `deepseek_v4_1/sp_prefill.py::SPRows`, padded rows are
  `ceil(live_rows / TP) * TP`; TP4 can add up to three rows, e.g. 205 → 208.
- Packed quantization scales, scratch buffers and matrix tiles retain their
  backend alignment. Divisibility by four is not divisibility by every tile.
- The vision tower is **still replicated at TP4**. The served
  `deepseek_v4_1/nvidia/vl_model.py` explicitly sets `use_data_parallel = True`
  and distributes images across ranks while keeping the weights replicated.
  Four-way divisibility of its 16 heads does not change this implementation.
- Experimental one-row LM-head duplication for batch-invariant arithmetic
  addresses a different problem. The measured r5o-derived configuration has
  no batch-invariant switch enabled. TP4 alone neither enables that experiment
  nor guarantees deterministic output.

Removing the now-inactive TP3 branches would remove three-node support without
removing runtime padding from TP4. Any further optimization should target a
measured cost, such as the compact MoE path or replicated work, while retaining
the required tail semantics.
