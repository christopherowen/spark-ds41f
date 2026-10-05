# Code entry points

Exact definition locations in the frozen copies. These are navigation aids; importable or prepared code is not proof that a branch executes. See the main audit for dispatch conditions. The line numbers remain those of the original files.

| Source | Definition | Line | Frozen file |
|---|---|---:|---|
| V01 | `B12xEmbeddingMethod` | 133 | [b12x_layers.py](sources/vllm/vllm/models/deepseek_v4_1/b12x_layers.py) |
| V01 | `B12xLinearMethod` | 218 | [b12x_layers.py](sources/vllm/vllm/models/deepseek_v4_1/b12x_layers.py) |
| V01 | `B12xFP8LinearMethod` | 304 | [b12x_layers.py](sources/vllm/vllm/models/deepseek_v4_1/b12x_layers.py) |
| V01 | `B12xMHC` | 669 | [b12x_layers.py](sources/vllm/vllm/models/deepseek_v4_1/b12x_layers.py) |
| V01 | `_execution_capacities` | 50 | [b12x_layers.py](sources/vllm/vllm/models/deepseek_v4_1/b12x_layers.py) |
| V02 | `DeepseekCompressor._project` | 324 | [compressor.py](sources/vllm/vllm/models/deepseek_v4_1/compressor.py) |
| V02 | `DeepseekCompressor._projection_unit` | 258 | [compressor.py](sources/vllm/vllm/models/deepseek_v4_1/compressor.py) |
| V03 | `DeepseekV4Indexer.__init__` | 336 | [attention.py](sources/vllm/vllm/models/deepseek_v4_1/attention.py) |
| V13 | `LogitsProcessor._apply_head` | 287 | [logits_processor.py](sources/vllm/vllm/model_executor/layers/logits_processor.py) |
| V25 | `Sampler` | 33 | [sampler.py](sources/vllm/vllm/v1/worker/gpu/sample/sampler.py) |
| V31 | `VocabParallelEmbedding.__init__` | 324 | [vocab_parallel_embedding.py](sources/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py) |
| V31 | `VocabParallelEmbedding.forward` | 649 | [vocab_parallel_embedding.py](sources/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py) |
| V31 | `VocabParallelEmbedding.forward_reduce_scatter` | 696 | [vocab_parallel_embedding.py](sources/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py) |
| B01 | `default_config` | 80 | [_tuning.py](sources/b12x/b12x/gemm/bf16_gemv/_tuning.py) |
| B06 | `make_plan` | 50 | [_preparation.py](sources/b12x/b12x/gemm/bf16_vocab_projection/_preparation.py) |
| B08 | `dense_query` | 29 | [_tuning.py](sources/b12x/b12x/gemm/block_fp8_linear/_tuning.py) |
| B10 | `_dense_gemm_policy_for` | 410 | [dense_gemm.py](sources/b12x/b12x/_lib/dense_gemm.py) |
| B10 | `_select_block_fp8_decode_slices` | 564 | [dense_gemm.py](sources/b12x/b12x/_lib/dense_gemm.py) |
| B10 | `_select_default_dense_gemm_plan` | 7496 | [dense_gemm.py](sources/b12x/b12x/_lib/dense_gemm.py) |
| B11 | `_projection_default_config` | 115 | [_tuning.py](sources/b12x/b12x/norm/mhc/_tuning.py) |
| B11 | `_default_config` | 198 | [_tuning.py](sources/b12x/b12x/norm/mhc/_tuning.py) |
| D05 | `family_fp8` | 175 | [transition_map.py](sources/deployment/experiments/2026-09-29-determinism/transition_map.py) |
| D05 | `family_head` | 416 | [transition_map.py](sources/deployment/experiments/2026-09-29-determinism/transition_map.py) |
| D05 | `family_wo` | 456 | [transition_map.py](sources/deployment/experiments/2026-09-29-determinism/transition_map.py) |
| D11 | `step_times_ms` | 3147 | [spark3](sources/deployment/bin/spark3) |
