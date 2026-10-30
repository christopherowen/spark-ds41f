# DeepSeek V4.1 Dead Optionality Cleanup & Multi-Rank DSpark TP Defaults

## Summary

1. **Remove Dead Static Sequence-Parallel & `use_mega_moe` Branches in V4.1:**
   - V4.1 exclusively uses per-forward prefill sequence parallelism (`sp_prefill.plan_rows`); `_use_sequence_parallel(vllm_config)` in `vllm/models/deepseek_v4_1/nvidia/model.py` was hardcoded to `False`, and `self.use_mega_moe` was hardcoded to `False`.
   - Removes `_use_sequence_parallel`, `self.use_sequence_parallel`, `self.use_mega_moe`, and the unreachable `sp_shard`, `sp_all_gather`, and `sp_padding_mask` branches from `DeepseekV4DecoderLayer`, `DeepseekV4Model`, and `DSparkDeepseekV4Model`, simplifying the hot forward paths of both the target and draft models.
   - Leaves `SPARK_DS41F_INDEXER_SPLIT_CHECK` (`_INDEX_SPLIT_CHECK`) in `vllm/models/deepseek_v4_1/attention.py` completely intact.
2. **Default Multi-Rank DSpark TP Optimizations to Enabled:**
   - Defaults `SPARK_DS41F_DSPARK_VOCAB_PARALLEL` (`_vocab_parallel_greedy()`) and `SPARK_DS41F_DSPARK_MAIN_PROJ_TP` (`main_proj_tp`) to `"1"` when `get_tensor_model_parallel_world_size() > 1`, so multi-rank TP deployments automatically use vocabulary-parallel greedy Markov drafting and column-parallel `main_proj` without requiring manual environment flags (while preserving `=0` override capability).

## Verification (`dgx4`, `spark-ds41f:1030`)

- `pytest tests/models/test_deepseek_v4_1_ced_model.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/models/test_deepseek_v4_1_engram.py`: **112 passed**.
