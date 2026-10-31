# Consolidate Multi-Group DSpark / DFlash Input Preparation into a Single Kernel Launch

## Hypothesis & Objective

In DeepSeek V4.1 Flash with DSpark (`dflash` / `dspark`), the draft model has three KV-cache groups (`mtp.0`, `mtp.1`, `mtp.2`). Previously, `DFlashSpeculator.propose` looped over `for i, gid in enumerate(self.draft_kv_cache_group_ids):` and invoked `prepare_dflash_inputs` three separate times per step:

- Recomputing `int(input_batch.num_scheduled_tokens.max())` on the host three times per step.
- Launching `prepare_dflash_inputs_kernel` three times on the GPU, redundantly loading all per-request metadata (`idx_mapping`, `target_query_start_loc`, `num_rejected`, `num_sampled`, `target_positions`, `last_sampled`, `next_prefill_tokens`, `temperature`, `seeds`) and redundantly writing all shared outputs (`input_ids`, `positions`, `query_start_loc`, `seq_lens`, `context_positions`, `sample_indices`, `sample_pos`, `sample_idx_mapping`, `temperature`, `seeds`, and their graph-padding tails) three times.

By adding a `multi_group=True` variant of `prepare_dflash_inputs_kernel` (`prepare_dflash_inputs_multi_group`) that reads the pointer/stride/block-size tables (`_draft_query_slot_ptrs`, `_draft_table_ptrs`, `_draft_table_strides`, `_draft_block_sizes`) prebuilt in `DFlashSpeculator.set_attn`, all draft KV-cache groups' context and query slot mappings are written in a single kernel launch while shared request metadata and outputs are read and written only once.

## Method & Variants

Patch: [`dspark-input-prep-launch-consolidation.patch`](./dspark-input-prep-launch-consolidation.patch) against `.work/upstreams/vllm` (`d5df88d83`).

Modified files in vLLM:
- `vllm/v1/worker/gpu/tilelang_inputs.py`
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
- `tests/v1/spec_decode/test_dflash_prepare_inputs.py`

## Verification Run

Verified on `dgx4` inside `spark-ds41f:1030`:

```bash
pytest tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/worker/test_gpu_tilelang_inputs.py \
  -k "dflash or declared_calls" -v
```

Result: `45 passed`.
