# DSpark / DFlash Draft-Step Host Fast Path During Full CUDA Graph Replay

## Hypothesis & Objective

During `DFlashSpeculator.propose` when replaying captured context and query CUDA graphs (`use_context_graph=True` and `batch_desc.cg_mode == CUDAGraphMode.FULL`), several per-step CPU operations execute even though their results are unused by the captured graphs:

1. **Skip `build_slot_mappings_by_layer` on `CUDAGraphMode.FULL` replay**:
   `build_slot_mappings_by_layer` constructs a 63-entry Python dictionary of sliced tensor views across all target and draft layers. When `batch_desc.cg_mode == CUDAGraphMode.FULL`, `self.query_cudagraph_manager.run_fullgraph(batch_desc)` replays the captured graph directly from `self.block_tables.slot_mappings` and never reads `draft_slot_mappings_by_layer`.

2. **Skip `_build_uniform_attn_metadata` and `draft_max_seq_len` reduction on `CUDAGraphMode.FULL` replay when attention builders have no replay side-effects**:
   In `DFlashSpeculator.set_attn`, compute `self._full_cg_needs_attn_metadata_build` (`False` when `draft_cp_size == 1`, `block_tables.cp_size == 1`, `model_state` uses the default no-op `prepare_draft_attn_metadata`, and every draft attention builder is a `FlashAttentionMetadataBuilder` with `aot_schedule=False`, `dcp_world_size == 1`, `rswa_window is None`, and `mm_prefix_query_ranges_np is None`). When `False` (the standard DeepSeek V4.1 DSpark configuration), `CUDAGraphMode.FULL` replay skips computing `max_seq_len` on CPU and skips `_build_uniform_attn_metadata`.

3. **Skip slicing `context_positions` and `context_slots` when `use_context_graph` is True**:
   `_context_preparer.run(aux_hidden_states, num_target_tokens)` replays against its captured buffers (`self.context_positions`, `self._context_slot_mappings`) and never reads the sliced `context_positions` or per-layer `context_slots` list.

4. **Pre-allocate persistent CPU buffers in `DraftModelSpeculator`**:
   Pre-allocate `self._query_start_loc_cpu` and `self._draft_seq_lens_cpu_upper_bound` in `DraftModelSpeculator.__init__` instead of allocating new CPU tensors on every call to `_build_attn_metadata`, and pass `max_query_len=num_query_per_req` directly from `_build_uniform_attn_metadata` to avoid a per-step CPU diff/max reduction.

## Method & Variants

Patch: [`dspark-draft-step-host-fastpath.patch`](./dspark-draft-step-host-fastpath.patch) against `.work/upstreams/vllm` (`d5df88d83`).

Modified files in vLLM:
- `vllm/v1/worker/gpu/spec_decode/speculator.py`
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
- `tests/v1/spec_decode/test_dflash_prepare_inputs.py`

## Verification Run

Verified on `dgx4` inside `spark-ds41f:1030`:

```bash
pytest tests/v1/spec_decode/test_dflash_prepare_inputs.py \
  tests/v1/spec_decode/test_dflash_causality.py \
  tests/v1/spec_decode/test_dflash_dcp.py -v
```

Result: `25 passed`.
