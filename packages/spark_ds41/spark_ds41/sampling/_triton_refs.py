# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test and benchmark reference kernels superseded by TileLang in the V2 GPU worker.

Active V2 worker modules do not import this module during normal startup or step
execution; each worker module exposes its historical ``_..._kernel`` attributes
lazily via ``__getattr__`` for unit tests and kernel benchmarks.
"""

import torch
from torch._inductor.runtime.triton_helpers import libdevice

from vllm.triton_utils import HAS_TRITON, tl, triton
try:
    from vllm.v1.core.boundary_checkpoint import (
        INSTRUCTION_CHECKPOINT_SLOT,
        PREFILL_TAIL_CHECKPOINT_SLOT,
        PROMPT_CHECKPOINT_SLOT,
        RESPONSE_CHECKPOINT_SLOT,
    )
except ImportError:
    PROMPT_CHECKPOINT_SLOT = 0
    INSTRUCTION_CHECKPOINT_SLOT = 1
    PREFILL_TAIL_CHECKPOINT_SLOT = 2
    RESPONSE_CHECKPOINT_SLOT = 3
from spark_ds41.sampling.gumbel import (
    gumbel_block_argmax,
    gumbel_noised_scores,
)

_PROMPT_CHECKPOINT_SLOT = (
    tl.constexpr(PROMPT_CHECKPOINT_SLOT) if HAS_TRITON else PROMPT_CHECKPOINT_SLOT
)
_RESPONSE_CHECKPOINT_SLOT = (
    tl.constexpr(RESPONSE_CHECKPOINT_SLOT) if HAS_TRITON else RESPONSE_CHECKPOINT_SLOT
)
_INSTRUCTION_CHECKPOINT_SLOT = (
    tl.constexpr(INSTRUCTION_CHECKPOINT_SLOT)
    if HAS_TRITON
    else INSTRUCTION_CHECKPOINT_SLOT
)
_PREFILL_TAIL_CHECKPOINT_SLOT = (
    tl.constexpr(PREFILL_TAIL_CHECKPOINT_SLOT)
    if HAS_TRITON
    else PREFILL_TAIL_CHECKPOINT_SLOT
)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)


@triton.jit
def _apply_write_kernel(
    output_ptr,  # MULTI_GROUP: ptr-to-ptrs [num_groups]; else: data ptr
    output_stride,  # MULTI_GROUP: ptr-to-strides [num_groups]; else: row stride
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    write_group_ids_ptr,  # [num_writes], used only when MULTI_GROUP
    BLOCK_SIZE: tl.constexpr,
    MULTI_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    if MULTI_GROUP:
        # Each write targets a different output tensor (KV cache group);
        # resolve its base pointer and row stride per write.
        group_id = tl.load(write_group_ids_ptr + pid)
        row_ptr = _load_ptr(output_ptr + group_id, tl.int32)
        row_stride = tl.load(output_stride + group_id)
    else:
        row_ptr = output_ptr
        row_stride = output_stride
    row_ptr += row_idx * row_stride + start_idx

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(row_ptr + block, content, mask=mask)


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)

    for i in tl.range(num_blocks, max_num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    block_sizes,  # [num_kv_cache_groups]
    kernel_block_sizes,  # [num_kv_cache_groups]
    slot_mapping_enabled,  # [num_kv_cache_groups]
    slot_mapping_circular,  # [num_kv_cache_groups]
    group_cp_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    kv_block_size = tl.load(block_sizes + group_id)
    kernel_block_size = tl.load(kernel_block_sizes + group_id)
    mapping_enabled = tl.load(slot_mapping_enabled + group_id)
    circular = tl.load(slot_mapping_circular + group_id)
    group_cp_size = tl.load(group_cp_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    num_blocks = tl.load(group_num_blocks_ptr + req_state_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        token_mask = offset < end_idx
        positions = tl.load(pos + offset, mask=token_mask, other=0)

        if CP_SIZE == 1 or group_cp_size == 1:
            # Common case: Context parallelism is not used.
            local_positions = positions
            is_local = token_mask
        else:
            # Context parallelism is used.
            virtual_block_size = kv_block_size * CP_SIZE
            virtual_block_indices = positions // virtual_block_size
            virtual_block_offsets = positions % virtual_block_size
            is_local = virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = virtual_block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            local_positions = virtual_block_indices * kv_block_size + local_offsets

        block_indices = tl.where(
            mapping_enabled & ~circular, local_positions // kernel_block_size, 0
        )
        block_offsets = local_positions % kernel_block_size
        valid_block = token_mask & (block_indices < num_blocks)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local & valid_block,
            other=0,
        )
        slot_ids = block_numbers * kernel_block_size + block_offsets
        if CP_SIZE != 1 and group_cp_size != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        slot_ids = tl.where(valid_block, slot_ids, PAD_ID)

        slot_ids = tl.where(mapping_enabled, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def cp_local_slot(
    positions,
    block_numbers,
    block_size,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
):
    """Return rank-local KV slots, or PAD_ID for positions not owned by this rank."""
    block_offsets = positions % (block_size * CP_SIZE)
    if CP_SIZE == 1:
        return block_numbers * block_size + block_offsets
    is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
    rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
    remainder = block_offsets % CP_INTERLEAVE
    local_offsets = rounds * CP_INTERLEAVE + remainder
    return tl.where(is_local, block_numbers * block_size + local_offsets, PAD_ID)


@triton.jit
def prepare_boundary_capture(
    req_idx,
    batch_idx,
    num_computed,
    total_len,
    query_start,
    query_len,
    num_sampled,
    num_rejected,
    sampled_tokens_ptr,
    sampled_stride,
    metadata_ptr,
    stop_tokens_ptr,
    seen_ptr,
    capture_tokens_ptr,
    capture_bias_ptr,
    capture_rows_ptr,
    STOP_CAPACITY: tl.constexpr,
    NUM_CAPTURES: tl.constexpr,
    METADATA_WIDTH: tl.constexpr,
):
    metadata_offset = req_idx * METADATA_WIDTH
    enabled = tl.load(metadata_ptr + metadata_offset)
    prompt_capture = 0
    response_capture = 0
    instruction_capture = 0
    prefill_tail_capture = 0
    if enabled:
        prompt_len = tl.load(metadata_ptr + metadata_offset + 1)
        instruction_len = tl.load(metadata_ptr + metadata_offset + 2)
        min_len = tl.load(metadata_ptr + metadata_offset + 3)
        max_len = tl.load(metadata_ptr + metadata_offset + 4)
        num_stops = tl.load(metadata_ptr + metadata_offset + 5)
        prefill_tail_len = tl.load(metadata_ptr + metadata_offset + 6)
        offsets = tl.arange(0, STOP_CAPACITY)
        stops = tl.load(
            stop_tokens_ptr + req_idx * STOP_CAPACITY + offsets,
            offsets < num_stops,
            other=-1,
        )
        stopped = False
        valid = num_sampled
        for i in range(num_sampled):
            token = tl.load(sampled_tokens_ptr + batch_idx * sampled_stride + i)
            length = total_len + i + 1
            is_stop = (
                tl.sum(((stops == token) & (offsets < num_stops)).to(tl.int32)) > 0
            )
            if not stopped and length >= min_len and (is_stop or length >= max_len):
                valid = i + 1
                stopped = True
        num_rejected += num_sampled - valid
        num_sampled = valid
        computed_after = num_computed + query_len - num_rejected
        if (
            num_computed < prompt_len
            and computed_after == prompt_len
            and tl.load(seen_ptr + req_idx * NUM_CAPTURES + _PROMPT_CHECKPOINT_SLOT)
            == 0
        ):
            prompt_capture = prompt_len
            tl.store(seen_ptr + req_idx * NUM_CAPTURES + _PROMPT_CHECKPOINT_SLOT, 1)
        if (
            instruction_len > 0
            and num_computed < instruction_len
            and computed_after == instruction_len
            and tl.load(
                seen_ptr + req_idx * NUM_CAPTURES + _INSTRUCTION_CHECKPOINT_SLOT
            )
            == 0
        ):
            instruction_capture = instruction_len
            tl.store(
                seen_ptr + req_idx * NUM_CAPTURES + _INSTRUCTION_CHECKPOINT_SLOT, 1
            )
        if (
            prefill_tail_len > 0
            and num_computed < prefill_tail_len
            and computed_after == prefill_tail_len
            and tl.load(
                seen_ptr + req_idx * NUM_CAPTURES + _PREFILL_TAIL_CHECKPOINT_SLOT
            )
            == 0
        ):
            prefill_tail_capture = prefill_tail_len
            tl.store(
                seen_ptr + req_idx * NUM_CAPTURES + _PREFILL_TAIL_CHECKPOINT_SLOT, 1
            )
        if (
            stopped
            and tl.load(seen_ptr + req_idx * NUM_CAPTURES + _RESPONSE_CHECKPOINT_SLOT)
            == 0
        ):
            response_capture = computed_after
            tl.store(seen_ptr + req_idx * NUM_CAPTURES + _RESPONSE_CHECKPOINT_SLOT, 1)

    output_offset = batch_idx * NUM_CAPTURES
    tl.store(
        capture_tokens_ptr + output_offset + _PROMPT_CHECKPOINT_SLOT, prompt_capture
    )
    tl.store(
        capture_tokens_ptr + output_offset + _RESPONSE_CHECKPOINT_SLOT,
        response_capture,
    )
    tl.store(
        capture_tokens_ptr + output_offset + _INSTRUCTION_CHECKPOINT_SLOT,
        instruction_capture,
    )
    tl.store(capture_bias_ptr + output_offset + _PROMPT_CHECKPOINT_SLOT, 0)
    tl.store(
        capture_bias_ptr + output_offset + _RESPONSE_CHECKPOINT_SLOT,
        tl.maximum(num_sampled - 1, 0),
    )
    tl.store(capture_bias_ptr + output_offset + _INSTRUCTION_CHECKPOINT_SLOT, 0)
    tl.store(
        capture_tokens_ptr + output_offset + _PREFILL_TAIL_CHECKPOINT_SLOT,
        prefill_tail_capture,
    )
    tl.store(capture_bias_ptr + output_offset + _PREFILL_TAIL_CHECKPOINT_SLOT, 0)
    tl.store(
        capture_rows_ptr + output_offset + _PREFILL_TAIL_CHECKPOINT_SLOT,
        query_start + prefill_tail_capture - num_computed - 1,
    )
    tl.store(
        capture_rows_ptr + output_offset + _PROMPT_CHECKPOINT_SLOT,
        query_start + prompt_capture - num_computed - 1,
    )
    tl.store(
        capture_rows_ptr + output_offset + _RESPONSE_CHECKPOINT_SLOT,
        query_start + response_capture - num_computed - 1,
    )
    tl.store(
        capture_rows_ptr + output_offset + _INSTRUCTION_CHECKPOINT_SLOT,
        query_start + instruction_capture - num_computed - 1,
    )
    return num_sampled, num_rejected


@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    next_prefill_tokens_stride,
    num_lookahead,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
    LOOKAHEAD_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if num_computed >= prefill_len:
        # Not prefill.
        return

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        tokens = tl.load(request_ptr + num_computed + block, mask=mask)
        tl.store(input_ids_ptr + query_start + block, tokens, mask=mask)

    # Store the next num_lookahead prefill tokens.
    lookahead = tl.arange(0, LOOKAHEAD_BLOCK)
    pos = num_computed + query_len + lookahead
    in_lookahead = lookahead < num_lookahead
    tokens = tl.load(
        request_ptr + pos, mask=in_lookahead & (pos < prefill_len), other=0
    )
    tl.store(
        next_prefill_tokens_ptr
        + lookahead * next_prefill_tokens_stride
        + req_state_idx,
        tokens,
        mask=in_lookahead,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    confidence_ptr,
    confidence_stride,
    is_padding_ptr,
    dead_cut_ptr,
    log_tau,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr = 1,
    DEAD_ROWS: tl.constexpr = False,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    if DEAD_ROWS:
        # Requests without drafts are never clamped.
        tl.store(dead_cut_ptr + batch_idx, BLOCK_SIZE + 1)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - NUM_NEW_SAMPLED_TOKENS

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    first_logit_seq_pos = seq_len - num_logits
    if NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len:
        # Write the last sampled token ID to input_ids.
        last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
        tl.store(input_ids_ptr + logits_start, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )
        if DEAD_ROWS:
            # Dead rows: the drafts' survival is the running product of this
            # step's confidences. Rows past the first draft below the
            # threshold become padding, which routed MoE skips; the sampled
            # fix-up then commits at most one token past the last live row.
            confidence = tl.load(
                confidence_ptr + req_state_idx * confidence_stride + block,
                mask=mask,
                other=1.0,
            )
            log_survival = tl.cumsum(tl.log(confidence), axis=0)
            cut = tl.sum(((log_survival >= log_tau) & mask).to(tl.int32), axis=0)
            tl.store(dead_cut_ptr + batch_idx, cut)
            tl.store(
                is_padding_ptr + query_end - num_draft_tokens + block,
                tl.full([BLOCK_SIZE], 1, tl.uint8),
                mask=mask & (block >= cut),
            )


@triton.jit
def _get_num_sampled_and_rejected_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
    dead_cut_ptr,
    DEAD_ROWS: tl.constexpr = False,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    if DEAD_ROWS:
        # Commit only tokens predicted by live rows.
        num_sampled = tl.minimum(num_sampled, tl.load(dead_cut_ptr + batch_idx) + 1)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + batch_idx, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = logits_end - logits_start

    num_rejected = num_logits - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + batch_idx, num_rejected)


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    boundary_metadata_ptr=None,
    boundary_stop_tokens_ptr=None,
    boundary_seen_ptr=None,
    boundary_capture_tokens_ptr=None,
    boundary_capture_bias_ptr=None,
    boundary_capture_rows_ptr=None,
    NUM_CAPTURES: tl.constexpr = 0,
    BOUNDARY_METADATA_WIDTH: tl.constexpr = 0,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    if req_state_idx < 0:
        # Filter rows with negative index entries.
        if boundary_capture_tokens_ptr is not None:
            for kind in range(NUM_CAPTURES):
                tl.store(boundary_capture_tokens_ptr + req_id * NUM_CAPTURES + kind, 0)
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + req_id)
    num_rejected = tl.load(num_rejected_ptr + req_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    if query_start_loc_ptr is None:
        query_start = 0
        query_len = 0
    else:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    if boundary_metadata_ptr is not None:
        num_sampled, num_rejected = prepare_boundary_capture(
            req_state_idx,
            req_id,
            num_computed,
            total_len,
            query_start,
            query_len,
            num_sampled,
            num_rejected,
            sampled_tokens_ptr,
            sampled_tokens_stride,
            boundary_metadata_ptr,
            boundary_stop_tokens_ptr,
            boundary_seen_ptr,
            boundary_capture_tokens_ptr,
            boundary_capture_bias_ptr,
            boundary_capture_rows_ptr,
            STOP_CAPACITY=128,
            NUM_CAPTURES=NUM_CAPTURES,
            METADATA_WIDTH=BOUNDARY_METADATA_WIDTH,
        )
        tl.store(num_sampled_ptr + req_id, num_sampled)
        tl.store(num_rejected_ptr + req_id, num_rejected)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if output_bin_counts_ptr is not None:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


@triton.jit
def _post_update_num_computed_tokens_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
):
    batch_id = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + batch_id)
    query_end = tl.load(query_start_loc_ptr + batch_id + 1)
    query_len = query_end - query_start

    req_state_idx = tl.load(idx_mapping_ptr + batch_id)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + query_len)


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)


@triton.jit
def _num_nans_kernel(
    logits_ptr,
    logits_stride,
    num_nans_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_nans = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_stride + block, mask=mask, other=0
        )
        logits = logits.to(tl.float32)
        is_nan = libdevice.isnan(logits).to(tl.int1)
        num_nans += tl.sum(is_nan).to(tl.int32)
    tl.store(num_nans_ptr + req_idx, num_nans)


@triton.jit
def _bad_words_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    bad_word_token_ids_ptr,
    bad_word_token_ids_stride,
    bad_word_offsets_ptr,
    bad_word_offsets_stride,
    num_bad_words_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    bw_idx = tl.program_id(1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    num_bad_words = tl.load(num_bad_words_ptr + req_state_idx)

    if bw_idx >= num_bad_words:
        return

    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos

    bd_offsets_base = bad_word_offsets_ptr + req_state_idx * bad_word_offsets_stride
    bd_tokens_base = bad_word_token_ids_ptr + req_state_idx * bad_word_token_ids_stride
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len

    start = tl.load(bd_offsets_base + bw_idx)
    end = tl.load(bd_offsets_base + bw_idx + 1)
    bad_word_len = end - start
    prefix_len = bad_word_len - 1

    if prefix_len > effective_len:
        return

    last_token = tl.load(bd_tokens_base + end - 1)
    match = 1
    for i in range(prefix_len):
        expected = tl.load(bd_tokens_base + start + i)
        actual_pos = effective_len - prefix_len + i

        from_spec_input = actual_pos >= output_len
        if from_spec_input:
            # input_ids at local position 0 is the last committed token;
            # draft tokens start at local position 1.
            spec_offset = actual_pos - output_len
            actual = tl.load(input_ids_ptr + cur_req_first_pos + spec_offset + 1)
        else:
            actual = tl.load(output_base + actual_pos)

        match = match & (expected == actual)

    if match:
        tl.store(logits_ptr + token_idx * logits_stride + last_token, -float("inf"))


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        vocab_size,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask
        )

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(
                restore_when_all_masked_ptr + req_state_idx
            )
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    logits = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(logits, axis=0))

                if row_max == -float("inf"):
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        stop_logits,
                        mask=mask
                        & (stop_logits > -float("inf"))
                        & (stop_logits < float("inf")),
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )


@triton.jit(do_not_specialize=["topk"])
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


@triton.jit
def _fill_logprob_token_ids_kernel(
    # [batch_size, 1 + num_cols]
    out_token_ids_ptr,
    out_token_ids_stride,
    # [batch_size, 1 + num_cols]
    out_valid_mask_ptr,
    out_valid_mask_stride,
    sampled_token_ids_ptr,  # [batch_size]
    topk_indices_ptr,  # [batch_size, NUM_TOPK] (unused when NUM_TOPK == 0)
    topk_indices_stride,
    expanded_idx_mapping_ptr,  # [batch_size] -> req_state_idx
    num_per_req_token_ids_ptr,  # [max_num_reqs]
    per_req_token_ids_ptr,  # [max_num_reqs, MAX_LOGPROB_TOKEN_IDS]
    per_req_token_ids_stride,
    NUM_TOPK: tl.constexpr,
    PADDED_COLS: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Column 0: always the sampled token, always valid.
    sampled = tl.load(sampled_token_ids_ptr + batch_idx)
    tl.store(out_token_ids_ptr + batch_idx * out_token_ids_stride, sampled)
    tl.store(out_valid_mask_ptr + batch_idx * out_valid_mask_stride, 1)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + batch_idx)
    num_custom = tl.load(num_per_req_token_ids_ptr + req_state_idx)

    col = tl.arange(0, PADDED_COLS)
    tid_base = out_token_ids_ptr + batch_idx * out_token_ids_stride + 1
    mask_base = out_valid_mask_ptr + batch_idx * out_valid_mask_stride + 1

    if num_custom > 0:
        # Override topk with per-request custom tokens.
        src = per_req_token_ids_ptr + req_state_idx * per_req_token_ids_stride
        valid = col < num_custom
    else:
        # Fill with topk indices (no-op when NUM_TOPK == 0).
        src = topk_indices_ptr + batch_idx * topk_indices_stride
        valid = col < NUM_TOPK

    tokens = tl.load(src + col, mask=valid, other=0).to(tl.int64)
    tl.store(tid_base + col, tokens, mask=valid)
    tl.store(mask_base + col, tl.full([PADDED_COLS], 1, tl.int1), mask=valid)


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    min_p_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    min_p = tl.load(min_p_ptr + req_state_idx).to(tl.float32)
    if min_p == 0.0:
        return

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    threshold = max_val + tl.log(min_p)
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + block,
            mask=mask,
            other=float("-inf"),
        )
        logits = tl.where(logits < threshold, float("-inf"), logits)
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Accumulate draft token counts from previous positions directly into
    # output_bin_counts (preserves its native tensor layout, avoiding an
    # expensive shared-memory layout conversion after the loop).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    output_bin_counts = base_output_counts
    for prev_pos in tl.range(pos):
        prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
        token_match = block == prev_token
        output_bin_counts = output_bin_counts + token_match.to(tl.int32)
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


@triton.jit
def _bincount_kernel(
    expanded_idx_mapping_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    prefill_len_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if block_idx * BLOCK_SIZE >= prefill_len:
        return

    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if block_idx * BLOCK_SIZE < prompt_len:
        mask = block < prompt_len
        prompt_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        idx = prompt_tokens // 32
        bit_idx = prompt_tokens % 32
        bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx
        tl.atomic_or(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + idx,
            bit,
            mask=mask,
        )

    if (block_idx + 1) * BLOCK_SIZE >= prompt_len:
        mask = block < prefill_len
        mask &= block >= prompt_len
        output_tokens = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + block, mask=mask
        )
        tl.atomic_add(
            output_bin_counts_ptr
            + req_state_idx * output_bin_counts_stride
            + output_tokens,
            1,
            mask=mask,
        )


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


@triton.jit
def _sink_candidates_kernel(
    # [num_reqs, num_blocks * NUM_CANDIDATES]
    values_ptr,
    indices_ptr,
    candidates_stride,
    # [num_logits, logits_stride]: the scores' logits, after sampling params
    logits_ptr,
    logits_stride,
    vocab_size,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, sampled_stride]
    sampled_ptr,
    sampled_stride,
    # [num_logits]: each row's input token
    draft_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    seeds_ptr,
    # [num_logits]
    pos_ptr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
    MASK_REJECTED_DRAFT: tl.constexpr,
):
    req = tl.program_id(0)
    block_idx = tl.program_id(1)
    start = tl.load(cu_num_logits_ptr + req)
    end = tl.load(cu_num_logits_ptr + req + 1)
    local = tl.maximum(tl.load(num_sampled_ptr + req) - 1, 0)
    # The row that produced the request's last sampled token.
    row = (start + local).to(tl.int64)
    req_state_idx = tl.load(idx_mapping_ptr + req).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    scores = tl.load(
        logits_ptr + row * logits_stride + block, mask=mask, other=float("-inf")
    ).to(tl.float32)
    if MASK_REJECTED_DRAFT:
        # A random row that rejected its draft resampled the residual, which
        # for one-hot drafts is the target without the draft.
        is_bonus = row == end - 1
        verified = tl.load(draft_ptr + row + 1, mask=not is_bonus, other=-1)
        emitted = tl.load(sampled_ptr + req * sampled_stride + local)
        rejected = (
            (not is_bonus) & (verified >= 0) & (verified != emitted) & (temp != 0.0)
        )
        scores = tl.where(rejected & (block == verified), float("-inf"), scores)
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + row)
    scores = gumbel_noised_scores(
        scores,
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    offsets = tl.arange(0, BLOCK_SIZE)
    out = req * candidates_stride + block_idx * NUM_CANDIDATES
    for i in tl.static_range(NUM_CANDIDATES):
        value, idx = tl.max(scores, axis=0, return_indices=True)
        tl.store(values_ptr + out + i, value)
        tl.store(indices_ptr + out + i, block_idx * BLOCK_SIZE + idx)
        scores = tl.where(offsets == idx, float("-inf"), scores)


@triton.jit
def _sink_policy_kernel(
    # Verification results, updated in place for rejected draft zeros.
    # [num_reqs, SAMPLED_WIDTH] int64, [num_reqs] int32, [num_reqs] int32
    sampled_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Host outputs: [num_reqs, SAMPLED_WIDTH + 1] int64, [num_reqs] int32,
    # [num_reqs, NUM_EVENT_FIELDS] int32, [num_reqs] float32 / int64
    out_tokens_ptr,
    num_out_ptr,
    events_ptr,
    recovered_logprobs_ptr,
    recovered_ranks_ptr,
    # Batch.
    cu_num_logits_ptr,
    idx_mapping_ptr,
    # [num_reqs] int8: the request was decoding (row 0's input is draft zero)
    decoding_ptr,
    # [num_logits] int8, [num_logits] float32 (each row's largest |column|),
    # [num_logits]
    flags_ptr,
    values_ptr,
    draft_ptr,
    # This step's alternates: [num_reqs, NUM_ALTERNATES]
    new_alternates_ptr,
    new_alternate_logprobs_ptr,
    new_alternate_ranks_ptr,
    # Per-slot state: [max_num_reqs, NUM_ALTERNATES], [max_num_reqs]
    alternates_ptr,
    alternate_logprobs_ptr,
    alternate_ranks_ptr,
    tries_ptr,
    # [max_num_reqs] int32, [max_num_reqs, vocab] int32
    total_len_ptr,
    bin_counts_ptr,
    bin_counts_stride,
    # [3] int64: this rank's flagged rows per SinkRow
    counts_ptr,
    NUM_ALTERNATES: tl.constexpr,
    REJECTION: tl.constexpr,
    LOGPROBS: tl.constexpr,
    SAMPLED_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    PADDED_ALTERNATES: tl.constexpr,
    NUM_EVENT_FIELDS: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu_num_logits_ptr + req)
    num_logits = tl.load(cu_num_logits_ptr + req + 1) - start
    num_sampled = tl.load(num_sampled_ptr + req)
    events = events_ptr + req * NUM_EVENT_FIELDS
    if num_sampled == 0:
        # A prefill chunk: its row is not sampled.
        tl.store(num_out_ptr + req, 0)
        tl.store(events + 0, 0)
        tl.store(events + 1, 2)
        tl.store(events + 2, 0)
        tl.store(events + 3, -1)
        tl.store(events + 4, 0)
        tl.store(events + 5, -1)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req).to(tl.int64)
    decoding = tl.load(decoding_ptr + req) != 0
    row0_flag = tl.load(flags_ptr + start) != 0
    rows = tl.arange(0, PADDED_WIDTH)
    row_flags = tl.load(flags_ptr + start + rows, mask=rows < num_logits, other=0) != 0
    # Drafter rows whose inputs were accepted (1 .. num_sampled - 1), and the
    # row of the first rejected draft.
    accepted_flags = tl.sum(
        (row_flags & (rows >= 1) & (rows < num_sampled)).to(tl.int32), axis=0
    )
    cut = (tl.sum((row_flags & (rows == num_sampled)).to(tl.int32), axis=0) > 0) & (
        num_sampled < num_logits
    )
    draft_zero_flag = decoding & row0_flag
    # Prompt row flagged: FLAGGED; it cannot be replaced.
    outcome = tl.where((not decoding) & row0_flag, 1, 0)
    out_row = out_tokens_ptr + req * (SAMPLED_WIDTH + 1)
    sampled_row = sampled_ptr + req * SAMPLED_WIDTH
    if REJECTION:
        tries = tl.load(tries_ptr + req_state_idx)
        slot = tl.minimum(tries, NUM_ALTERNATES - 1)
        state = req_state_idx * NUM_ALTERNATES
        alternate = tl.load(alternates_ptr + state + slot)
        rejected = draft_zero_flag & (tries < NUM_ALTERNATES) & (alternate >= 0)
        outcome = tl.where(rejected, 2, tl.where(draft_zero_flag, 3, outcome))
        drafter_flags = tl.where(rejected, 0, accepted_flags + cut.to(tl.int32))
        cut_row = tl.where(cut & (not rejected), num_sampled, -1)
        if rejected:
            # Draft zero is replaced by its next alternate; the request's
            # position does not advance (all of its rows are rejected).
            draft_zero = tl.load(draft_ptr + start).to(tl.int64)
            tl.store(sampled_row, alternate)
            tl.store(num_sampled_ptr + req, 1)
            tl.store(num_rejected_ptr + req, num_logits)
            tl.store(tries_ptr + req_state_idx, tries + 1)
            if LOGPROBS:
                tl.store(
                    recovered_logprobs_ptr + req,
                    tl.load(alternate_logprobs_ptr + state + slot),
                )
                tl.store(
                    recovered_ranks_ptr + req,
                    tl.load(alternate_ranks_ptr + state + slot),
                )
            # The request state already holds draft zero as its last token:
            # take it out, so the update appends the alternate in its place.
            tl.store(
                total_len_ptr + req_state_idx,
                tl.load(total_len_ptr + req_state_idx) - 1,
            )
            count_ptr = bin_counts_ptr + req_state_idx * bin_counts_stride + draft_zero
            tl.store(count_ptr, tl.load(count_ptr) - 1)
            tl.store(out_row, alternate)
            tl.store(num_out_ptr + req, 1)
        else:
            # The last sampled token's draw provides the next alternates.
            a = tl.arange(0, PADDED_ALTERNATES)
            in_range = a < NUM_ALTERNATES
            new = req * NUM_ALTERNATES + a
            tl.store(
                alternates_ptr + state + a,
                tl.load(new_alternates_ptr + new, mask=in_range),
                mask=in_range,
            )
            if LOGPROBS:
                tl.store(
                    alternate_logprobs_ptr + state + a,
                    tl.load(new_alternate_logprobs_ptr + new, mask=in_range),
                    mask=in_range,
                )
                tl.store(
                    alternate_ranks_ptr + state + a,
                    tl.load(new_alternate_ranks_ptr + new, mask=in_range),
                    mask=in_range,
                )
            tl.store(tries_ptr + req_state_idx, 0)
            # Draft zero first, then the step's sampled tokens.
            shift = decoding.to(tl.int32)
            tokens = tl.load(sampled_row + rows, mask=rows < num_sampled)
            tl.store(out_row + shift + rows, tokens, mask=rows < num_sampled)
            if decoding:
                tl.store(out_row, tl.load(draft_ptr + start).to(tl.int64))
            tl.store(num_out_ptr + req, num_sampled + shift)
    else:
        outcome = tl.where(draft_zero_flag, 1, outcome)
        drafter_flags = accepted_flags
        cut_row = -1
    # The largest flagged value among the rows the step judged (up to the
    # first rejected draft) and its row, for the log line.
    judged = row_flags & (rows <= num_sampled)
    row_values = tl.load(values_ptr + start + rows, mask=rows < num_logits, other=0.0)
    value, value_row = tl.max(
        tl.where(judged, row_values, -1.0), axis=0, return_indices=True
    )
    flagged_any = value >= 0.0
    tl.store(events + 0, outcome)
    tl.store(events + 1, tl.where(decoding, 0, 2))
    tl.store(events + 2, drafter_flags)
    tl.store(events + 3, cut_row)
    tl.store(events + 4, tl.where(flagged_any, value, 0.0).to(tl.int32))
    tl.store(events + 5, tl.where(flagged_any, value_row, -1))
    tl.atomic_add(counts_ptr + 0, draft_zero_flag.to(tl.int64))
    tl.atomic_add(counts_ptr + 1, drafter_flags.to(tl.int64))
    tl.atomic_add(counts_ptr + 2, ((not decoding) & row0_flag).to(tl.int64))


@triton.jit
def _load_effective_token(
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    cur_req_first_pos,
    req_state_idx,
    total_len,
    pos,
):
    if pos < total_len:
        return tl.load(all_token_ids_ptr + req_state_idx * all_token_ids_stride + pos)
    # In decode/spec-decode, input_ids at local position 0 is the already
    # committed last sampled token. Effective draft-prefix positions start at
    # local position 1.
    input_pos = cur_req_first_pos + pos - total_len + 1
    return tl.load(input_ids_ptr + input_pos)


@triton.jit
def _update_committed_marker_cache_kernel(
    req_ids_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    cached_scan_pos_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    scan_pos = tl.load(cached_scan_pos_ptr + req_state_idx)
    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    if scan_pos > total_len:
        scan_pos = 0
        last_start = -1
        last_end = -1

    if scan_pos == 0 and last_start < 0 and last_end < 0:
        # Cold scan matching the reasoning parser state machine: find the latest
        # natural end marker and the earliest start marker strictly after it.
        block_hi = total_len
        while block_hi > 0 and last_end < 0 and (budget > 0 or last_start < 0):
            block_lo = block_hi - BLOCK
            if block_lo < 0:
                block_lo = 0
            offs = block_lo + tl.arange(0, BLOCK)

            end_match = (offs < block_hi) & (offs + NATURAL_END_LEN <= total_len)
            for j in tl.static_range(0, NATURAL_END_LEN):
                expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                end_match = end_match & (actual == expected)
            chunk_end = tl.max(tl.where(end_match, offs, -1), axis=0)

            start_match = (
                (offs < block_hi)
                & (offs > chunk_end)
                & (offs + START_LEN <= total_len)
            )
            for j in tl.static_range(0, START_LEN):
                expected = tl.load(reasoning_start_token_ids_ptr + j)
                actual = tl.load(
                    all_token_ids_ptr + req_state_idx * all_token_ids_stride + offs + j,
                    mask=offs + j < total_len,
                    other=-1,
                )
                start_match = start_match & (actual == expected)
            chunk_start = tl.min(tl.where(start_match, offs, 2147483647), axis=0)
            if chunk_start < 2147483647:
                last_start = chunk_start
            if chunk_end >= 0:
                last_end = chunk_end
            block_hi = block_lo
    else:
        for end_pos in tl.range(scan_pos + 1, total_len + 1):
            if last_start >= 0 and last_start > last_end:
                i = end_pos - NATURAL_END_LEN
                if i >= last_start + START_LEN:
                    end_match = True
                    for j in tl.static_range(0, NATURAL_END_LEN):
                        expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                        actual = tl.load(
                            all_token_ids_ptr
                            + req_state_idx * all_token_ids_stride
                            + i
                            + j
                        )
                        end_match = end_match & (actual == expected)
                    if end_match:
                        last_end = i
            else:
                i = end_pos - START_LEN
                if i >= 0 and (last_end < 0 or i >= last_end + NATURAL_END_LEN):
                    start_match = True
                    for j in tl.static_range(0, START_LEN):
                        expected = tl.load(reasoning_start_token_ids_ptr + j)
                        actual = tl.load(
                            all_token_ids_ptr
                            + req_state_idx * all_token_ids_stride
                            + i
                            + j
                        )
                        start_match = start_match & (actual == expected)
                    if start_match:
                        last_start = i

    tl.store(cached_last_start_ptr + req_state_idx, last_start)
    tl.store(cached_last_end_ptr + req_state_idx, last_end)
    tl.store(cached_scan_pos_ptr + req_state_idx, total_len)


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    thinking_token_budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    cached_last_start_ptr,
    cached_last_end_ptr,
    reasoning_start_token_ids_ptr,
    natural_reasoning_end_token_ids_ptr,
    reasoning_end_token_ids_ptr,
    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return

    local_pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - local_pos
    total_len = tl.load(total_len_ptr + req_state_idx)
    effective_len = total_len + local_pos

    last_start = tl.load(cached_last_start_ptr + req_state_idx)
    last_end = tl.load(cached_last_end_ptr + req_state_idx)

    for end_pos in tl.range(total_len + 1, effective_len + 1):
        if last_start >= 0 and last_start > last_end:
            i = end_pos - NATURAL_END_LEN
            if i >= last_start + START_LEN:
                end_match = True
                for j in tl.static_range(0, NATURAL_END_LEN):
                    expected = tl.load(natural_reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        i + j,
                    )
                    end_match = end_match & (actual == expected)
                if end_match:
                    last_end = i
        else:
            i = end_pos - START_LEN
            if i >= 0 and (last_end < 0 or i >= last_end + NATURAL_END_LEN):
                start_match = True
                for j in tl.static_range(0, START_LEN):
                    expected = tl.load(reasoning_start_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        i + j,
                    )
                    start_match = start_match & (actual == expected)
                if start_match:
                    last_start = i

    if last_start < 0 or last_start <= last_end:
        return

    reasoning_start = last_start + START_LEN
    # If the request resumes from a prompt that already contains generated
    # reasoning content, count it against the remaining budget.
    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return

    # If the tail already ends with a prefix of the forced end sequence
    # (even from a resumed prompt), continue from the next marker token.
    end_prefix_len = 0
    max_prefix_len = END_LEN - 1
    if effective_len < max_prefix_len:
        max_prefix_len = effective_len

    for prefix_len in tl.static_range(1, END_LEN):
        if prefix_len <= max_prefix_len:
            prefix_match = True
            suffix_start = effective_len - prefix_len
            for j in tl.static_range(0, END_LEN):
                if j < prefix_len:
                    expected = tl.load(reasoning_end_token_ids_ptr + j)
                    actual = _load_effective_token(
                        all_token_ids_ptr,
                        all_token_ids_stride,
                        input_ids_ptr,
                        cur_req_first_pos,
                        req_state_idx,
                        total_len,
                        suffix_start + j,
                    )
                    prefix_match = prefix_match & (actual == expected)
            if prefix_match:
                end_prefix_len = prefix_len

    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)


def _assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    num_steps: int,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place."""
    survival = confidence_probs[idx_mapping].cumprod(dim=1)
    steps = torch.arange(num_steps, device=survival.device)
    # Out-of-range slots score -inf so they never outrank a real draft.
    out_of_range = steps[None, :] >= capacities[:, None]
    survival = survival.masked_fill(out_of_range, -float("inf"))
    flat = survival.flatten()
    winners = flat.topk(draft_budget).indices
    admitted = torch.zeros_like(flat, dtype=torch.bool).index_fill_(0, winners, True)
    torch.sum(admitted.view_as(survival), dim=1, dtype=capacities.dtype, out=capacities)


@triton.jit
def _prepare_dflash_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Sampling params
    temperature_ptr,
    seeds_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    # No valid row: every row was rejected, the first one included (sink
    # rejection replaced draft zero). The anchor, its replacement, takes draft
    # zero's position; the context before it is in the cache already.
    last_valid_pos = tl.where(
        num_valid_ctx > 0,
        tl.load(target_positions_ptr + tl.maximum(valid_ctx_end - 1, ctx_start)),
        tl.load(target_positions_ptr + ctx_start) - 1,
    )
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions / slots ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // (block_size * CP_SIZE)
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction; rejected suffix rows are invalid context as well.
    # Neither kind of row may write draft KV into physical block 0.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    local_ctx_slot = cp_local_slot(
        ctx_pos, ctx_block_id, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD_SLOT_ID
    )
    ctx_slot = tl.where(
        ctx_resident,
        local_ctx_slot,
        PAD_SLOT_ID,
    )
    # Stored over the full [0, num_ctx) span while the loads above are masked to
    # [0, num_valid_ctx): the rejected suffix rows in between get position 0 and
    # PAD_SLOT_ID. That is intentional — those rows write no KV and their
    # positions are never consumed, but the span must stay fully initialized so
    # a replayed graph cannot observe a stale value from an earlier batch.
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // (block_size * CP_SIZE)
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & (q_block_id != 0)
    local_q_slot = cp_local_slot(
        query_pos,
        q_block_id,
        block_size,
        cp_rank,
        CP_SIZE,
        CP_INTERLEAVE,
        PAD_SLOT_ID,
    )
    q_slot = tl.where(
        q_resident,
        local_q_slot,
        PAD_SLOT_ID,
    )

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), so we sample at EVERY query position
    # and each position k predicts the NEXT token (sampled position = query_pos + 1).
    # Otherwise (DFlash default) the anchor is the bonus token and only the mask tokens
    # at offsets > 0 are sampled from, each AT its own position.
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        # seq_lens is the absolute sequence length the draft attention
        # reads up to (context + query), not just the count of accepted
        # tokens this step.
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        # Copy sampling state.
        tl.store(
            out_temperature_ptr + req_state_idx,
            tl.load(temperature_ptr + req_state_idx),
        )
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            # Padded sample slots point at query index 0 (a valid row in
            # last_hidden_states) so CG replay never reads OOB. Padded
            # sample idx mappings point to -1, which is ignored during
            # sampling to prevent writing stale values to draft logits.
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            # Pad query slot mappings past num_query_tokens with PAD so the
            # captured CG sees PAD slots (no K/V write) for replay sizes
            # larger than the current request count.
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


@triton.jit
def _first_argmax(av, ai, bv, bi):
    a_nan = av != av
    b_nan = bv != bv
    take_a = (a_nan & (~b_nan | (ai < bi))) | (
        ~a_nan & ~b_nan & ((av > bv) | ((av == bv) & (ai < bi)))
    )
    return tl.where(take_a, av, bv), tl.where(take_a, ai, bi)


@triton.jit
def _partial_argmax(
    base,
    bias,
    values,
    indices,
    base_stride,
    bias_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(base + row * base_stride + col, col < VOCAB, other=0)
    b = tl.load(bias + row * bias_stride + col, col < VOCAB, other=0)
    # Preserve the materialized torch.add rounding before comparing logits.
    summed = (
        (a.to(tl.float32) + b.to(tl.float32)).to(base.dtype.element_ty).to(tl.float32)
    )
    summed = tl.where(col < VOCAB, summed, -float("inf"))
    idx = tl.where(col < VOCAB, col, 2147483647)
    value, index = tl.reduce((summed, idx), 0, _first_argmax)
    tl.store(values + row * PARTS + part, value)
    tl.store(indices + row * PARTS + part, index)


@triton.jit
def _partial_argmax_logits(
    logits,
    values,
    indices,
    logits_stride,
    VOCAB: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(logits + row * logits_stride + col, col < VOCAB, other=0)
    value = tl.where(col < VOCAB, value.to(tl.float32), -float("inf"))
    idx = tl.where(col < VOCAB, col, 2147483647)
    best, index = tl.reduce((value, idx), 0, _first_argmax)
    tl.store(values + row * PARTS + part, best)
    tl.store(indices + row * PARTS + part, index)


@triton.jit
def _finish_argmax(
    values,
    indices,
    output,
    output_stride,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    _, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(output + row * output_stride, result)


@triton.jit
def _finish_argmax_pair(
    values,
    indices,
    pairs,
    vocab_start,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(values + row * PARTS + part, part < PARTS, other=-float("inf"))
    index = tl.load(indices + row * PARTS + part, part < PARTS, other=2147483647)
    best, result = tl.reduce((value, index), 0, _first_argmax)
    tl.store(pairs + row * 2, best)
    tl.store(pairs + row * 2 + 1, (result + vocab_start).to(tl.float32))


@triton.jit
def _select_first_argmax(
    pairs,
    output,
    output_stride,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    value = tl.load(
        pairs + row * 2 * RANKS + 2 * rank, rank < RANKS, other=-float("inf")
    )
    index = tl.load(
        pairs + row * 2 * RANKS + 2 * rank + 1, rank < RANKS, other=1073741824.0
    )
    _, result = tl.reduce((value, index.to(tl.int32)), 0, _first_argmax)
    tl.store(output + row * output_stride, result)


@triton.jit
def _greedy_rejection_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    target_ids_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        if verifying:
            draft = tl.load(draft_sampled_ptr + start_idx + i + 1).to(tl.int64)
            target = tl.load(target_ids_ptr + start_idx + i)
            # -1 is a placeholder draft that is always rejected.
            accepted = (target == draft) & (draft >= 0)
            tl.store(
                sampled_ptr + req_idx * sampled_stride + i,
                tl.where(accepted, draft, target),
            )
            verifying = accepted
            accepted_length += accepted.to(tl.int64)
    if accepted_length == num_draft_tokens:
        # Every draft accepted: the bonus position's argmax.
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_draft_tokens,
            tl.load(target_ids_ptr + start_idx + num_draft_tokens),
        )
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    resample_token_idx = start_idx + num_sampled
    req_state_idx = tl.load(expanded_idx_mapping_ptr + resample_token_idx)

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + resampled_max_block_idx,
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + num_sampled,
        resampled,
    )


@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    cu_num_logits_ptr,
    input_ids_ptr,
    input_logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    MASK_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    mapping_idx = tl.load(logits_indices_ptr + bitmask_idx)
    if input_ids_ptr is not None:
        invalid_draft = mapping_idx < 0
        mapping_idx = tl.where(invalid_draft, -mapping_idx - 1, mapping_idx)
    req_idx = mapping_idx // MASK_STRIDE
    position_idx = mapping_idx % MASK_STRIDE
    logits_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_req_logits = tl.load(cu_num_logits_ptr + req_idx + 1) - logits_idx
    logits_idx += position_idx
    position_is_active = position_idx < num_req_logits

    # Load the bitmask.
    block_id = tl.program_id(1)
    if input_ids_ptr is not None:  # noqa: SIM102
        if block_id == 0 and position_is_active and invalid_draft:
            input_idx = tl.load(input_logits_indices_ptr + logits_idx)
            tl.store(input_ids_ptr + input_idx, -1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=position_is_active & bitmask & (block_offset < vocab_size),
    )
