"""Our TileLang indexer behind the module names Apple's DS4.1 sparse-indexer checks import.

The task's checks (its ``sparse_indexer`` contract) call three things:

- ``vllm.model_executor.layers.sparse_indexer_eval.prefill`` / ``decode``: one step of
  the four candidate-consuming indexers (layers 24, 28, 32, 36) over one shared key set
  and one shared candidate pool, from quantized inputs to each consumer's top 512
  request-local positions;
- ``vllm.third_party.deep_gemm_v41_sparse``: the sparse MQA-logits operations (scores of
  the candidate blocks' positions) and their metadata;
- ``vllm.model_executor.kernels.attention.dsa.candidate_blocks.select_candidate_blocks``:
  the candidate source's pool (the top 2,048 blocks of 8 by block maximum, the newest
  block always kept).

Each runs the kernels serving runs: ``IndexerV41Plan(16384, page, candidates=True)`` (the
reindex layers' score and top-k, chunked by the rows serving gives one call) and the
source plan's block selection ``IndexerTopK(width, 2048, 8)``. The adaptation is only
argument layout:

- a candidate pool of block ids becomes the candidate positions the source publishes in
  serving (its blocks ascending, expanded, positions past the row's length dropped);
- packed keys become index-cache pages, and packed rows, whose starts need not be
  page-aligned, address them by absolute position (their selections are mapped back to
  request-local positions);
- the source's FP32 logits are rounded to BF16, the dtype the source scores in.

``install()`` registers the three modules in ``sys.modules``.
"""

import functools
import sys
import types

import torch
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    INDEX_TOPK,
    IndexerTopK,
    IndexerV41Plan,
    page_bytes,
)

WIDTH = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
PACKED_PAGE = 64
PREFILL_CHUNK = 256  # DeepseekV4Attention.INDEX_CHUNK
DECODE_CHUNK = 64  # DeepseekV4Attention.DECODE_CHUNK
CONSUMERS = 4


@functools.cache
def _consumer_plan(page_size):
    return IndexerV41Plan(WIDTH, page_size, candidates=True)


SOURCE_WIDTH = 32768  # the source plan's width is the context's; rows use a prefix of it


@functools.cache
def _block_select(width):
    return IndexerTopK(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)


def _active(device):
    return torch.full((1,), WIDTH, dtype=torch.int32, device=device)


def _positions(blocks, lengths, shift=None):
    """Candidate positions of block ids: valid blocks (``0 <= b``, ``8 b < length``)
    ascending, expanded, positions at or past the length dropped. Invalid entries are a
    suffix (only the newest block can be partial and it sorts last), so the list needs no
    compaction. ``shift`` ([R]) is added to every position. Returns positions [R, 16384]
    int32 (``-1`` padded) and their counts [R]."""
    rows = blocks.shape[0]
    lengths = lengths.to(torch.int64)
    b = blocks.to(torch.int64)
    valid = (b >= 0) & (b * CANDIDATE_BLOCK < lengths[:, None])
    never = torch.iinfo(torch.int32).max
    key = torch.sort(torch.where(valid, b, never), dim=1).values
    offsets = torch.arange(CANDIDATE_BLOCK, device=blocks.device)
    pos = key[:, :, None] * CANDIDATE_BLOCK + offsets
    ok = (key[:, :, None] != never) & (pos < lengths[:, None, None])
    if shift is not None:
        pos = pos + shift.to(torch.int64)[:, None, None]
    pos = torch.where(ok, pos, -1).view(rows, WIDTH).to(torch.int32)
    return pos, ok.view(rows, WIDTH).sum(1, dtype=torch.int32)


def _packed_pool(k_values, k_scales):
    """Packed keys ([N, 64] int8, [N] int32 scale words) as index-cache pages of
    ``PACKED_PAGE`` positions, and the identity page table's width."""
    n = k_values.shape[0]
    pages = max(1, -(-n // PACKED_PAGE))
    values = torch.zeros((pages * PACKED_PAGE, 64), dtype=torch.uint8, device=k_values.device)
    scales = torch.zeros((pages * PACKED_PAGE, 4), dtype=torch.uint8, device=k_values.device)
    values[:n] = k_values.view(torch.uint8)
    scales[:n] = k_scales.view(torch.uint8).view(n, 4)
    pool = torch.cat((values.view(pages, -1), scales.view(pages, -1)), dim=1)
    assert pool.shape[1] == page_bytes(PACKED_PAGE)
    return pool, pages


def _table(pages, rows, device):
    return torch.arange(pages, dtype=torch.int32, device=device).repeat(rows, 1)


def _paged_pool(kv_cache):
    """The fused paged cache [pages, page_kv, 1, 68] as the 2-D page view serving passes."""
    page = kv_cache.shape[1]
    return kv_cache.as_strided(
        (kv_cache.shape[0], page_bytes(page)), (kv_cache.stride(0), 1)
    ), page


def _score(plan, q, q_scales, weights, pool, table, candidates, counts, scores):
    """The plan's score kernel alone, with ``IndexerV41Plan.run``'s arguments."""
    rows = q.shape[0]
    nbytes = page_bytes(plan.page_size)
    plan.score(
        q.reshape(rows, 32, 64).view(torch.int8),
        q_scales.reshape(rows, 32).view(torch.uint32),
        weights,
        pool.view(torch.int8)[:, :nbytes],
        pool.view(torch.uint32)[:, : nbytes // 4],
        table,
        counts,
        _active(q.device),
        candidates,
        counts,
        scores,
    )


def _consumers(plan, q, q_scale, weights, pool, table, candidates, counts, out, chunk):
    """The four consumer layers' calls, each in serving's row chunks."""
    rows = q.shape[1]
    active = _active(q.device)
    for c in range(CONSUMERS):
        for r0 in range(0, rows, chunk):
            r1 = min(rows, r0 + chunk)
            plan.run(
                q[c, r0:r1].reshape(r1 - r0, 32, 64),
                q_scale[c, r0:r1].reshape(r1 - r0, 32),
                weights[c, r0:r1],
                pool,
                table[r0:r1],
                counts[r0:r1],
                active,
                candidates[r0:r1],
                counts[r0:r1],
                out=out[c, r0:r1],
            )


# ------------------------------------------------------------- the entry points


def prefill(q, q_scale, k_quant, k_scale, weights, k_start, k_end, candidate_blocks, out, *, workspace=None):
    rows = q.shape[1]
    if rows == 0:
        return workspace
    lengths = (k_end - k_start).clamp_min(0)
    candidates, counts = _positions(candidate_blocks, lengths, shift=k_start)
    pool, pages = _packed_pool(k_quant, k_scale)
    table = _table(pages, rows, q.device)
    _consumers(_consumer_plan(PACKED_PAGE), q, q_scale, weights, pool, table, candidates, counts, out,
               PREFILL_CHUNK)
    # Absolute positions back to request-local ones.
    out.copy_(torch.where(out >= 0, out - k_start.view(1, rows, 1), -1))
    return workspace


def decode(q, q_scale, kv_cache, weights, context_lens, block_table, row_indices, candidate_blocks, out, *,
           workspace=None):
    del row_indices  # every row carries its own page-table row
    rows = q.shape[1]
    if rows == 0:
        return workspace
    pool, page = _paged_pool(kv_cache)
    candidates, counts = _positions(candidate_blocks, context_lens)
    _consumers(_consumer_plan(page), q, q_scale, weights, pool, block_table.contiguous(), candidates,
               counts, out, DECODE_CHUNK)
    return workspace


# ------------------------------------------------------- the logits operations


def get_sparse_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, num_kv_tokens, sparse_kv_block_indices,
                                   qk_dtype, sparse_block_kv, use_unaligned_ks=False):
    assert qk_dtype == torch.int8 and sparse_block_kv == CANDIDATE_BLOCK
    ks = cu_seq_len_k_start.to(torch.int64)
    ke = cu_seq_len_k_end.to(torch.int64)
    shift = ks % CANDIDATE_BLOCK if use_unaligned_ks else torch.zeros_like(ks)
    offsets = torch.arange(CANDIDATE_BLOCK, device=ks.device)
    pos = sparse_kv_block_indices.to(torch.int64)[:, :, None] * CANDIDATE_BLOCK + shift[:, None, None] + offsets
    ok = (pos >= ks[:, None, None]) & (pos < ke[:, None, None]) & (pos < num_kv_tokens)
    rows = pos.shape[0]
    return {
        "positions": torch.where(ok, pos, -1).view(rows, -1).to(torch.int32),
        "unaligned": bool(use_unaligned_ks),
    }


def get_paged_sparse_mqa_logits_metadata(context_lens, block_table, indices, page_kv, sparse_kv_block_indices,
                                         qk_dtype, sparse_block_kv):
    del indices  # the block table already holds each row's pages
    assert qk_dtype == torch.int8 and sparse_block_kv == CANDIDATE_BLOCK
    offsets = torch.arange(CANDIDATE_BLOCK, device=context_lens.device)
    pos = sparse_kv_block_indices.to(torch.int64)[:, :, None] * CANDIDATE_BLOCK + offsets
    ok = pos < context_lens.to(torch.int64)[:, None, None]
    rows = pos.shape[0]
    return {
        "positions": torch.where(ok, pos, -1).view(rows, -1).to(torch.int32),
        "table": block_table.contiguous(),
        "page_kv": int(page_kv),
    }


def _logits(plan, q, weights, pool, table, positions, chunk):
    q_values, q_scales = q
    rows = q_values.shape[0]
    logits = torch.empty((rows, WIDTH), dtype=torch.bfloat16, device=q_values.device)
    counts = torch.full((rows,), positions.shape[1], dtype=torch.int32, device=q_values.device)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        _score(plan, q_values[r0:r1], q_scales[r0:r1], weights[r0:r1], pool, table[r0:r1],
               positions[r0:r1], counts[r0:r1], logits[r0:r1])
    return logits


def fp8_fp4_sparse_mqa_logits(q, kv, weights, metadata, num_max_sparse_blocks, sparse_block_kv,
                              use_unaligned_ks=False):
    assert num_max_sparse_blocks * sparse_block_kv == WIDTH
    assert metadata["unaligned"] == bool(use_unaligned_ks)
    pool, pages = _packed_pool(*kv)
    rows = q[0].shape[0]
    return _logits(_consumer_plan(PACKED_PAGE), q, weights, pool, _table(pages, rows, q[0].device),
                   metadata["positions"], PREFILL_CHUNK)


def fp8_fp4_paged_sparse_mqa_logits(q, kv_cache, weights, metadata, num_max_sparse_blocks, sparse_block_kv):
    assert num_max_sparse_blocks * sparse_block_kv == WIDTH
    pool, page = _paged_pool(kv_cache)
    assert page == metadata["page_kv"]
    return _logits(_consumer_plan(page), q, weights, pool, metadata["table"], metadata["positions"],
                   DECODE_CHUNK)


# ------------------------------------------------------------ the source pool


def select_candidate_blocks(logits, row_ks, row_ke, topk_blocks, block_size, out, row_repeat=1):
    """The source's pool from its scores: the top ``topk_blocks`` blocks by maximum, the
    newest kept, as block ids (``-1`` padded) in ``out``."""
    assert topk_blocks == CANDIDATE_BLOCKS and block_size == CANDIDATE_BLOCK
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    ke = row_ke.to(torch.int64).repeat_interleave(row_repeat)[:rows]
    if row_ks is None:
        scores, lengths = logits.to(torch.bfloat16).contiguous(), ke
    else:
        ks = row_ks.to(torch.int64).repeat_interleave(row_repeat)[:rows]
        cols = (ks[:, None] + torch.arange(width, device=logits.device)).clamp(max=width - 1)
        scores = torch.gather(logits, 1, cols).to(torch.bfloat16).contiguous()
        lengths = ke - ks
    lengths = lengths.clamp(0, width).to(torch.int32)
    planned = -(-width // SOURCE_WIDTH) * SOURCE_WIDTH
    padded = torch.full((rows, planned), float("-inf"), dtype=torch.bfloat16, device=logits.device)
    padded[:, :width] = scores
    scores, width = padded, planned
    positions = torch.empty((rows, CANDIDATE_BLOCKS * CANDIDATE_BLOCK), dtype=torch.int32, device=logits.device)
    values = torch.empty((rows, CANDIDATE_BLOCKS), dtype=torch.bfloat16, device=logits.device)
    counts = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    active = torch.full((1,), width, dtype=torch.int32, device=logits.device)
    _block_select(width).run(scores, lengths, active, positions, values, counts=counts)
    first = positions.view(rows, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)[:, :, 0]
    out.copy_(torch.where(first >= 0, first // CANDIDATE_BLOCK, -1))


# ---------------------------------------------------------------- registration


def _module(name, **functions):
    module = types.ModuleType(name)
    module.__dict__.update(functions)
    sys.modules[name] = module
    return module


def install():
    _module("vllm.model_executor.layers.sparse_indexer_eval", prefill=prefill, decode=decode)
    _module(
        "vllm.third_party.deep_gemm_v41_sparse",
        get_sparse_mqa_logits_metadata=get_sparse_mqa_logits_metadata,
        get_paged_sparse_mqa_logits_metadata=get_paged_sparse_mqa_logits_metadata,
        fp8_fp4_sparse_mqa_logits=fp8_fp4_sparse_mqa_logits,
        fp8_fp4_paged_sparse_mqa_logits=fp8_fp4_paged_sparse_mqa_logits,
    )
    _module(
        "vllm.model_executor.kernels.attention.dsa.candidate_blocks",
        select_candidate_blocks=select_candidate_blocks,
    )


assert INDEX_TOPK == 512
