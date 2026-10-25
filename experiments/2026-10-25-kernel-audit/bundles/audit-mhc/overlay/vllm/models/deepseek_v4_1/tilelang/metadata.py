# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang sparse-MLA metadata for DeepSeek-V4.1: integer kernels, exact.

The same contracts as the Triton kernels they replace (``sparse_mla._requests``,
``_tokens``, ``attention._chunk`` and ``_pages``), element for element, including
which rows each launch writes:

- ``requests``: request starts (``nt`` past the live requests), each request's first
  position (``seq - query``, -1 past them) and the live counts ``[nt, nr]``.
- ``tokens``: per token its request, position, ring or paged slot (-1 unless the
  state is emitted) and visible compressed length, over the rows of the first
  ``ceil(nt / 128)`` blocks of 128 (the readable domain; capacity rows past it keep
  their contents).

  ``builder`` writes both in one launch: every KV-cache group's builder runs them
  each step (26 groups at DS4.1's decode, target and drafter), and Triton's pair
  was two launches a group.
- ``chunk``: per row its sliding-window page indices (-1 holes), window length and
  top-k length.
- ``pages``: per row its request's page table, block 0 and holes as -1, over the
  pages of the row's visible entries only. Triton's ``_pages`` writes every column;
  at a 1M-token context that is 4,096 a row, where a row reads
  ``ceil(visible / page)``: the sparse MLA's indexed lanes and the indexer's tiles
  look up only positions below the row's visible length, and a masked lane's lookup
  is discarded (``test_sparse_mla_reads_only_live_pages``,
  ``test_indexer_reads_only_live_pages``). Columns past them keep their contents.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T

_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
_BLOCK = 128
_TOP = 512  # top-k length cap


def _search_steps(requests: int) -> int:
    """Binary-search iterations over ``requests`` (+1 for the empty tail)."""
    return max(1, (requests + 1).bit_length())


@tilelang.jit(pass_configs=_PASS)
def builder_kernel(max_requests: int, page: int, ratio: int, circular: bool):
    """A metadata builder's ``requests`` and ``tokens`` in one launch: the request
    blocks first (``ceil((max_requests + 1) / 128)``), then the token blocks (the
    first ``ceil(nt / 128)`` blocks of rows). The two parts read only the step's
    inputs, never each other's outputs."""
    cover = T.dynamic("cover")  # rows written: the first ceil(nt / 128) blocks
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    slot_rows = T.dynamic("slot_rows")
    steps = _search_steps(max_requests)
    request_blocks = (max_requests + 1 + _BLOCK - 1) // _BLOCK

    @T.prim_func
    def metadata_builder(
        starts: T.Tensor((starts_len,), T.int32),
        seq: T.Tensor((seq_len,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        input_slots: T.Tensor((slot_rows,), T.int64),
        out_starts: T.Tensor((max_requests + 1,), T.int32),
        request_positions: T.Tensor((max_requests,), T.int64),
        counts: T.Tensor((2,), T.int32),
        positions: T.Tensor((cover,), T.int64),
        reqs: T.Tensor((cover,), T.int32),
        slots: T.Tensor((cover,), T.int64),
        visible: T.Tensor((cover,), T.int32),
        nr: T.int32,
        nt: T.int32,
    ):
        with T.Kernel(request_blocks + T.ceildiv(cover, _BLOCK), threads=_BLOCK) as bx:
            tx = T.get_thread_binding()
            bounds = T.alloc_local((2,), T.int32)
            if bx < request_blocks:
                r = bx * _BLOCK + tx
                if r <= max_requests:
                    start = T.if_then_else(r <= nr, starts[r], nt)
                    end = T.if_then_else(r < nr, starts[r + 1], nt)
                    out_starts[r] = start
                    if r < max_requests:
                        request_positions[r] = T.if_then_else(
                            r < nr,
                            T.Cast(T.int64, seq[r] - (end - start)),
                            T.int64(-1),
                        )
                if r == 0:
                    counts[0] = nt
                    counts[1] = nr
            else:
                t = (bx - request_blocks) * _BLOCK + tx
                if t < cover:
                    # The request whose rows contain t (Triton's upper-bound search).
                    bounds[0] = 0
                    bounds[1] = nr
                    for _ in T.serial(steps):
                        if bounds[0] < bounds[1]:
                            mid = (bounds[0] + bounds[1]) // 2
                            end_mid = T.if_then_else(mid < nr, starts[mid + 1], nt)
                            if end_mid <= t:
                                bounds[0] = mid + 1
                            else:
                                bounds[1] = mid
                    lo = bounds[0]
                    live = (t < nt) & (lo < nr)
                    start = T.if_then_else(live, starts[lo], 0)
                    end = T.if_then_else(live, starts[lo + 1], 0)
                    sq = T.if_then_else(live, seq[lo], 0)
                    pos = (
                        T.Cast(T.int64, sq) - T.Cast(T.int64, end - start) + (t - start)
                    )
                    input_slot = T.if_then_else(t < nt, input_slots[t], T.int64(-1))
                    if circular:
                        valid0 = live & (pos >= 0)
                    else:
                        valid0 = live & (pos >= 0) & (input_slot >= 0)
                    logical = T.truncdiv(pos, ratio)
                    page_col = T.int64(0) if circular else T.truncdiv(logical, page)
                    valid1 = valid0 & (page_col < table_cols)
                    block = T.if_then_else(
                        valid1, T.Cast(T.int64, table[lo, page_col]), T.int64(0)
                    )
                    valid = valid1 & (block > 0)
                    slot = block * page + T.truncmod(logical, page)
                    emit = valid & (T.truncmod(pos + 1, ratio) == 0)
                    if circular:
                        # Only a request's last ring-size rows take a ring slot: the
                        # earlier rows of a long chunk would share them.
                        emit = emit & (t >= end - page)
                    positions[t] = T.if_then_else(valid, pos, T.int64(-1))
                    reqs[t] = T.if_then_else(valid, lo, -1)
                    slots[t] = T.if_then_else(emit, slot, T.int64(-1))
                    visible[t] = T.if_then_else(
                        valid, T.Cast(T.int32, T.truncdiv(pos + 1, ratio)), 0
                    )

    return metadata_builder


def _window_macro(page, window, width, draft, replay, threads, per_thread):
    """One row's sliding-window metadata (``attention._chunk``), for a block of
    ``threads`` threads."""

    @T.macro
    def window_row(
        row, tx, info, positions, reqs, table, swa, lengths, top_lengths, visible,
        starts, request_positions, replay_start, offset,
    ):  # fmt: skip
        req = reqs[offset + row]
        pos = positions[offset + row]
        valid = (req >= 0) & (pos >= 0)
        if draft:
            first = T.if_then_else(valid, T.Cast(T.int64, starts[req]), T.int64(0))
            end = T.if_then_else(valid, T.Cast(T.int64, starts[req + 1]), T.int64(0))
            prefix = T.if_then_else(valid, request_positions[req], T.int64(0))
            info[0] = T.max(prefix - window, T.int64(0))
            length = prefix + end - first - info[0]
        else:
            info[0] = T.max(pos + 1 - window, T.int64(0))
            if replay:
                info[0] = T.max(
                    info[0], T.if_then_else(valid, replay_start[req], T.int64(0))
                )
            length = pos + 1 - info[0]
        info[1] = T.if_then_else(
            valid, T.min(T.max(length, T.int64(0)), T.int64(width)), T.int64(0)
        )
        for k in T.serial(per_thread):
            col = k * threads + tx
            if col < width:
                logical = info[0] + col
                inside = valid & (logical >= 0) & (col < info[1])
                page_id = T.if_then_else(
                    inside,
                    T.Cast(T.int64, table[req, T.truncdiv(logical, page)]),
                    T.int64(-1),
                )
                physical = page_id * page + T.truncmod(logical, page)
                swa[row, col] = T.if_then_else(
                    (page_id > 0) & (logical >= 0) & (col < info[1]),
                    T.Cast(T.int32, physical),
                    -1,
                )
        if tx == 0:
            lengths[row] = T.Cast(T.int32, T.max(info[1], T.int64(0)))
            top_lengths[row] = T.if_then_else(
                valid, T.min(T.max(visible[offset + row], 0), _TOP), 0
            )

    return window_row


def _live_pages(visible, page: int, read_width):
    """The page-table columns a row's readers reach: the pages holding its
    ``visible`` entries (``page`` a page), at most ``read_width``."""
    return T.min((T.max(visible, 0) + page - 1) // page, read_width)


def _threads(width: int) -> tuple[int, int]:
    threads = min(256, max(32, (width + 31) // 32 * 32))
    return threads, (width + threads - 1) // threads


@tilelang.jit(pass_configs=_PASS)
def chunk_kernel(page: int, window: int, width: int, draft: bool, replay: bool):
    rows = T.dynamic("rows")
    src_rows = T.dynamic("src_rows")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    req_count = T.dynamic("req_count")
    starts_len = T.dynamic("starts_len")
    replay_len = T.dynamic("replay_len")
    threads, per_thread = _threads(width)
    window_row = _window_macro(page, window, width, draft, replay, threads, per_thread)

    @T.prim_func
    def metadata_chunk(
        positions: T.Tensor((src_rows,), T.int64),
        reqs: T.Tensor((src_rows,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        swa: T.Tensor((rows, width), T.int32),
        lengths: T.Tensor((rows,), T.int32),
        top_lengths: T.Tensor((rows,), T.int32),
        visible: T.Tensor((src_rows,), T.int32),
        starts: T.Tensor((starts_len,), T.int32),
        request_positions: T.Tensor((req_count,), T.int64),
        replay_start: T.Tensor((replay_len,), T.int64),
        offset: T.int32,
    ):
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding()
            info = T.alloc_local((2,), T.int64)  # window start, clamped length
            window_row(
                row, tx, info, positions, reqs, table, swa, lengths, top_lengths,
                visible, starts, request_positions, replay_start, offset,
            )  # fmt: skip

    return metadata_chunk


@tilelang.jit(pass_configs=_PASS)
def decode_metadata_kernel(
    page: int,
    window: int,
    width: int,
    draft: bool,
    replay: bool,
    main_width: int,
    main_page: int,
):
    """``chunk`` and ``pages`` of the main cache in one launch, a block for each part
    of each row: the window metadata and the main page table (block 0 and holes as
    -1) over the pages of the row's ``visible`` entries (``main_page`` a page)."""
    rows = T.dynamic("rows")
    src_rows = T.dynamic("src_rows")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")
    main_src_rows = T.dynamic("main_src_rows")
    main_rows = T.dynamic("main_rows")
    main_cols = T.dynamic("main_cols")
    main_stride = T.dynamic("main_stride")
    req_count = T.dynamic("req_count")
    starts_len = T.dynamic("starts_len")
    replay_len = T.dynamic("replay_len")
    # Blocks as wide as the window part; the page part loops over the row's live
    # pages (idle threads cost the prefill rows' blocks). The loop keeps the table
    # width's trip count, so it unrolls and a long context's loads overlap; past the
    # live pages an iteration is a compare.
    threads, per_thread = _threads(width)
    window_row = _window_macro(page, window, width, draft, replay, threads, per_thread)
    main_per_thread = (main_width + threads - 1) // threads

    @T.prim_func
    def metadata_decode(
        positions: T.Tensor((src_rows,), T.int64),
        reqs: T.Tensor((src_rows,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        swa: T.Tensor((rows, width), T.int32),
        lengths: T.Tensor((rows,), T.int32),
        top_lengths: T.Tensor((rows,), T.int32),
        visible: T.Tensor((src_rows,), T.int32),
        starts: T.Tensor((starts_len,), T.int32),
        request_positions: T.Tensor((req_count,), T.int64),
        replay_start: T.Tensor((replay_len,), T.int64),
        main_reqs: T.Tensor((main_src_rows,), T.int32),
        main_table: T.StridedTensor((main_rows, main_cols), (main_stride, 1), T.int32),
        main_pages: T.Tensor((rows, main_width), T.int32),
        offset: T.int32,
        main_read_width: T.int32,
    ):
        # Two blocks a row, one launch: the window metadata, then the page table.
        with T.Kernel(rows, 2, threads=threads) as (row, part):
            tx = T.get_thread_binding()
            info = T.alloc_local((2,), T.int64)  # window start, clamped length
            live = T.alloc_var(T.int32)  # the page part's columns
            if part == 0:
                window_row(
                    row, tx, info, positions, reqs, table, swa, lengths, top_lengths,
                    visible, starts, request_positions, replay_start, offset,
                )  # fmt: skip
            else:
                req = main_reqs[offset + row]
                live = _live_pages(
                    visible[offset + row], main_page, T.min(main_read_width, main_width)
                )
                if req < 0:
                    live = 0
                for k in T.serial(main_per_thread):
                    col = k * threads + tx
                    if col < live:
                        page_id = main_table[req, col]
                        main_pages[row, col] = T.if_then_else(page_id > 0, page_id, -1)

    return metadata_decode


@tilelang.jit(pass_configs=_PASS)
def pages_kernel(width: int, page: int):
    """Each row's page table over the pages of its ``lengths`` entries (``page`` a
    page)."""
    rows = T.dynamic("rows")
    src_rows = T.dynamic("src_rows")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")

    @T.prim_func
    def metadata_pages(
        reqs: T.Tensor((src_rows,), T.int32),
        lengths: T.Tensor((src_rows,), T.int32),
        table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        out: T.Tensor((rows, width), T.int32),
        offset: T.int32,
        read_width: T.int32,
    ):
        with T.Kernel(rows, T.ceildiv(width, _BLOCK), threads=_BLOCK) as (row, cb):
            col = cb * _BLOCK + T.get_thread_binding()
            req = reqs[offset + row]
            live = _live_pages(lengths[offset + row], page, T.min(read_width, width))
            if (req >= 0) & (col < live):
                # vLLM reserves physical block zero; page tables use -1 for holes.
                page_id = table[req, col]
                out[row, col] = T.if_then_else(page_id > 0, page_id, -1)

    return metadata_pages


def builder(
    starts, seq, table, input_slots, out_starts, request_positions, counts, positions,
    reqs, slots, visible, nr: int, nt: int, *, page: int, ratio: int, circular: bool,
) -> None:  # fmt: skip
    """``sparse_mla._requests`` then ``sparse_mla._tokens``, one launch: the request
    starts, positions and counts, and the token rows of the first
    ``max(1, ceil(nt / 128))`` blocks."""
    cover = min(positions.shape[0], max(1, -(-nt // _BLOCK)) * _BLOCK)
    builder_kernel(request_positions.shape[0], page, ratio, circular)(
        starts, seq, table, input_slots, out_starts, request_positions, counts,
        positions[:cover], reqs[:cover], slots[:cover], visible[:cover], nr, nt,
    )  # fmt: skip


def chunk(
    positions, reqs, table, swa, lengths, top_lengths, visible, starts,
    request_positions, offset: int, rows: int, *, page: int, window: int,
    draft: bool, replay_start=None,
) -> None:  # fmt: skip
    """``attention._chunk`` for ``rows`` rows from source row ``offset``."""
    width = swa.shape[1]
    replay = replay_start is not None and not draft
    chunk_kernel(page, window, width, draft, replay)(
        positions, reqs, table, swa[:rows], lengths[:rows], top_lengths[:rows],
        visible, starts, request_positions,
        replay_start if replay_start is not None else request_positions, offset,
    )  # fmt: skip


def decode_metadata(
    positions, reqs, table, swa, lengths, top_lengths, visible, starts,
    request_positions, main_reqs, main_table, main_pages, offset: int, rows: int,
    main_read_width: int, *, page: int, window: int, draft: bool, main_page: int,
    replay_start=None,
) -> None:  # fmt: skip
    """``chunk`` then ``pages`` of the main cache for ``rows`` rows, one launch; a row's
    main page table covers the pages of its ``visible`` entries, ``main_page`` a page
    (its other columns keep their contents)."""
    width, main_width = swa.shape[1], main_pages.shape[1]
    replay = replay_start is not None and not draft
    decode_metadata_kernel(page, window, width, draft, replay, main_width, main_page)(
        positions, reqs, table, swa[:rows], lengths[:rows], top_lengths[:rows],
        visible, starts, request_positions,
        replay_start if replay_start is not None else request_positions, main_reqs,
        main_table, main_pages[:rows], offset, main_read_width,
    )  # fmt: skip


def pages(
    reqs, lengths, table, out, offset: int, rows: int, read_width: int, *, page: int
) -> None:
    """``attention._pages`` for ``rows`` rows from source row ``offset``, over the
    pages of each row's ``lengths`` entries (``page`` a page; ``out`` is ``[*, width]``,
    its other columns keep their contents)."""
    pages_kernel(out.shape[1], page)(
        reqs, lengths, table, out[:rows], offset, read_width
    )


__all__ = [
    "builder",
    "builder_kernel",
    "chunk",
    "chunk_kernel",
    "decode_metadata",
    "decode_metadata_kernel",
    "pages",
    "pages_kernel",
]
