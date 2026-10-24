"""TileLang variants of the V2 runner's tiny input-preparation kernels.

Each factory is one point of the sweep in bench.py: the same contract as the
production kernel in vllm/v1/worker/gpu/tilelang_inputs.py (and the Triton kernel
it replaces), with a different launch shape or code shape:

- ``hoist``: every scalar a request needs is read once into a register at the top
  (``T.alloc_var``), so independent loads issue together. In the production
  kernels a Python name bound to ``buf[i]`` is inlined at each use, and TileLang's
  IfStmtBinding re-substitutes binds under every guarded statement, so the same
  loads are re-issued after each store and branch, in a long dependent chain.
- ``threads`` / requests a block / ``per`` (elements a thread a pass).
- ``lanes``: Triton's layout, a lane per element of a request's block (the dead-row
  scan by warp shuffles in Triton's Kogge-Stone order) instead of a thread per
  request.
- ``spread``: a block per request (Triton's grid) instead of a request per thread.
- ``index``: the request index (and the products with row strides) in int32
  instead of int64.
- ``static``: row strides as compile-time constants instead of kernel parameters.

Only the deployment's configuration is covered (one sampled token a step, output
bin counts and query starts in the commit, no context parallelism in DFlash, an
Engram window of 3), which is what the benchmark times.
"""

import tilelang
import tilelang.language as T

PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
PAD = -1
SPAN = 1 << 40


def pow2(value: int) -> int:
    return 1 << max(value - 1, 0).bit_length()


def scan_strides(block: int) -> tuple:
    return tuple(1 << i for i in range(max(block - 1, 0).bit_length()))


# --- combine_sampled_and_draft_tokens ------------------------------------------


@tilelang.jit(pass_configs=PASS)
def combine_thread_kernel(
    drafts: int, num_new: int, dead: bool, threads: int, spread: bool, index: str
):
    """A thread a request (``spread``: a block a request, its first thread), every
    scalar the request needs read once up front, the drafts (and confidences)
    included: they are the request's row of a [max_reqs, drafts] tensor."""
    block = pow2(drafts + num_new)
    per_block = 1 if spread else threads
    ids_len = T.dynamic("ids_len")
    num_reqs = T.dynamic("num_reqs")
    sampled_len = T.dynamic("sampled_len")
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    prefill_rows = T.dynamic("prefill_rows")
    draft_rows = T.dynamic("draft_rows")
    draft_stride = T.dynamic("draft_stride")
    cu_len = T.dynamic("cu_len")
    logits_len = T.dynamic("logits_len")
    conf_rows = T.dynamic("conf_rows")
    conf_stride = T.dynamic("conf_stride")
    padding_len = T.dynamic("padding_len")
    cut_len = T.dynamic("cut_len")

    @T.macro
    def request(
        b, input_ids, idx_mapping, last_sampled, query_start_loc, seq_lens, prefill_len,
        draft_tokens, cu_num_logits, logits_indices, confidence, is_padding, dead_cut,
        log_tau,
    ):  # fmt: skip
        req = T.alloc_var(index, init=T.cast(idx_mapping[b], index))
        first = T.alloc_var(T.int32, init=cu_num_logits[b])
        end = T.alloc_var(T.int32, init=cu_num_logits[b + 1])
        query_end = T.alloc_var(T.int32, init=query_start_loc[b + 1])
        seq = T.alloc_var(T.int32, init=seq_lens[b])
        prefill = T.alloc_var(T.int32, init=prefill_len[req])
        last = T.alloc_var(T.int64, init=last_sampled[req])
        tokens = T.alloc_local((drafts,), T.int64)
        for j in T.unroll(drafts):
            tokens[j] = draft_tokens[req, j]
        if dead:
            confs = T.alloc_local((drafts,), T.float32)
            for j in T.unroll(drafts):
                confs[j] = confidence[req, j]
        num_logits = end - first
        num_drafts = num_logits - num_new
        for j in T.unroll(block):
            if j < num_logits:
                logits_indices[first + j] = T.cast(query_end - num_logits + j, T.int64)
        if seq > prefill:
            if num_new > 0:
                if seq - num_logits >= prefill:
                    input_ids[query_end - num_logits] = T.cast(last, T.int32)
            for j in tuple(range(min(block, drafts))):
                if j < num_drafts:
                    input_ids[query_end - num_drafts + j] = T.cast(tokens[j], T.int32)
            for j in tuple(range(drafts, block)):
                if j < num_drafts:
                    input_ids[query_end - num_drafts + j] = T.cast(draft_tokens[req, j], T.int32)
        if dead:
            cut = T.alloc_var(T.int32, init=block + 1)
            if seq > prefill and num_drafts > 0:
                survival = T.alloc_local((block,), T.float32)
                for j in tuple(range(min(block, drafts))):
                    survival[j] = T.log(T.if_then_else(j < num_drafts, confs[j], T.float32(1.0)))
                for j in tuple(range(drafts, block)):
                    survival[j] = T.log(
                        T.if_then_else(j < num_drafts, confidence[req, j], T.float32(1.0))
                    )
                for stride in scan_strides(block):
                    for j in tuple(range(block - 1, stride - 1, -1)):
                        survival[j] = survival[j - stride] + survival[j]
                cut = 0
                for j in tuple(range(block)):
                    if survival[j] >= log_tau and j < num_drafts:
                        cut = cut + 1
                for j in tuple(range(block)):
                    if j < num_drafts and j >= cut:
                        is_padding[query_end - num_drafts + j] = T.uint8(1)
            dead_cut[b] = cut

    if not dead:

        @T.prim_func
        def combine(
            input_ids: T.Tensor((ids_len,), T.int32),
            idx_mapping: T.Tensor((num_reqs,), T.int64),
            last_sampled: T.Tensor((sampled_len,), T.int64),
            query_start_loc: T.Tensor((starts_len,), T.int32),
            seq_lens: T.Tensor((seq_len,), T.int32),
            prefill_len: T.Tensor((prefill_rows,), T.int32),
            draft_tokens: T.StridedTensor((draft_rows, drafts), (draft_stride, 1), T.int64),
            cu_num_logits: T.Tensor((cu_len,), T.int32),
            logits_indices: T.Tensor((logits_len,), T.int64),
        ):
            with T.Kernel(T.ceildiv(num_reqs, per_block), threads=threads) as bx:
                tx = T.get_thread_binding()
                if spread:
                    if tx == 0:
                        request(
                            bx, input_ids, idx_mapping, last_sampled, query_start_loc,
                            seq_lens, prefill_len, draft_tokens, cu_num_logits,
                            logits_indices, None, None, None, None,
                        )  # fmt: skip
                else:
                    if bx * threads + tx < num_reqs:
                        request(
                            bx * threads + tx, input_ids, idx_mapping, last_sampled,
                            query_start_loc, seq_lens, prefill_len, draft_tokens,
                            cu_num_logits, logits_indices, None, None, None, None,
                        )  # fmt: skip

        return combine

    @T.prim_func
    def combine_dead(
        input_ids: T.Tensor((ids_len,), T.int32),
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        last_sampled: T.Tensor((sampled_len,), T.int64),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        seq_lens: T.Tensor((seq_len,), T.int32),
        prefill_len: T.Tensor((prefill_rows,), T.int32),
        draft_tokens: T.StridedTensor((draft_rows, drafts), (draft_stride, 1), T.int64),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        logits_indices: T.Tensor((logits_len,), T.int64),
        confidence: T.StridedTensor((conf_rows, drafts), (conf_stride, 1), T.float32),
        is_padding: T.Tensor((padding_len,), T.uint8),
        dead_cut: T.Tensor((cut_len,), T.int32),
        log_tau: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_reqs, per_block), threads=threads) as bx:
            tx = T.get_thread_binding()
            if spread:
                if tx == 0:
                    request(
                        bx, input_ids, idx_mapping, last_sampled, query_start_loc,
                        seq_lens, prefill_len, draft_tokens, cu_num_logits,
                        logits_indices, confidence, is_padding, dead_cut, log_tau,
                    )  # fmt: skip
            else:
                if bx * threads + tx < num_reqs:
                    request(
                        bx * threads + tx, input_ids, idx_mapping, last_sampled,
                        query_start_loc, seq_lens, prefill_len, draft_tokens,
                        cu_num_logits, logits_indices, confidence, is_padding,
                        dead_cut, log_tau,
                    )  # fmt: skip

    return combine_dead


@tilelang.jit(pass_configs=PASS)
def combine_lanes_kernel(
    drafts: int, num_new: int, dead: bool, threads: int, spread: bool, index: str
):
    """Triton's layout: ``block`` lanes a request (a lane a logit or draft), so a
    block holds threads / block requests (``spread``: one, as Triton's grid). Every
    lane reads its request's scalars once; the dead-row survival is a Kogge-Stone
    scan by warp shuffles (strides 1, 2, 4, ...) and the cut a shuffle sum."""
    block = pow2(drafts + num_new)
    lanes = block
    assert threads % 32 == 0 and 32 % lanes == 0
    per_block = 1 if spread else threads // lanes
    sums = tuple(lanes >> (i + 1) for i in range(max(lanes - 1, 0).bit_length()))
    ids_len = T.dynamic("ids_len")
    num_reqs = T.dynamic("num_reqs")
    sampled_len = T.dynamic("sampled_len")
    starts_len = T.dynamic("starts_len")
    seq_len = T.dynamic("seq_len")
    prefill_rows = T.dynamic("prefill_rows")
    draft_rows = T.dynamic("draft_rows")
    draft_stride = T.dynamic("draft_stride")
    cu_len = T.dynamic("cu_len")
    logits_len = T.dynamic("logits_len")
    conf_rows = T.dynamic("conf_rows")
    conf_stride = T.dynamic("conf_stride")
    padding_len = T.dynamic("padding_len")
    cut_len = T.dynamic("cut_len")

    @T.macro
    def lanes_body(
        bx, tx, num_reqs, input_ids, idx_mapping, last_sampled, query_start_loc,
        seq_lens, prefill_len, draft_tokens, cu_num_logits, logits_indices, confidence,
        is_padding, dead_cut, log_tau,
    ):  # fmt: skip
        lane = tx % lanes
        slot = tx // lanes
        live = slot < per_block and bx * per_block + slot < num_reqs
        row = T.alloc_var(T.int32, init=T.if_then_else(live, bx * per_block + slot, 0))
        req = T.alloc_var(index, init=T.cast(idx_mapping[row], index))
        first = T.alloc_var(T.int32, init=cu_num_logits[row])
        end = T.alloc_var(T.int32, init=cu_num_logits[row + 1])
        query_end = T.alloc_var(T.int32, init=query_start_loc[row + 1])
        seq = T.alloc_var(T.int32, init=seq_lens[row])
        prefill = T.alloc_var(T.int32, init=prefill_len[req])
        num_logits = end - first
        num_drafts = num_logits - num_new
        if live:
            if lane < num_logits:
                logits_indices[first + lane] = T.cast(query_end - num_logits + lane, T.int64)
            if seq > prefill:
                if num_new > 0:
                    if lane == 0 and seq - num_logits >= prefill:
                        input_ids[query_end - num_logits] = T.cast(last_sampled[req], T.int32)
                if lane < num_drafts:
                    input_ids[query_end - num_drafts + lane] = T.cast(
                        draft_tokens[req, lane], T.int32
                    )
        if dead:
            # Every lane takes part in the shuffles: lanes past a request's drafts,
            # and of rows that are not decoding, carry log(1).
            survival = T.alloc_var(
                T.float32,
                init=T.log(
                    T.if_then_else(
                        live and seq > prefill and lane < num_drafts,
                        confidence[req, lane],
                        T.float32(1.0),
                    )
                ),
            )
            shifted = T.alloc_var(T.float32, init=T.float32(0))
            for stride in scan_strides(block):
                shifted = T.shfl_up(survival, stride, lanes)
                if lane >= stride:
                    survival = shifted + survival
            hits = T.alloc_var(T.int32, init=0)
            if survival >= log_tau and lane < num_drafts:
                hits = 1
            for delta in sums:
                hits = hits + T.shfl_xor(hits, delta, lanes)
            if live:
                if seq > prefill and num_drafts > 0:
                    if lane == 0:
                        dead_cut[row] = hits
                    if lane < num_drafts and lane >= hits:
                        is_padding[query_end - num_drafts + lane] = T.uint8(1)
                else:
                    if lane == 0:
                        dead_cut[row] = block + 1

    if not dead:

        @T.prim_func
        def combine(
            input_ids: T.Tensor((ids_len,), T.int32),
            idx_mapping: T.Tensor((num_reqs,), T.int64),
            last_sampled: T.Tensor((sampled_len,), T.int64),
            query_start_loc: T.Tensor((starts_len,), T.int32),
            seq_lens: T.Tensor((seq_len,), T.int32),
            prefill_len: T.Tensor((prefill_rows,), T.int32),
            draft_tokens: T.StridedTensor((draft_rows, drafts), (draft_stride, 1), T.int64),
            cu_num_logits: T.Tensor((cu_len,), T.int32),
            logits_indices: T.Tensor((logits_len,), T.int64),
        ):
            with T.Kernel(T.ceildiv(num_reqs, per_block), threads=threads) as bx:
                lanes_body(
                    bx, T.get_thread_binding(), num_reqs, input_ids, idx_mapping,
                    last_sampled, query_start_loc, seq_lens, prefill_len, draft_tokens,
                    cu_num_logits, logits_indices, None, None, None, None,
                )  # fmt: skip

        return combine

    @T.prim_func
    def combine_dead(
        input_ids: T.Tensor((ids_len,), T.int32),
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        last_sampled: T.Tensor((sampled_len,), T.int64),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        seq_lens: T.Tensor((seq_len,), T.int32),
        prefill_len: T.Tensor((prefill_rows,), T.int32),
        draft_tokens: T.StridedTensor((draft_rows, drafts), (draft_stride, 1), T.int64),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        logits_indices: T.Tensor((logits_len,), T.int64),
        confidence: T.StridedTensor((conf_rows, drafts), (conf_stride, 1), T.float32),
        is_padding: T.Tensor((padding_len,), T.uint8),
        dead_cut: T.Tensor((cut_len,), T.int32),
        log_tau: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_reqs, per_block), threads=threads) as bx:
            lanes_body(
                bx, T.get_thread_binding(), num_reqs, input_ids, idx_mapping,
                last_sampled, query_start_loc, seq_lens, prefill_len, draft_tokens,
                cu_num_logits, logits_indices, confidence, is_padding, dead_cut, log_tau,
            )  # fmt: skip

    return combine_dead


# --- expand_idx_mapping ----------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def expand_kernel(block: int, threads: int, spread: bool):
    """A thread an expanded entry (``spread``: a block a request, a thread a logit);
    the request's starts and state index read once, before the bound check."""
    num_reqs = T.dynamic("num_reqs")
    expanded_len = T.dynamic("expanded_len")
    local_len = T.dynamic("local_len")
    cu_len = T.dynamic("cu_len")
    grid = num_reqs if spread else T.ceildiv(num_reqs * block, threads)

    @T.macro
    def entry(r, j, num_reqs, idx_mapping, expanded_idx_mapping, expanded_local_pos,
              cu_num_logits):  # fmt: skip
        if r < num_reqs and j < block:
            start = T.alloc_var(T.int32, init=cu_num_logits[r])
            end = T.alloc_var(T.int32, init=cu_num_logits[r + 1])
            value = T.alloc_var(T.int64, init=idx_mapping[r])
            if j < end - start:
                expanded_idx_mapping[start + j] = value
                expanded_local_pos[start + j] = j

    @T.prim_func
    def expand(
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        expanded_idx_mapping: T.Tensor((expanded_len,), T.int64),
        expanded_local_pos: T.Tensor((local_len,), T.int32),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
    ):
        with T.Kernel(grid, threads=threads) as bx:
            tx = T.get_thread_binding()
            if spread:
                entry(bx, tx, num_reqs, idx_mapping, expanded_idx_mapping,
                      expanded_local_pos, cu_num_logits)  # fmt: skip
            else:
                entry(
                    (bx * threads + tx) // block, (bx * threads + tx) % block,
                    num_reqs, idx_mapping, expanded_idx_mapping, expanded_local_pos,
                    cu_num_logits,
                )  # fmt: skip

    return expand


# Triton stores under a mask as predicated instructions. A guarded store that is a
# branch instead lets ptxas sink the loads feeding it below the branch, behind the
# loads the guard waits for: two round trips to memory where Triton has one. The
# ``tile`` variants store through these helpers, so nothing is branched over.
STORE_IF = r"""
__device__ __forceinline__ void vllm_store_if_b64(void* p, long long v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q st.global.b64 [%0], %1;\n}"
               :: "l"(p), "l"(v), "r"((int)ok) : "memory");
}
__device__ __forceinline__ void vllm_store_if_b32(void* p, int v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q st.global.b32 [%0], %1;\n}"
               :: "l"(p), "r"(v), "r"((int)ok) : "memory");
}
"""


@tilelang.jit(pass_configs=PASS)
def expand_tile_kernel(block: int, threads: int, plain: bool = False):
    """``expand_kernel`` in Triton's form: the request's starts and state index read
    up front, the two guarded stores predicated (``STORE_IF``). ``plain``: the
    inputs without ``__restrict__``, so their loads are ordinary global loads, as
    Triton's, instead of the read-only cache path (``ld.global.nc``)."""
    num_reqs = T.dynamic("num_reqs")
    expanded_len = T.dynamic("expanded_len")
    local_len = T.dynamic("local_len")
    cu_len = T.dynamic("cu_len")

    @T.prim_func
    def expand(
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        expanded_idx_mapping: T.Tensor((expanded_len,), T.int64),
        expanded_local_pos: T.Tensor((local_len,), T.int32),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
    ):
        with T.Kernel(T.ceildiv(num_reqs * block, threads), threads=threads) as bx:
            if plain:
                T.annotate_restrict_buffers(idx_mapping, cu_num_logits)
            T.import_source(STORE_IF)
            element = bx * threads + T.get_thread_binding()
            r = element // block
            j = element % block
            if r < num_reqs:
                start = T.alloc_var(T.int32, init=cu_num_logits[r])
                end = T.alloc_var(T.int32, init=cu_num_logits[r + 1])
                value = T.alloc_var(T.int64, init=idx_mapping[r])
                ok = j < end - start
                T.call_extern(
                    "handle", "vllm_store_if_b64",
                    T.address_of(expanded_idx_mapping[start + j]), value, ok,
                )  # fmt: skip
                T.call_extern(
                    "handle", "vllm_store_if_b32",
                    T.address_of(expanded_local_pos[start + j]), j, ok,
                )  # fmt: skip

    return expand


# --- prepare_prefill_inputs ------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def prefill_inputs_kernel(lookahead_block: int, threads: int, per: int, index: str):
    """A block a request, ``threads`` threads of ``per`` tokens a pass; the query
    starts are read with the request's state, before the prefill check."""
    chunk = threads * per
    ids_len = T.dynamic("ids_len")
    lookahead = T.dynamic("lookahead")
    lookahead_cols = T.dynamic("lookahead_cols")
    lookahead_stride = T.dynamic("lookahead_stride")
    num_reqs = T.dynamic("num_reqs")
    starts_len = T.dynamic("starts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")
    prefill_rows = T.dynamic("prefill_rows")
    counts_len = T.dynamic("counts_len")

    @T.prim_func
    def prepare_prefill_inputs(
        input_ids: T.Tensor((ids_len,), T.int32),
        next_prefill_tokens: T.StridedTensor(
            (lookahead, lookahead_cols), (lookahead_stride, 1), T.int32
        ),
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        all_token_ids: T.StridedTensor((history_rows, history_cols), (history_stride, 1), T.int32),
        prefill_len: T.Tensor((prefill_rows,), T.int32),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
    ):
        with T.Kernel(num_reqs, threads=threads) as b:
            tx = T.get_thread_binding()
            start = T.alloc_var(T.int32, init=query_start_loc[b])
            end = T.alloc_var(T.int32, init=query_start_loc[b + 1])
            req = T.alloc_var(index, init=T.cast(idx_mapping[b], index))
            prefill = T.alloc_var(T.int32, init=prefill_len[req])
            computed = T.alloc_var(T.int32, init=num_computed_tokens[req])
            if computed < prefill:
                for it in T.serial(T.ceildiv(end - start, chunk)):
                    for k in T.unroll(per):
                        j = it * chunk + k * threads + tx
                        if j < end - start:
                            input_ids[start + j] = all_token_ids[req, computed + j]
                if tx < lookahead_block and tx < lookahead:
                    next_prefill_tokens[tx, req] = T.if_then_else(
                        computed + end - start + tx < prefill,
                        all_token_ids[req, computed + end - start + tx],
                        0,
                    )

    return prepare_prefill_inputs


# --- post_update ------------------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def post_update_kernel(width: int, threads: int, spread: bool, index: str, strides: tuple):
    """Commit a request's sampled tokens (``bins`` and query starts present, no
    request-boundary captures): its scalars and its whole row of sampled tokens
    read up front (the row has ``width`` columns), the last token selected in
    registers. ``strides``: (bins, history) row strides baked in, or () as
    parameters."""
    per_block = 1 if spread else threads
    num_reqs = T.dynamic("num_reqs")
    computed_len = T.dynamic("computed_len")
    last_len = T.dynamic("last_len")
    bin_rows = T.dynamic("bin_rows")
    bin_cols = T.dynamic("bin_cols")
    sampled_rows = T.dynamic("sampled_rows")
    sampled_stride = T.dynamic("sampled_stride")
    num_sampled_len = T.dynamic("num_sampled_len")
    num_rejected_len = T.dynamic("num_rejected_len")
    starts_len = T.dynamic("starts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    total_len_len = T.dynamic("total_len_len")
    bin_stride = strides[0] if strides else T.dynamic("bin_stride")
    history_stride = strides[1] if strides else T.dynamic("history_stride")

    @T.macro
    def commit(
        r, idx_mapping, num_computed_tokens, last_sampled, output_bin_counts,
        sampled_tokens, num_sampled, num_rejected, query_start_loc, all_token_ids,
        total_len,
    ):  # fmt: skip
        req = T.alloc_var(index, init=T.cast(idx_mapping[r], index))
        count = T.alloc_var(T.int32, init=num_sampled[r])
        rejected = T.alloc_var(T.int32, init=num_rejected[r])
        query_len = T.alloc_var(T.int32, init=query_start_loc[r + 1] - query_start_loc[r])
        tokens = T.alloc_local((width,), T.int64)
        for i in T.unroll(width):
            tokens[i] = sampled_tokens[r, i]
        # Rows with negative indices are filtered.
        if req >= 0:
            computed = T.alloc_var(T.int32, init=num_computed_tokens[req])
            total = T.alloc_var(T.int32, init=total_len[req])
            if count > 0:
                last = T.alloc_var(T.int64, init=tokens[0])
                for i in tuple(range(1, width)):
                    if count - 1 == i:
                        last = tokens[i]
                last_sampled[req] = last
                total_len[req] = total + count
            for i in tuple(range(width)):
                if i < count:
                    all_token_ids[req, total + i] = T.cast(tokens[i], T.int32)
            firsts = []
            for i in tuple(range(width)):
                first = i < count
                for j in tuple(range(i)):
                    first = first and tokens[j] != tokens[i]
                firsts.append(first)
            counts = T.alloc_local((width,), T.int32)
            for i in tuple(range(width)):
                counts[i] = T.if_then_else(firsts[i], output_bin_counts[req, tokens[i]], 0)
            for i in tuple(range(width)):
                if firsts[i]:
                    occurrences = T.cast(i < count, T.int32)
                    for j in tuple(range(i + 1, width)):
                        occurrences = occurrences + T.cast(
                            j < count and tokens[j] == tokens[i], T.int32
                        )
                    output_bin_counts[req, tokens[i]] = counts[i] + occurrences
            if query_len - rejected != 0:
                num_computed_tokens[req] = computed + query_len - rejected

    @T.prim_func
    def post_update(
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        num_computed_tokens: T.Tensor((computed_len,), T.int32),
        last_sampled: T.Tensor((last_len,), T.int64),
        output_bin_counts: T.StridedTensor((bin_rows, bin_cols), (bin_stride, 1), T.int32),
        sampled_tokens: T.StridedTensor((sampled_rows, width), (sampled_stride, 1), T.int64),
        num_sampled: T.Tensor((num_sampled_len,), T.int32),
        num_rejected: T.Tensor((num_rejected_len,), T.int32),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        all_token_ids: T.StridedTensor((history_rows, history_cols), (history_stride, 1), T.int32),
        total_len: T.Tensor((total_len_len,), T.int32),
    ):
        with T.Kernel(T.ceildiv(num_reqs, per_block), threads=threads) as bx:
            tx = T.get_thread_binding()
            if spread:
                if tx == 0:
                    commit(
                        bx, idx_mapping, num_computed_tokens, last_sampled,
                        output_bin_counts, sampled_tokens, num_sampled, num_rejected,
                        query_start_loc, all_token_ids, total_len,
                    )  # fmt: skip
            else:
                if bx * threads + tx < num_reqs:
                    commit(
                        bx * threads + tx, idx_mapping, num_computed_tokens,
                        last_sampled, output_bin_counts, sampled_tokens, num_sampled,
                        num_rejected, query_start_loc, all_token_ids, total_len,
                    )  # fmt: skip

    return post_update


# --- prepare_dflash_inputs --------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def dflash_kernel(threads: int, per: int, index: str):
    """``num_blocks`` blocks of threads * per rows a request (no context
    parallelism, the DFlash default: the anchor is the bonus token and the drafts
    are sampled at their own positions). The request's scalars, its anchor
    position and its bonus token are read once into registers at the top; a row's
    position and block id once."""
    rows_per_block = threads * per
    ids_len = T.dynamic("ids_len")
    query_positions_len = T.dynamic("query_positions_len")
    query_starts_len = T.dynamic("query_starts_len")
    seq_lens_len = T.dynamic("seq_lens_len")
    query_slots_len = T.dynamic("query_slots_len")
    context_positions_len = T.dynamic("context_positions_len")
    context_slots_len = T.dynamic("context_slots_len")
    sample_indices_len = T.dynamic("sample_indices_len")
    sample_pos_len = T.dynamic("sample_pos_len")
    sample_map_len = T.dynamic("sample_map_len")
    temperature_out_len = T.dynamic("temperature_out_len")
    seeds_out_len = T.dynamic("seeds_out_len")
    target_positions_len = T.dynamic("target_positions_len")
    target_starts_len = T.dynamic("target_starts_len")
    idx_len = T.dynamic("idx_len")
    last_len = T.dynamic("last_len")
    next_len = T.dynamic("next_len")
    num_sampled_len = T.dynamic("num_sampled_len")
    num_rejected_len = T.dynamic("num_rejected_len")
    temperature_len = T.dynamic("temperature_len")
    seeds_len = T.dynamic("seeds_len")
    table_rows = T.dynamic("table_rows")
    table_cols = T.dynamic("table_cols")
    table_stride = T.dynamic("table_stride")

    @T.prim_func
    def prepare_dflash_inputs(
        out_input_ids: T.Tensor((ids_len,), T.int32),
        out_query_positions: T.Tensor((query_positions_len,), T.int64),
        out_query_start_loc: T.Tensor((query_starts_len,), T.int32),
        out_seq_lens: T.Tensor((seq_lens_len,), T.int32),
        out_query_slot_mapping: T.Tensor((query_slots_len,), T.int64),
        out_context_positions: T.Tensor((context_positions_len,), T.int64),
        out_context_slot_mapping: T.Tensor((context_slots_len,), T.int64),
        out_sample_indices: T.Tensor((sample_indices_len,), T.int64),
        out_sample_pos: T.Tensor((sample_pos_len,), T.int64),
        out_sample_idx_mapping: T.Tensor((sample_map_len,), T.int32),
        out_temperature: T.Tensor((temperature_out_len,), T.float32),
        out_seeds: T.Tensor((seeds_out_len,), T.int64),
        target_positions: T.Tensor((target_positions_len,), T.int64),
        target_query_start_loc: T.Tensor((target_starts_len,), T.int32),
        idx_mapping: T.Tensor((idx_len,), T.int64),
        last_sampled: T.Tensor((last_len,), T.int64),
        next_prefill_tokens: T.Tensor((next_len,), T.int32),
        num_sampled: T.Tensor((num_sampled_len,), T.int32),
        num_rejected: T.Tensor((num_rejected_len,), T.int32),
        temperature: T.Tensor((temperature_len,), T.float32),
        seeds: T.Tensor((seeds_len,), T.int64),
        block_table: T.StridedTensor((table_rows, table_cols), (table_stride, 1), T.int32),
        parallel_drafting_token_id: T.int32,
        block_size: T.int32,
        num_query_per_req: T.int32,
        num_speculative_steps: T.int32,
        max_num_reqs: T.int32,
        max_num_tokens: T.int32,
        max_model_len: T.int32,
        num_reqs: T.int32,
        num_blocks: T.int32,
    ):
        with T.Kernel(num_blocks, num_reqs, threads=threads) as (bk, r):
            tx = T.get_thread_binding()
            req = T.alloc_var(index, init=T.cast(idx_mapping[r], index))
            context_start = T.alloc_var(T.int32, init=target_query_start_loc[r])
            context_end = T.alloc_var(T.int32, init=target_query_start_loc[r + 1])
            rejected = T.alloc_var(T.int32, init=num_rejected[r])
            sampled = T.alloc_var(T.int32, init=num_sampled[r])
            num_valid = context_end - rejected - context_start
            last_valid = T.alloc_var(
                T.int64,
                init=T.if_then_else(
                    num_valid > 0,
                    target_positions[T.max(context_end - rejected - 1, context_start)],
                    target_positions[context_start] - 1,
                ),
            )
            bonus = T.alloc_var(
                T.int32,
                init=T.if_then_else(
                    sampled > 0,
                    T.cast(last_sampled[req], T.int32),
                    next_prefill_tokens[req],
                ),
            )
            last_column = T.cast(table_stride - 1, T.int64)
            for k in T.unroll(per):
                j = bk * rows_per_block + k * threads + tx
                if j < context_end - context_start:
                    position = T.alloc_var(
                        T.int64,
                        init=T.if_then_else(
                            j < num_valid, target_positions[context_start + j], T.int64(0)
                        ),
                    )
                    block_id = T.alloc_var(
                        T.int64,
                        init=T.if_then_else(
                            j < num_valid,
                            T.cast(
                                block_table[
                                    r, T.min(T.truncdiv(position, block_size), last_column)
                                ],
                                T.int64,
                            ),
                            T.int64(0),
                        ),
                    )
                    out_context_positions[context_start + j] = position
                    out_context_slot_mapping[context_start + j] = T.if_then_else(
                        j < num_valid and block_id != 0,
                        block_id * block_size + T.truncmod(position, block_size),
                        T.int64(PAD),
                    )
                if j >= num_valid and j < num_valid + num_query_per_req:
                    offset = j - num_valid
                    query = r * num_query_per_req + offset
                    position = T.alloc_var(T.int64, init=last_valid + 1 + offset)
                    block_id = T.alloc_var(
                        T.int64,
                        init=T.cast(
                            block_table[r, T.min(T.truncdiv(position, block_size), last_column)],
                            T.int64,
                        ),
                    )
                    out_input_ids[query] = T.if_then_else(
                        offset == 0, bonus, parallel_drafting_token_id
                    )
                    out_query_positions[query] = T.min(position, T.cast(max_model_len - 1, T.int64))
                    out_query_slot_mapping[query] = T.if_then_else(
                        block_id != 0,
                        block_id * block_size + T.truncmod(position, block_size),
                        T.int64(PAD),
                    )
                    if offset >= 1:
                        sample = r * num_speculative_steps + offset - 1
                        out_sample_indices[sample] = T.cast(query, T.int64)
                        out_sample_pos[sample] = position
                        out_sample_idx_mapping[sample] = T.cast(req, T.int32)
            if bk == 0:
                if tx == 0:
                    out_query_start_loc[r] = r * num_query_per_req
                    out_seq_lens[r] = T.cast(
                        T.min(
                            last_valid + 1 + num_query_per_req,
                            T.cast(max_model_len, T.int64),
                        ),
                        T.int32,
                    )
                    out_temperature[req] = temperature[req]
                    out_seeds[req] = seeds[req]
                if r == num_reqs - 1:
                    last_query_end = num_reqs * num_query_per_req
                    for it in T.serial(T.ceildiv(max_num_reqs + 1 - num_reqs, threads)):
                        i = num_reqs + it * threads + tx
                        if i < max_num_reqs + 1:
                            out_query_start_loc[i] = last_query_end
                        if i < max_num_reqs:
                            out_seq_lens[i] = 0
                    pad_start = num_reqs * num_speculative_steps
                    pad_end = max_num_reqs * num_speculative_steps
                    for it in T.serial(T.ceildiv(pad_end - pad_start, threads)):
                        i = pad_start + it * threads + tx
                        if i < pad_end:
                            out_sample_indices[i] = T.int64(0)
                            out_sample_pos[i] = T.int64(0)
                            out_sample_idx_mapping[i] = -1
                    query_end = num_reqs * num_query_per_req
                    for it in T.serial(T.ceildiv(max_num_tokens - query_end, threads)):
                        i = query_end + it * threads + tx
                        if i < max_num_tokens:
                            out_query_slot_mapping[i] = T.int64(PAD)

    return prepare_dflash_inputs


# --- gather_lookback ------------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def lookback_kernel(depth: int, threads: int, spread: bool, index: str):
    """A thread an element of the window (``spread``: a block a row, a thread an
    element, as Triton's grid), the request's computed count read once."""
    rows = T.dynamic("rows")
    num_reqs = T.dynamic("num_reqs")
    counts_len = T.dynamic("counts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")
    grid = rows if spread else T.ceildiv(rows * depth, threads)

    @T.macro
    def element(b, j, rows, num_reqs, lookback, idx_mapping, num_computed_tokens,
                all_token_ids):  # fmt: skip
        if b < rows and j < depth:
            live = b < num_reqs
            req = T.alloc_var(index, init=T.if_then_else(live, T.cast(idx_mapping[b], index), 0))
            computed = T.alloc_var(T.int32, init=T.if_then_else(live, num_computed_tokens[req], 0))
            lookback[b, j] = T.if_then_else(
                live and computed - depth + j >= 0,
                all_token_ids[req, computed - depth + j],
                -1,
            )

    @T.prim_func
    def gather_lookback(
        lookback: T.Tensor((rows, depth), T.int32),
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
        all_token_ids: T.StridedTensor((history_rows, history_cols), (history_stride, 1), T.int32),
    ):
        with T.Kernel(grid, threads=threads) as bx:
            tx = T.get_thread_binding()
            if spread:
                element(bx, tx, rows, num_reqs, lookback, idx_mapping,
                        num_computed_tokens, all_token_ids)  # fmt: skip
            else:
                element(
                    (bx * threads + tx) // depth, (bx * threads + tx) % depth, rows,
                    num_reqs, lookback, idx_mapping, num_computed_tokens, all_token_ids,
                )  # fmt: skip

    return gather_lookback


# --- zero_kv_blocks ---------------------------------------------------------------


@tilelang.jit(pass_configs=PASS)
def zero_kv_kernel(threads: int, per: int):
    """Zero each segment's page of each block: a block covers threads * per int32s
    of one page, in 16-byte stores where the segment's address and block stride
    allow (``per`` / 4 of them a thread, block-strided), else an int32 at a time."""
    assert per % 4 == 0
    chunk = threads * per
    segments = T.dynamic("segments")
    stride_len = T.dynamic("stride_len")
    page_len = T.dynamic("page_len")
    num_ids = T.dynamic("num_ids")

    @T.prim_func
    def zero_kv_blocks(
        seg_addrs: T.Tensor((segments,), T.ptr),
        seg_block_strides: T.Tensor((stride_len,), T.int64),
        seg_page_sizes: T.Tensor((page_len,), T.int64),
        block_ids: T.Tensor((num_ids,), T.int64),
        chunks: T.int32,
    ):
        with T.Kernel(num_ids, segments, chunks, threads=threads) as (b, s, cx):
            tx = T.get_thread_binding()
            page = T.alloc_var(T.int64, init=seg_page_sizes[s])
            first = T.cast(cx, T.int64) * chunk
            if first < page:
                stride = T.alloc_var(T.int64, init=seg_block_strides[s])
                base = T.alloc_var(T.int64, init=block_ids[b] * stride)
                segment = T.make_tensor(seg_addrs[s], (SPAN,), T.int32)
                if T.truncmod(seg_addrs[s], 16) == 0 and T.truncmod(stride, 4) == 0:
                    for k in T.unroll(per // 4):
                        col = first + (k * threads + tx) * 4
                        if col + 4 <= page:
                            T.stg128(segment[base + col], T.Broadcast(T.uint32(0), 4))
                        else:
                            for e in T.unroll(4):
                                if col + e < page:
                                    segment[base + col + e] = 0
                else:
                    for k in T.unroll(per):
                        col = first + k * threads + tx
                        if col < page:
                            segment[base + col] = 0

    return zero_kv_blocks


# --- Tile form for gather_lookback and zero_kv_blocks (round 5) -------------------------

LOAD_IF = r"""
__device__ __forceinline__ long long rv_load_if_b64(const void* p, long long other, bool ok) {
  long long v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q ld.global.b64 %0, [%1];\n}"
               : "+l"(v) : "l"(p), "r"((int)ok));
  return v;
}
__device__ __forceinline__ int rv_load_if_b32(const void* p, int other, bool ok) {
  int v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n @q ld.global.b32 %0, [%1];\n}"
               : "+r"(v) : "l"(p), "r"((int)ok));
  return v;
}
__device__ __forceinline__ void rv_zero_if_v4(void* p, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %1, 0;\n"
               " @q st.global.v4.b32 [%0], {%2, %2, %2, %2};\n}"
               :: "l"(p), "r"((int)ok), "r"(0) : "memory");
}
"""


@tilelang.jit(pass_configs=PASS)
def lookback_tile_kernel(depth: int, threads: int, spread: bool):
    """``lookback_kernel`` in Triton's form: the element's row and column in int32,
    then the request index, its computed count and the token as predicated loads
    (``@p ld``, the masked lanes keeping 0 / -1), and the store predicated: no
    branch, so each load issues as soon as its address is known."""
    rows = T.dynamic("rows")
    num_reqs = T.dynamic("num_reqs")
    counts_len = T.dynamic("counts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")
    grid = rows if spread else T.ceildiv(rows * depth, threads)

    @T.prim_func
    def gather_lookback(
        lookback: T.Tensor((rows, depth), T.int32),
        idx_mapping: T.Tensor((num_reqs,), T.int64),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
        all_token_ids: T.StridedTensor((history_rows, history_cols), (history_stride, 1), T.int32),
    ):
        with T.Kernel(grid, threads=threads) as bx:
            T.import_source(STORE_IF + LOAD_IF)
            tx = T.get_thread_binding()
            element = T.alloc_var(T.int32, init=bx * threads + tx)
            b = T.alloc_var(T.int32, init=bx if spread else element // depth)
            j = T.alloc_var(T.int32, init=tx if spread else element - b * depth)
            inside = (b < rows) & (j < depth)
            live = inside & (b < num_reqs)
            req = T.alloc_var(T.int64, init=T.call_extern(
                "int64", "rv_load_if_b64", T.address_of(idx_mapping[b]), T.int64(0), live))
            computed = T.alloc_var(T.int32, init=T.call_extern(
                "int32", "rv_load_if_b32", T.address_of(num_computed_tokens[req]), 0, live))
            position = T.alloc_var(T.int32, init=computed - depth + j)
            token = T.alloc_var(T.int32, init=T.call_extern(
                "int32", "rv_load_if_b32", T.address_of(all_token_ids[req, position]), -1,
                live & (position >= 0)))
            T.call_extern("handle", "vllm_store_if_b32", T.address_of(lookback[b, j]), token, inside)

    return gather_lookback


@tilelang.jit(pass_configs=PASS)
def zero_kv_tile_kernel(threads: int, per: int, vector: bool):
    """``zero_kv_kernel`` in Triton's form: only the page size before the early exit,
    then the block id, stride and address together, and every store predicated (no
    branch). ``vector``: 16-byte stores (``per`` / 4 a thread, block-strided) when the
    segment's address, block stride and page allow, else ``per`` int32 stores a
    thread, ``threads`` apart (Triton's layout)."""
    chunk = threads * per
    segments = T.dynamic("segments")
    stride_len = T.dynamic("stride_len")
    page_len = T.dynamic("page_len")
    num_ids = T.dynamic("num_ids")

    @T.prim_func
    def zero_kv_blocks(
        seg_addrs: T.Tensor((segments,), T.ptr),
        seg_block_strides: T.Tensor((stride_len,), T.int64),
        seg_page_sizes: T.Tensor((page_len,), T.int64),
        block_ids: T.Tensor((num_ids,), T.int64),
        chunks: T.int32,
    ):
        with T.Kernel(num_ids, segments, chunks, threads=threads) as (b, s, cx):
            T.import_source(STORE_IF + LOAD_IF)
            tx = T.get_thread_binding()
            page = T.alloc_var(T.int64, init=seg_page_sizes[s])
            first = T.cast(cx, T.int64) * chunk
            if first < page:
                stride = T.alloc_var(T.int64, init=seg_block_strides[s])
                block_id = T.alloc_var(T.int64, init=block_ids[b])
                segment = T.make_tensor(seg_addrs[s], (SPAN,), T.int32)
                base = block_id * stride
                if vector:
                    # Whole 16-byte runs only: the page too a multiple of four.
                    if (T.truncmod(seg_addrs[s], 16) == 0 and T.truncmod(stride, 4) == 0
                            and T.truncmod(page, 4) == 0):
                        for k in T.unroll(per // 4):
                            col = first + (k * threads + tx) * 4
                            T.call_extern("handle", "rv_zero_if_v4",
                                          T.address_of(segment[base + col]), col < page)  # fmt: skip
                    else:
                        for k in T.unroll(per):
                            col = first + k * threads + tx
                            T.call_extern("handle", "vllm_store_if_b32",
                                          T.address_of(segment[base + col]), 0, col < page)  # fmt: skip
                else:
                    for k in T.unroll(per):
                        col = first + k * threads + tx
                        T.call_extern("handle", "vllm_store_if_b32",
                                      T.address_of(segment[base + col]), 0, col < page)  # fmt: skip

    return zero_kv_blocks
