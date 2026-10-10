# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill sequence parallelism (SP) for DeepSeek V4.1 tensor parallelism.

## Mode

At tensor parallel size N every rank repeats the row-wise work of a decoder
layer (mHC mixes, norms, the Engram gate, the residual carry) on every token.
For a long prefill forward, SP turns the per-layer all-reduces into dim-0
reduce-scatters so that each rank owns 1/N of the rows for that work, and
all-gathers the rows back before the ops that need all of them. Weights stay
exactly as in ordinary TP; nothing is replicated. The static
``_use_sequence_parallel`` construction flag stays off: this mode changes no
module.

SP trades two all-gathers per layer for reduce-scatters that split the
row-wise work across the ranks, so a forward runs sequence-parallel exactly
when that trade can pay; there is no setting. It must be prompt processing:
decode forwards (at most the largest CUDA graph capture size, and at most
``max_num_seqs * (1 + num_speculative_tokens)`` tokens) replay captured
graphs and keep full rows. And its reduce-scatter must really scatter: a
conservative hidden-state byte threshold comes from the TP group's
registered one-shot RoCE capacity. This preserves the qualified SP boundary
when transport dispatch is tuned independently below that capacity. It is
a performance policy, not a claim that every smaller message uses RoCE.
``DeepseekV4Model.forward`` decides per forward from the token count T,
which every rank shares, so no agreement between ranks is needed. SP is off
only where it cannot apply: tensor parallel size 1, pipeline parallelism,
static sequence parallelism and request boundary checkpoints. With CED
compaction, only the encoder layers before ``ced_decoder_start`` run on
every row: SP covers those, and the carried rows are gathered back in full
before the boundary layer, which builds the decoder's global KV from every
row and then compacts.

## Rows

- Collectives see ``Tp = N * ceil(T / N)`` rows; each rank owns ``L = Tp / N``
  rows, rank r the padded rows ``[r * L, (r + 1) * L)``.
- Kernels never see more than T rows: gathered tensors are used as ``[:T]``.
- Reduce-scatter inputs are Tp-row buffers whose pad rows are zero.
- The last rank's local rows can include pad rows. They stay finite and are
  never read back into real outputs.

## Where each op runs

- Local rows (L): the residual carry, mHC pre / post / post_pre / collapse,
  attn_norm, ffn_norm, the Engram gate (``run_engram_mix``) and the final norm.
- Gathered rows (T): the Engram lookup reduction and ``wkv``, attention
  (``wq_b``, indexer, compressor, KV cache writes, sparse MLA, WO), and the
  MoE gate, top-k, routed and shared experts.
- In attention layers without a compressor, the row-wise front
  (``fused_wqa_wkv``, the q/kv norms and the index weights) runs on local
  rows and its narrower product is gathered instead of the hidden state.

Per decoder layer: all-gather before attention (inside
``DeepseekV4Attention._forward``, behind the opaque attention op, which reads
the rows from ``current()``), reduce-scatter after WO instead of the
all-reduce, all-gather before the MoE, and reduce-scatter of the MoE output
(shared + routed) instead of the runner's final all-reduce. The embedding
reduce-scatters its masked partial rows (each token has one owning
vocabulary shard, so the sum is exact); ``inputs_embeds`` are sliced. After
the final collapse and norm, and for every aux hidden state a drafter
consumes, the rows are gathered back to T. DSpark drafts from the aux hidden
states only, so its pre-collapse MTP buffer is not filled under SP; other
drafters gather it.

## Unreduced results

With ``UNREDUCED`` and the TP group's rank-order reduce-scatter (DS4.1's TileLang
family on the RoCE fabric), the attention's WO output and the MoE output skip the
sum: every rank's partial of this rank's rows is exchanged into one
``(N, L, hidden)`` buffer, this rank's own written in place, and the decoder layer
hands it on as ``RankOrderParts``. TileLang mHC's ``post_pre`` adds the parts in
rank order as it loads them, the same arithmetic and the same bits as the
rank-order reduce-scatter; any other consumer calls ``sum()``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from vllm.distributed import (
    get_tp_group,
    model_parallel_is_initialized,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _prefill_sp_capacity_bytes() -> int:
    """Registered RoCE capacity used as the conservative SP floor; 0 if none."""
    if not model_parallel_is_initialized():
        return 0
    comm = getattr(get_tp_group().device_communicator, "b12x_ar_comm", None)
    return int(getattr(comm, "all_reduce_capacity_bytes", 0)) if comm is not None else 0


def rows_above_bytes(limit: int, row_bytes: int, world_size: int) -> int:
    """The fewest tokens T whose ``Tp``-row message exceeds ``limit`` bytes."""
    padded = -(-(limit // row_bytes + 1) // world_size) * world_size
    return padded - world_size + 1


def resolve_min_rows(vllm_config, *, static_sequence_parallel: bool = False) -> int:
    """The smallest forward, in tokens, that runs sequence-parallel; 0 is never.

    The larger of: one token more than any decode forward (the largest CUDA
    graph capture size, ``max_num_seqs * (1 + num_speculative_tokens)``), and
    the fewest tokens whose padded hidden-state message exceeds the
    registered RoCE capacity. Dispatch tuning must not also retune SP.
    """
    parallel = vllm_config.parallel_config
    reason = None
    if parallel.tensor_parallel_size <= 1:
        reason = "tensor parallel size is 1"
    elif parallel.pipeline_parallel_size != 1:
        reason = "pipeline parallelism keeps full-row intermediate tensors"
    elif static_sequence_parallel:
        reason = "static sequence parallelism is enabled"
    elif vllm_config.use_request_boundary_checkpoints:
        reason = "request boundary checkpoints are enabled"
    if reason is not None:
        logger.info("DeepSeek V4.1 prefill sequence parallelism off: %s", reason)
        return 0
    spec = vllm_config.speculative_config
    decode_rows = max(
        vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        vllm_config.scheduler_config.max_num_seqs
        * (1 + (spec.num_speculative_tokens if spec is not None else 0)),
        parallel.tensor_parallel_size,
    )
    rows = decode_rows + 1
    limit = _prefill_sp_capacity_bytes()
    if limit > 0:
        model = vllm_config.model_config
        row_bytes = model.get_hidden_size() * model.dtype.itemsize
        tp = parallel.tensor_parallel_size
        rows = max(rows, rows_above_bytes(limit, row_bytes, tp))
    logger.info(
        "DeepSeek V4.1 prefill sequence parallelism from %d tokens "
        "(capacity floor=%d bytes; independent of all-reduce dispatch)",
        rows,
        limit,
    )
    return rows


@dataclass(frozen=True)
class SPRows:
    """Row layout of one sequence-parallel forward on one TP rank."""

    num_tokens: int
    world_size: int
    rank: int
    # Results stay unreduced (see "Unreduced results").
    unreduced: bool = False

    @property
    def padded_rows(self) -> int:
        """Tp: rows seen by collectives, T rounded up to a multiple of N."""
        return -(-self.num_tokens // self.world_size) * self.world_size

    @property
    def local_rows(self) -> int:
        """L: rows this rank owns between collectives."""
        return self.padded_rows // self.world_size

    @property
    def start(self) -> int:
        """First padded row owned by this rank."""
        return self.rank * self.local_rows

    @property
    def real_rows(self) -> int:
        """Leading local rows that are real tokens; the rest are padding."""
        return max(0, min(self.local_rows, self.num_tokens - self.start))

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather this rank's L rows into the first T rows."""
        x = materialize(x)
        if x.shape[0] != self.local_rows:
            raise ValueError(f"SP gather expects {self.local_rows} rows, got {x.shape}")
        return tensor_model_parallel_all_gather(x, dim=0)[: self.num_tokens]

    def reduce_scatter(self, padded: torch.Tensor) -> torch.Tensor:
        """Sum a zero-padded Tp-row buffer over the ranks into this rank's rows."""
        if padded.shape[0] != self.padded_rows:
            raise ValueError(
                f"SP reduce-scatter expects {self.padded_rows} rows, got {padded.shape}"
            )
        return tensor_model_parallel_reduce_scatter(padded, dim=0)

    def empty_padded(
        self, row_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """A Tp-row buffer whose pad rows (T and beyond) are zero."""
        padded = torch.empty((self.padded_rows, *row_shape), dtype=dtype, device=device)
        padded[self.num_tokens :].zero_()
        return padded

    def local_slice(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's L rows of a T-row tensor; pad rows are zero."""
        if x.shape[0] != self.num_tokens:
            raise ValueError(f"SP slice expects {self.num_tokens} rows, got {x.shape}")
        real = self.real_rows
        rows = x[self.start : self.start + real]
        if real == self.local_rows:
            return rows
        local = x.new_zeros((self.local_rows, *x.shape[1:]))
        local[:real].copy_(rows)
        return local

    def project_reduce_scatter(
        self,
        project,
        row_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        slices: int = 1,
        then=None,
    ) -> torch.Tensor:
        """This rank's L rows of the sum over ranks of a T-row projection.

        ``project(start, stop, out)`` writes rows ``start:stop`` of the projection
        into ``out``. With ``slices`` > 1 each rank's block is projected in that many
        pieces, every piece of every block into one contiguous buffer, and each
        piece is reduce-scattered on a side stream while the next is projected.
        A projection whose rows do not depend on one another reduces to the same
        bits as one projection and one reduce-scatter. ``then()`` runs on the
        current stream after the last projection, before the last reduce-scatter
        is awaited.

        With ``unreduced`` rows the result is every rank's unreduced partial of
        this rank's rows instead, ``(N, L, *row_shape)``; see "Unreduced results".
        """
        if self.unreduced:
            return self._project_exchange(
                project, row_shape, dtype, device, slices, then
            )
        if slices <= 1:
            padded = self.empty_padded(row_shape, dtype, device)
            project(0, self.num_tokens, padded[: self.num_tokens])
            if then is not None:
                then()
            return self.reduce_scatter(padded)
        local, world, tokens = self.local_rows, self.world_size, self.num_tokens
        out = torch.empty((local, *row_shape), dtype=dtype, device=device)
        bounds = [local * i // slices for i in range(slices + 1)]
        main = torch.cuda.current_stream(device)
        side = _side_stream(device)
        for lo, hi in zip(bounds, bounds[1:]):
            if hi == lo:
                continue
            piece = torch.empty(
                (world, hi - lo, *row_shape), dtype=dtype, device=device
            )
            for owner in range(world):
                start, stop = owner * local + lo, owner * local + hi
                real = max(0, min(stop, tokens) - start)
                if real:
                    project(start, start + real, piece[owner, :real])
                if real < hi - lo:
                    piece[owner, real:].zero_()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                reduced = tensor_model_parallel_reduce_scatter(
                    piece.view(world * (hi - lo), *row_shape), dim=0
                )
                out[lo:hi].copy_(reduced)
            piece.record_stream(side)
        if then is not None:
            then()
        main.wait_stream(side)
        return out

    def _project_exchange(self, project, row_shape, dtype, device, slices, then):
        """``project_reduce_scatter`` without the sum: each rank's block goes into
        its slot of one send buffer, this rank's own straight into its slot of
        the parts, and the chunks are exchanged into the parts; with ``slices``
        > 1 each slice is exchanged on the side stream while the next is
        projected."""
        from vllm.models.deepseek_v4_1.tilelang.collectives import exchange_into

        local, world, tokens = self.local_rows, self.world_size, self.num_tokens
        send = torch.empty((world, local, *row_shape), dtype=dtype, device=device)
        parts = torch.empty_like(send)
        comm = get_tp_group().device_communicator.pynccl_comm
        slices = max(slices, 1)
        bounds = [local * i // slices for i in range(slices + 1)]
        main = torch.cuda.current_stream(device)
        side = _side_stream(device) if slices > 1 else None
        for lo, hi in zip(bounds, bounds[1:]):
            if hi == lo:
                continue
            for owner in range(world):
                dest = parts if owner == self.rank else send
                start, stop = owner * local + lo, owner * local + hi
                real = max(0, min(stop, tokens) - start)
                if real:
                    project(start, start + real, dest[owner, lo : lo + real])
                if real < hi - lo:
                    dest[owner, lo + real : hi].zero_()
            if side is not None:
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    exchange_into(
                        comm, send[:, lo:hi], parts[:, lo:hi], self.rank, world
                    )
        if then is not None:
            then()
        if side is None:
            exchange_into(comm, send, parts, self.rank, world)
        else:
            # send and parts are freed only after this wait, on this stream.
            main.wait_stream(side)
        return parts


# The attention's WO projection is reduce-scattered in two slices from this many
# tokens, so half its reduce-scatter runs under the other half's projection.
WO_SLICE_TOKENS = 2048


def wo_slices(num_tokens: int) -> int:
    """Slices of a sequence-parallel WO projection of ``num_tokens`` tokens."""
    return 2 if num_tokens >= WO_SLICE_TOKENS else 1


# Results of sequence-parallel steps stay unreduced where the TP group's
# reduce-scatter is the rank-order one (see "Unreduced results").
UNREDUCED = False


def unreduced_enabled() -> bool:
    """Whether this process's SP steps keep their results unreduced."""
    if not UNREDUCED or not model_parallel_is_initialized():
        return False
    comm = get_tp_group().device_communicator
    return bool(getattr(comm, "rank_order_reduce_scatter", False))


def materialize(x):
    """``x``, or the rank-order sum of its parts."""
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

    return x.sum() if isinstance(x, RankOrderParts) else x


def unreduced_parts(x: torch.Tensor):
    """A packed ``(N, L, ...)`` result of an unreduced step as ``RankOrderParts``."""
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts

    return RankOrderParts(x)


def project_reduce_scatter_current(
    project, rows, padded_rows, row_shape, dtype, device, then=None
) -> torch.Tensor:
    """The MoE runner's reduce-scatter hook: ``project_reduce_scatter`` over the
    rows of the active SP forward (``activate``)."""
    sp = current()
    if sp is None or sp.num_tokens != rows or sp.padded_rows != padded_rows:
        raise RuntimeError(
            f"SP reduce-scatter of {rows} rows ({padded_rows} padded) outside its "
            f"forward ({sp})"
        )
    return sp.project_reduce_scatter(project, row_shape, dtype, device, then=then)


_SIDE_STREAMS: dict = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    """One stream per device for the reduce-scatters of sliced projections."""
    if device not in _SIDE_STREAMS:
        _SIDE_STREAMS[device] = torch.cuda.Stream(device)
    return _SIDE_STREAMS[device]


def plan_rows(
    num_tokens: int,
    min_rows: int,
    world_size: int,
    rank: int,
    unreduced: bool = False,
) -> SPRows | None:
    """The SP row layout for a forward of ``num_tokens`` tokens, or None."""
    if min_rows <= 0 or world_size <= 1 or num_tokens < min_rows:
        return None
    if not 0 <= rank < world_size:
        raise ValueError(f"TP rank {rank} is outside world size {world_size}")
    return SPRows(num_tokens, world_size, rank, unreduced)


_ROWS: ContextVar[SPRows | None] = ContextVar("ds41_prefill_sp_rows", default=None)


def current() -> SPRows | None:
    """The SP rows of the attention call in progress, if any."""
    return _ROWS.get()


@contextmanager
def activate(rows: SPRows) -> Iterator[None]:
    """Expose ``rows`` across the opaque attention op and MoE runner boundaries."""
    token = _ROWS.set(rows)
    try:
        yield
    finally:
        _ROWS.reset(token)
