# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang output projection of the DeepSeek-V4.1 attention, as DeepSeek's
reference computes it::

    o[..., -64:] = inverse_rope(o[..., -64:])          (BF16, in place)
    o = act_quant(o, 32, "ue8m0", inplace=True)        (MXFP8 values, back in BF16)
    a = wo_a(o.view(T, groups, -1))                    (grouped, block-32 FP8, BF16 out)
    out = wo_b(a.flatten(1))                           (block-32 FP8, BF16 out)

The first two lines run in the sparse MLA kernels' output store (``sparse_mla.
wo_input``, since the reference's 2026-10-12 revision for both V4.1 Flash
checkpoints), once per row: ``o`` arrives as WO-A's input. Quantizing there instead
of in WO-A's tile loads keeps the GEMM from redoing it for every column tile (the
release on 1017 measured 3% of prefill, 2026-10-17). WO-A is the BF16 einsum on the
weights ``convert.py`` dequantizes: one launch of the grouped W8A16 GEMM expands each
FP8 weight tile exactly to BF16 and accumulates in FP32.
WO-B is the ordinary block-32 projection with MXFP8 activations (DeepSeek's E4M3 and
one UE8M0 scale per 32), as the reference's ``linear`` quantizes them. WO-A's last
kernel (the GEMM's epilogue, or the split-K reduce at decode) rounds ``a`` to BF16 and
applies that cast itself, so ``a`` never reaches memory in BF16: the bytes are those
of ``per_token_cast`` on the BF16 ``a``. The checkpoint tensors are read in place.

Under sequence-parallel prefill on encoder layers, each rank runs WO-A once on all
``T`` rows into its native MXFP8 ``(xq, sf)`` buffers, exchanges those narrower
``o_lora`` rows across ranks while projecting its own WO-B shard on the ``L`` local
rows, and then projects each peer rank's WO-B shard on the ``L`` local rows into
``parts`` (``[world, L, hidden]`` BF16), keeping every rank shard's BF16 rounding and
the rank-order FP32 sum identical to the full-row path.
"""

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.v1.worker.workspace import current_preallocated_workspace

from .. import l2_prefetch
from . import prepare
from .gemm import (
    SPLIT_FP8,
    fp8_prefill_config,
    fp8w_bf16_gemm_cast,
    fp8w_bf16_gemm_partials,
    mxfp8_gemm,
    scale_words,
    splitk_reduce_cast,
)
from .linear import (
    _activation_specs,
    _block32_gemm,
    _block32_scale_words,
    _calls,
    _declare_block32,
    _launch,
    _scratch,
)

# WO-A's W8A16 GEMM sums K in _WO_A_SHARDS shards at every row count, each from zero,
# added in shard order, so a row's bits do not depend on the path. A bucket either
# splits the shards over CTAs (FP32 partials, then splitk_reduce: the SMs a narrow grid
# leaves idle at decode) or adds them inside the kernel (no partials to write and read
# back). Path and tile per bucket from the split sweeps (bits depend on neither): 64
# columns and two stages throughout; four shards beat two at decode, and tiles as tall
# as the bucket from 65 rows (2026-10-13 sweep 2: 72-96 rows 27.9-29.5 against 34.5 us,
# 512 rows 119.1 against 131.8 us, 8192 rows 1675 against 1758 us).
_WO_A_SHARDS = 4
_WO_A_TILES = {
    16: dict(split=True, block_M=16),
    32: dict(split=True, block_M=32),
    64: dict(split=True, block_M=64),
    96: dict(split=True, block_M=96),
    128: dict(split=True, block_M=128),
    None: dict(split=False, block_M=96),
}
_WO_A_COMMON = dict(block_N=64, num_stages=2)


class _Weight:
    """A block-32 FP8 weight and its scales; weak-referenceable for the custom op."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.weight_scale_inv = layer.weight_scale_inv.detach()
        self.groups = groups
        _block32_scale_words(self)

    def declare_kernels(self) -> None:
        """Declare the GEMMs; the caller hands them MXFP8 activations, so there is
        no per-token cast to prepare."""
        _declare_block32(self, self.groups, cast=False)
        # Verification steps stay on decode tiles to 128 rows: one 96-row tile to 96
        # (WIDE_DECODE), two of 64 rows past it.
        self.tilelang_decode_rows = 128


class _PeerWeight:
    """A peer TP rank's WO-B shard and its prepared prefill GEMM."""

    __slots__ = ("weight", "tilelang_weight_sf", "tilelang_prefill")

    def __init__(
        self,
        weight: torch.Tensor,
        tilelang_weight_sf: torch.Tensor,
        tilelang_prefill,
    ) -> None:
        self.weight = weight
        self.tilelang_weight_sf = tilelang_weight_sf
        self.tilelang_prefill = tilelang_prefill


class _GroupedBF16:
    """WO-A for the W8A16 GEMM: the FP8 weight, its UE8M0 exponents (also what the L2
    prefetch reads beside it, as ``tilelang_weight_sf``) and one kernel per row tile."""

    def __init__(self, layer: nn.Module, groups: int):
        self.weight = layer.weight.detach()
        self.tilelang_weight_sf = layer.weight_scale_inv.detach().view(torch.uint8)
        self.columns = self.weight.shape[0]
        self.groups = groups

    def _kernels(self) -> dict:
        """Every kernel of WO-A, as calls: one per row bucket, and the split path's
        reduce."""
        n, k = self.weight.shape
        groups = self.groups
        return {
            "kernels": {
                bucket: prepare.call(
                    fp8w_bf16_gemm_partials,
                    n // groups,
                    k,
                    groups,
                    _WO_A_SHARDS,
                    block_M=tile["block_M"],
                    **_WO_A_COMMON,
                )
                if tile["split"]
                else prepare.call(
                    fp8w_bf16_gemm_cast,
                    n // groups,
                    k,
                    groups,
                    shards=_WO_A_SHARDS,
                    block_M=tile["block_M"],
                    **_WO_A_COMMON,
                )
                for bucket, tile in _WO_A_TILES.items()
            },
            "reduce": prepare.call(splitk_reduce_cast, n, _WO_A_SHARDS),
        }

    def declare_kernels(self) -> None:
        """Declare WO-A's kernels, bound as ``kernels`` and ``reduce``."""
        if self.groups == 0 or self.columns == 0:
            from . import columns

            prepare.declare(lambda: [prepare.call(columns.zero_packs_kernel)])
            return
        kernels = self._kernels()

        def bind() -> None:
            for name, kernel in _launch(kernels).items():
                setattr(self, name, kernel)

        prepare.declare(lambda: _calls(kernels), bind=bind)

    @staticmethod
    def bucket(rows: int):
        return next((b for b in _WO_A_TILES if b is not None and rows <= b), None)

    def partial_specs(self, rows: int):
        """The split path's FP32 partials (nothing for the in-kernel path)."""
        if self.columns == 0 or not _WO_A_TILES[self.bucket(rows)]["split"]:
            return ()
        return (((_WO_A_SHARDS, rows, self.columns), torch.float32),)

    def __call__(self, x, xq, sf, partials=None) -> None:
        """WO-A of ``x``, cast as the reference's ``act_quant`` into ``xq`` (E4M3) and
        ``sf`` (UE8M0 bytes); ``partials`` from ``partial_specs`` on the split path."""
        bucket = self.bucket(x.shape[0])
        if not _WO_A_TILES[bucket]["split"]:
            self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, xq, sf)
            return
        self.kernels[bucket](x, self.weight, self.tilelang_weight_sf, partials)
        self.reduce(partials, xq, sf)


class TileLangWOProjection:
    """WO-A and WO-B of one attention layer (built after loading; its layer declares
    the kernels with ``declare_kernels``)."""

    def __init__(
        self,
        wo_a: nn.Module,
        wo_b: nn.Module,
        *,
        groups: int,
        rank_width: int | None = None,
    ):
        self.a = _GroupedBF16(wo_a, groups)
        self.b = _Weight(wo_b, 1)
        if SPLIT_FP8.get(tuple(self.b.weight.shape)):
            raise ValueError("WO-B takes WO-A's MXFP8 output: no split-K path")
        self.hidden = wo_b.weight.shape[0]
        self.rank_width = (
            int(rank_width)
            if rank_width is not None
            else (
                self.a.columns // groups
                if groups
                else int(getattr(wo_a, "o_lora_rank", 1024))
            )
        )
        self.sp_rank_groups: tuple[int, ...] = ()
        self.sp_shards: tuple[_Weight | _PeerWeight, ...] | None = None

    def enable_sp(self, rank_groups: tuple[int, ...]) -> None:
        """Record every TP rank's trimmed WO group count for SP local projection."""
        self.sp_rank_groups = tuple(int(g) for g in rank_groups)

    def _sp_prefill_calls(self) -> dict[int, prepare.TileLangCall]:
        if not self.sp_rank_groups:
            return {}
        rank_width = self.rank_width
        calls: dict[int, prepare.TileLangCall] = {}
        for groups in dict.fromkeys(self.sp_rank_groups):
            if groups <= 0:
                continue
            cols = groups * rank_width
            calls[cols] = prepare.call(
                mxfp8_gemm,
                self.hidden,
                cols,
                **fp8_prefill_config(self.hidden, cols, 1),
                groups=1,
            )
        return calls

    def declare_kernels(self) -> None:
        """Declare WO-A's and WO-B's kernels, plus SP peer WO-B shards when enabled."""
        self.a.declare_kernels()
        self.b.declare_kernels()
        if not self.sp_rank_groups:
            return
        from . import collectives, columns

        sp_calls = self._sp_prefill_calls()
        world = len(self.sp_rank_groups)
        extra_calls = [
            *sp_calls.values(),
            prepare.call(columns.zero_packs_kernel),
            prepare.call(collectives.rank_order_sum_kernel, world),
        ]

        def bind() -> None:
            # Decode and non-SP steps retain their original TP shard and kernels.
            # SP encoder layers gather peer WO-B shards once before memory profiling
            # and reuse this rank's own shard in place.
            rank = get_tensor_model_parallel_rank()
            rank_width = self.rank_width
            max_cols = max(self.sp_rank_groups) * rank_width
            max_sf_words = scale_words(max_cols)
            device = self.b.weight.device
            w = self.b.weight
            if w.shape[1] < max_cols:
                padded_w = torch.zeros(
                    (self.hidden, max_cols), dtype=w.dtype, device=device
                )
                padded_w[:, : w.shape[1]].copy_(w)
            else:
                padded_w = w
            sf = self.b.tilelang_weight_sf
            if sf.shape[1] < max_sf_words:
                padded_sf = torch.zeros(
                    (self.hidden // 32, max_sf_words), dtype=sf.dtype, device=device
                )
                padded_sf[:, : sf.shape[1]].copy_(sf)
            else:
                padded_sf = sf
            gathered_w = tensor_model_parallel_all_gather(padded_w, dim=0).view(
                world, self.hidden, max_cols
            )
            gathered_sf = tensor_model_parallel_all_gather(padded_sf, dim=0).view(
                world, self.hidden // 32, max_sf_words
            )
            prefill_by_cols = {
                cols: prepare.launch(call) for cols, call in sp_calls.items()
            }
            shards: list[_Weight | _PeerWeight] = []
            for peer, peer_groups in enumerate(self.sp_rank_groups):
                if peer == rank:
                    shards.append(self.b)
                    continue
                cols = peer_groups * rank_width
                sf_words = scale_words(cols)
                shards.append(
                    _PeerWeight(
                        gathered_w[peer, :, :cols].contiguous(),
                        gathered_sf[peer, :, :sf_words].contiguous(),
                        prefill_by_cols.get(cols),
                    )
                )
            self.sp_shards = tuple(shards)

        prepare.declare(lambda: extra_calls, bind=bind)

    def l2_segments(self) -> list:
        """The bytes both GEMMs read at decode, WO-A first: each weight and its scale words."""
        return l2_prefetch.linear_segments(
            "wo_a", self.a
        ) + l2_prefetch.linear_segments("wo_b", self.b)

    def can_project_sp(self, sp_rows) -> bool:
        """Whether SP local WO-B shards are bound for ``sp_rows``."""
        return (
            self.sp_shards is not None
            and len(self.sp_shards) == sp_rows.world_size
            and 0 <= sp_rows.rank < sp_rows.world_size
        )

    def _project_sp_shard(
        self,
        shard_rank: int,
        xq: torch.Tensor,
        sf: torch.Tensor,
        out: torch.Tensor,
        real: int,
        local: int,
    ) -> None:
        from . import columns

        assert self.sp_shards is not None
        shard = self.sp_shards[shard_rank]
        if shard.weight.shape[1] == 0:
            if local:
                columns.zero_(out[:local])
            return
        if real:
            weight = getattr(shard, "tilelang_weight", shard.weight)
            # Always use the prefill GEMM, even for a short local slice: it keeps
            # the global prefill's K reduction and reads only the real rows.
            shard.tilelang_prefill(
                xq[:real],
                weight,
                sf[:real].view(torch.uint32),
                shard.tilelang_weight_sf,
                out[:real],
            )
        if real < local:
            columns.zero_(out[real:local])

    def project_sp(self, o: torch.Tensor, sp_rows, *, then=None) -> torch.Tensor:
        """Project ``o`` on all ``T`` rows through WO-A, exchange the native MXFP8
        ``o_lora`` rows across ranks while projecting this rank's own WO-B shard on
        local rows, and project the peer WO-B shards on local rows into rank-order
        ``parts`` (``[world, L, hidden]`` BF16, or their rank-order sum when
        ``sp_rows.unreduced`` is False)."""
        from .. import sp_prefill
        from . import collectives, columns

        rows = o.shape[0]
        if rows != sp_rows.num_tokens:
            raise ValueError(f"V4.1 SP WO expects {sp_rows.num_tokens} rows")
        if not self.can_project_sp(sp_rows):
            raise RuntimeError("SP WO shards must be bound before project_sp")
        assert self.sp_shards is not None
        world, rank = sp_rows.world_size, sp_rows.rank
        local, padded = sp_rows.local_rows, sp_rows.padded_rows
        real = sp_rows.real_rows
        device = o.device
        cols_r = self.a.columns
        sf_bytes_r = 4 * scale_words(cols_r)
        xq_send = torch.empty(
            (world, local, cols_r), dtype=torch.float8_e4m3fn, device=device
        )
        sf_send = torch.empty(
            (world, local, sf_bytes_r), dtype=torch.uint8, device=device
        )
        xq_flat = xq_send.view(padded, cols_r)
        sf_flat = sf_send.view(padded, sf_bytes_r)
        if cols_r:
            if rows:
                partial_specs = self.a.partial_specs(rows)
                partials = (
                    _scratch(partial_specs, current_preallocated_workspace())[0]
                    if partial_specs
                    else None
                )
                self.a(o.view(rows, -1), xq_flat[:rows], sf_flat[:rows], partials)
            if rows < padded:
                columns.zero_(xq_flat[rows:])
                columns.zero_(sf_flat[rows:])

        if len(set(self.sp_rank_groups)) == 1:
            xq_recv: torch.Tensor | list[torch.Tensor] = torch.empty_like(xq_send)
            sf_recv: torch.Tensor | list[torch.Tensor] = torch.empty_like(sf_send)
        else:
            xq_recv = [
                xq_send[rank]
                if peer == rank
                else torch.empty(
                    (local, self.sp_shards[peer].weight.shape[1]),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                for peer in range(world)
            ]
            sf_recv = [
                sf_send[rank]
                if peer == rank
                else torch.empty(
                    (local, 4 * scale_words(self.sp_shards[peer].weight.shape[1])),
                    dtype=torch.uint8,
                    device=device,
                )
                for peer in range(world)
            ]

        parts = torch.empty(
            (world, local, self.hidden), dtype=torch.bfloat16, device=device
        )
        if device.type == "cuda":
            main = torch.cuda.current_stream(device)
            side = sp_prefill._side_stream(device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                sp_rows.exchange_into(xq_send, xq_recv)
                sp_rows.exchange_into(sf_send, sf_recv)
            self._project_sp_shard(
                rank, xq_send[rank], sf_send[rank], parts[rank], real, local
            )
            main.wait_stream(side)
        else:
            sp_rows.exchange_into(xq_send, xq_recv)
            sp_rows.exchange_into(sf_send, sf_recv)
            self._project_sp_shard(
                rank, xq_send[rank], sf_send[rank], parts[rank], real, local
            )
        for peer in range(world):
            if peer != rank:
                self._project_sp_shard(
                    peer, xq_recv[peer], sf_recv[peer], parts[peer], real, local
                )
        if then is not None:
            then()
        if sp_rows.unreduced:
            return parts
        return collectives.rank_order_sum(list(parts.unbind(0)))

    def __call__(self, o: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Project WO-A's input ``o`` ([T, heads, head_dim] contiguous BF16, as the
        sparse MLA kernels store it) into ``out`` ([T, hidden] BF16)."""
        rows = o.shape[0]
        if not rows:
            return out
        if self.a.columns == 0:
            from . import columns

            columns.zero_(out[:rows])
            return out
        scratch = current_preallocated_workspace()
        # One borrow for everything live at once: WO-A's partials, then WO-B's MXFP8
        # input in whole decode tiles.
        partial_specs = self.a.partial_specs(rows)
        views = _scratch(
            (
                *partial_specs,
                *_activation_specs(rows, self.a.columns, self.b.tilelang_decode_rows),
            ),
            scratch,
        )
        partials = views[0] if partial_specs else None
        xq, sf = views[-2], views[-1]
        self.a(o.view(rows, -1), xq[:rows], sf[:rows], partials)
        _block32_gemm(self.b, xq, sf, out, rows)
        return out


__all__ = ["TileLangWOProjection"]
