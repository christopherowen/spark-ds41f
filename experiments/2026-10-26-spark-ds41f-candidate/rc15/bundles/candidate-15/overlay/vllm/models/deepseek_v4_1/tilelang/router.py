# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 expert routing with DeepSeek's TileLang top-k gate.

The TileLang kernel family routes both the target's MoE layers (6 of 384
experts) and the drafter's (3 of 128) with TileKernels'
``moe_topk_gate_forward``, in place of vLLM's Triton ``dsv4_topk`` and its
CUDA ``topk_hash_softplus_sqrt`` op. It implements DeepSeek's V4.1 reference
routing:

- scores ``sqrt(softplus(logits))``, with softplus the identity above 20;
- the top-k experts by score plus the correction bias, or the vision bias for
  image tokens, the lowest expert id first on ties;
- weights the chosen scores without bias, divided by their sum plus 1e-20 and
  multiplied by the routed scaling factor.

One warp routes one token. The kernel refuses NaN or infinite logits instead
of routing on them. Rows that pad a CUDA graph's batch are not routed: they
get expert -1 and weight 0, which the dispatch skips (``VLLM_MOE_SKIP_PADDING``,
on by default, as vLLM's routers do). V4.1 has no hash-routed layers.

``TileLangGateRouter`` also owns the gate projection. At decode the gate runs
as split-K FP32 partials, and ``route_partials`` adds them in shard order,
which reproduces ``splitk_reduce``'s logits bit for bit. It then routes them
with TileKernels' arithmetic in the same launch, so the logits never reach
memory and each MoE layer runs one launch fewer. At prefill the gate runs its
own GEMM and TileKernels routes the logits.
"""

import contextlib
import functools
from typing import NamedTuple

import tilelang
import tilelang.language as T
import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.models.deepseek_v4_1.tilelang import pdl, prepare

_MASK_THREADS = 256


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def routing_masks_kernel(image_ids: str | None, route: bool):
    """TileKernels' masks of a forward's rows, which torch computed once a forward
    in up to four launches, in one: with ``image_ids`` (the input ids' dtype), the
    image tokens ``lo <= id < lo + count`` (``(input_ids >= lo) & (input_ids < lo +
    count)``, compared in int64); with ``route``, the rows to route,
    ``~is_padding``. A thread a row; ``rows`` is the longer of the two."""
    if image_ids is None and not route:
        raise ValueError("routing masks: neither image tokens nor routed rows")
    ids_len, pad_len = T.dynamic("ids_len, pad_len")
    # Without image tokens, the ids' place is held by a mask, never read.
    ids_dtype = T.bool if image_ids is None else image_ids

    @T.prim_func
    def routing_masks(
        input_ids: T.Tensor((ids_len,), ids_dtype),
        image: T.Tensor((ids_len,), T.bool),
        is_padding: T.Tensor((pad_len,), T.bool),
        routed: T.Tensor((pad_len,), T.bool),
        lo: T.int64,
        count: T.int64,
        rows: T.int32,
    ):
        with T.Kernel(T.ceildiv(rows, _MASK_THREADS), threads=_MASK_THREADS) as b:
            pdl.chain()
            i = b * _MASK_THREADS + T.get_thread_binding()
            if image_ids is not None:  # noqa: SIM102
                if i < ids_len:
                    image[i] = (T.cast(input_ids[i], T.int64) >= lo) & (
                        T.cast(input_ids[i], T.int64) < lo + count
                    )
            if route:  # noqa: SIM102
                if i < pad_len:
                    routed[i] = T.if_then_else(
                        is_padding[i], T.bool(False), T.bool(True)
                    )

    return routing_masks


_IDS_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


def _masks(
    input_ids: torch.Tensor | None, lo: int, count: int, is_padding: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """The image-token mask of ``input_ids`` and the routed rows of ``is_padding``
    (each None without its input), in one ``routing_masks_kernel`` launch."""
    image = None if input_ids is None else torch.empty_like(input_ids, dtype=torch.bool)
    routed = None if is_padding is None else torch.empty_like(is_padding)
    rows = max(
        0 if input_ids is None else input_ids.numel(),
        0 if is_padding is None else is_padding.numel(),
    )
    if rows:
        kernel = routing_masks_kernel(
            None if input_ids is None else _IDS_DTYPES[input_ids.dtype],
            is_padding is not None,
        )
        # A mask the launch does not compute takes the other's tensors, never read.
        kernel(
            input_ids if input_ids is not None else is_padding,
            image if image is not None else routed,
            is_padding if is_padding is not None else image,
            routed if routed is not None else image,
            lo,
            count,
            rows,
        )
    return image, routed


def _cached_image_mask(context, input_ids: torch.Tensor) -> torch.Tensor | None:
    cached = getattr(context, "_ds41_image_token_mask", None)
    return cached[1] if cached is not None and cached[0] is input_ids else None


def _image_token_mask(input_ids: torch.Tensor, lo: int, count: int) -> torch.Tensor:
    """Image sentinel tokens of this step, computed once and shared by every layer."""
    context = get_forward_context() if is_forward_context_available() else None
    mask = _cached_image_mask(context, input_ids)
    if mask is None:
        mask, _ = _masks(input_ids, lo, count, None)
        if context is not None:
            context._ds41_image_token_mask = (input_ids, mask)
    assert mask is not None
    return mask


def _ced_routing_mask(context, rows: int, ced_decoder: bool) -> torch.Tensor | None:
    """A CED decoder layer's rows to route: it routes compact rows, so its mask
    comes from the decoder metadata (gathered by the CED indices), not from the
    step's. None for other layers, or without CED metadata."""
    compact_masks = getattr(context, "_ds41_ced_routing_masks", None)
    if compact_masks is None:
        compact_masks = {}
        metadata = context.attn_metadata
        if isinstance(metadata, dict):
            for value in metadata.values():
                decoder = getattr(value, "decoder", None)
                mask = getattr(decoder, "routing_mask", None)
                if mask is not None:
                    compact_masks[mask.numel()] = mask
        context._ds41_ced_routing_masks = compact_masks
    if ced_decoder and compact_masks:
        if rows not in compact_masks:
            raise RuntimeError("CED routing mask does not match the decoder rows")
        return compact_masks[rows]
    return None


def _local_mask(mask: torch.Tensor | None, rows: int, sp_rows) -> torch.Tensor | None:
    """``mask``'s first ``rows`` rows, or this rank's ``sp_rows.local_rows`` slice."""
    if mask is None:
        return None
    if sp_rows is not None and mask.shape[0] != rows:
        return sp_rows.local_slice(mask[: sp_rows.num_tokens])
    return mask if mask.shape[0] == rows else mask[:rows]


def _known_routing_mask(
    rows: int, ced_decoder: bool, sp_rows=None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(mask, None)`` when this step's rows to route are known (a CED decoder's,
    or the step's already computed), ``(None, is_padding)`` when the step's padding
    mask is still to be negated, ``(None, None)`` when no row is skipped."""
    if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
        return None, None
    context = get_forward_context()
    ced = _ced_routing_mask(context, rows, ced_decoder)
    if ced is not None:
        return ced, None
    is_padding = context.is_padding
    if is_padding is None:
        return None, None
    cached = getattr(context, "_ds41_routing_mask", None)
    if cached is not None and cached[0] is is_padding:
        return _local_mask(cached[1], rows, sp_rows), None
    return None, is_padding


def _routing_mask(rows: int, *, ced_decoder: bool = False) -> torch.Tensor | None:
    """Rows to route this step (False pads the batch), computed once per layout."""
    mask, is_padding = _known_routing_mask(rows, ced_decoder)
    if is_padding is None:
        return mask
    _, routed = _masks(None, 0, 0, is_padding)
    get_forward_context()._ds41_routing_mask = (is_padding, routed)
    assert routed is not None
    return routed[:rows]


def _routing_masks(
    rows: int,
    input_ids: torch.Tensor | None,
    lo: int,
    count: int,
    *,
    ced_decoder: bool = False,
    sp_rows=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``(_image_token_mask(input_ids, lo, count), _routing_mask(rows))``, the
    image mask None without ``input_ids``: what a forward's first routing layer
    computes of them, in one launch, and later layers find cached."""
    context = get_forward_context() if is_forward_context_available() else None
    image = None if input_ids is None else _cached_image_mask(context, input_ids)
    route, is_padding = _known_routing_mask(rows, ced_decoder, sp_rows=sp_rows)
    ids = input_ids if image is None else None
    if ids is not None or is_padding is not None:
        new_image, routed = _masks(ids, lo, count, is_padding)
        if ids is not None:
            image = new_image
            if context is not None:
                context._ds41_image_token_mask = (ids, image)
        if is_padding is not None:
            assert context is not None and routed is not None
            context._ds41_routing_mask = (is_padding, routed)
            route = _local_mask(routed, rows, sp_rows)
    return _local_mask(image, rows, sp_rows), route


def _partials_mask(
    rows: int, *, ced_decoder: bool = False, sp_rows=None
) -> tuple[torch.Tensor | None, bool]:
    """``route_partials``' mask of this step's rows and whether it is the padding
    mask itself (a set row is skipped; nothing negates it) or a CED decoder's
    routing mask (a clear row is skipped)."""
    if not envs.VLLM_MOE_SKIP_PADDING or not is_forward_context_available():
        return None, False
    context = get_forward_context()
    ced = _ced_routing_mask(context, rows, ced_decoder)
    if ced is not None:
        return ced, False
    is_padding = context.is_padding
    if is_padding is None:
        return None, False
    return _local_mask(is_padding, rows, sp_rows), True


def _routing_mask_variants(vllm_config) -> tuple[bool, ...]:
    """Whether ``_routing_mask`` can return a mask (True) or None (False).

    A mask needs ``VLLM_MOE_SKIP_PADDING``. The V2 model runner gives every
    forward, the drafter's included, its padding rows; the V1 runner gives none,
    so only a CED decoder layer's compact rows have a mask there. Without a
    config (kernel tests) either can come.
    """
    if not envs.VLLM_MOE_SKIP_PADDING:
        return (False,)
    if vllm_config is not None and vllm_config.use_v2_model_runner:
        return (True,)
    return (False, True)


def _topk_gate(
    logits: torch.Tensor,
    num_topk: int,
    routed_scaling_factor: float,
    mask: torch.Tensor | None,
    bias: torch.Tensor,
    image_bias: torch.Tensor | None,
    image_token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TileKernels' top-k gate as V4.1 routes: sqrtsoftplus with a correction
    bias and no shared experts."""
    return pdl.moe_topk_gate_forward(
        logits,
        num_topk,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=0,
        mask=mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
    )


@functools.cache
def _warm_topk_gate(num_topk: int, num_experts: int, image: bool, mask: bool) -> None:
    """TileKernels' top-k gate of ``num_topk`` of ``num_experts``, run once on a zero
    row with the inputs the router passes (an image bias and mask, a routing mask),
    so its kernel (keyed by which inputs exist, not by the rows) is compiled before
    the cache freezes."""
    logits = torch.zeros((1, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
    flags = torch.zeros(1, dtype=torch.bool, device="cuda")
    _topk_gate(
        logits,
        num_topk,
        1.0,
        mask=~flags if mask else None,
        bias=bias,
        image_bias=bias if image else None,
        image_token_mask=flags if image else None,
    )


class TileKernelsRouter(FusedTopKBiasRouter):
    """V4.1 routing through TileKernels' TileLang top-k gate."""

    indices_dtype = torch.int64

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._hash_indices_table is not None:
            raise ValueError("DeepSeek V4.1 routing has no hash-routed layers")
        if self.scoring_func != "sqrtsoftplus" or not self.renormalize:
            raise ValueError("DeepSeek V4.1 routing is normalized sqrtsoftplus")
        if self.e_score_correction_bias is None:
            raise ValueError("DeepSeek V4.1 routing needs the correction bias")
        if self.num_fused_shared_experts or self.eplb_state is not None:
            raise ValueError(
                "TileKernels routing covers routed experts without EPLB only"
            )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
        sp_rows=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_bias = None
        vision = self.bias_vl is not None and self.image_sentinel_lo > 0
        if vision:
            if input_ids is None:
                raise ValueError("DeepSeek V4.1 vision routing requires input_ids")
            image_bias = self.bias_vl.data
        image_token_mask, mask = _routing_masks(
            router_logits.shape[0],
            input_ids if vision else None,
            self.image_sentinel_lo,
            self.image_sentinel_count,
            ced_decoder=getattr(self, "_ds41_is_ced_decoder", False),
            sp_rows=sp_rows,
        )
        topk_ids, topk_weights = _topk_gate(
            router_logits,
            self.top_k,
            self.routed_scaling_factor,
            mask=mask,
            bias=self.e_score_correction_bias.data,
            image_bias=image_bias,
            image_token_mask=image_token_mask,
        )
        # TileKernels writes int64 expert ids; the TileLang dispatch reads them.
        if indices_type not in (None, torch.int64):
            raise ValueError(
                f"TileKernels routing writes int64 expert ids, not {indices_type}"
            )
        return topk_weights, topk_ids


_WARP = 32
_VECTOR = 4


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def route_partials(
    num_topk: int,
    num_experts: int,
    shards: int,
    image_mask: bool,
    routing_mask: bool,
    warps: int = 2,
    image_ids: str | None = None,
    image_lo: int = 0,
    image_count: int = 0,
    padding: bool = False,
):
    """Sum a gate's split-K partials in shard order and route each token.

    With ``image_ids`` (their dtype), the image tokens come from the step's input
    ids, ``image_lo <= id < image_lo + image_count``, which torch compared once a
    forward; with ``padding``, ``route_mask`` is the step's padding mask (a row it
    sets is skipped), which torch negated once a forward.

    The routing is TileKernels' ``moe_topk_gate_forward`` for sqrtsoftplus
    with a correction bias: the same scores, selection order and weights. Its
    inputs are the partials ``P[shards, rows, experts]``, added from shard 0
    as ``splitk_reduce`` adds them. Each round's warp-wide winner takes two
    integer reductions instead of TileKernels' ten shuffles. Non-finite
    logits are refused once per lane, not once per logit. Rows the routing
    mask excludes get expert -1 and weight 0, as TileKernels gives them.
    """
    assert num_topk <= _WARP and num_experts % _VECTOR == 0
    per_lane = tilelang.cdiv(tilelang.cdiv(num_experts, _WARP), _VECTOR) * _VECTOR
    rows, image_len, mask_len = T.dynamic("rows, image_len, mask_len")
    image_dtype = image_ids if image_ids is not None else T.bool

    @T.prim_func
    def router_partials(
        P: T.Tensor((shards, rows, num_experts), T.float32),
        bias: T.Tensor((num_experts,), T.float32),
        image_bias: T.Tensor((num_experts,), T.float32),
        image_token_mask: T.Tensor((image_len,), image_dtype),
        route_mask: T.Tensor((mask_len,), T.bool),
        topk_idx: T.Tensor((rows, num_topk), T.int64),
        topk_weights: T.Tensor((rows, num_topk), T.float32),
        routed_scaling_factor: T.float32,
    ):
        with T.Kernel(T.ceildiv(rows, warps), threads=warps * _WARP) as pid:
            pdl.chain()
            thread = T.get_thread_binding()
            token = thread // _WARP
            row = token + pid * warps
            lane = thread % _WARP
            if row >= rows:
                T.thread_return()
            if routing_mask and route_mask[row] == padding:
                if lane < num_topk:
                    topk_idx[row, lane] = -1
                    topk_weights[row, lane] = 0.0
                T.thread_return()

            unbiased = T.alloc_shared((warps, num_experts), T.float32)
            bias_local = T.alloc_local((per_lane,), T.float32)
            scores = T.alloc_local((per_lane,), T.float32)
            ids = T.alloc_local((per_lane,), T.int32)
            finite = T.alloc_var(T.bool, init=True)
            best_score = T.alloc_var(T.float32)
            best_idx = T.alloc_var(T.int32, init=-1)
            previous = T.alloc_var(T.int32, init=-1)
            chosen_idx = T.alloc_var(T.int64)
            chosen_score = T.alloc_var(T.float32)
            total = T.alloc_var(T.float32)
            is_image = T.alloc_var(T.bool)
            if image_mask and image_ids is not None:
                is_image = (image_token_mask[row] >= image_lo) & (
                    image_token_mask[row] < image_lo + image_count
                )
            elif image_mask:
                is_image = image_token_mask[row]
            else:
                is_image = False

            T.fill(ids, -1)
            T.fill(scores, -T.infinity(T.float32))
            T.fill(bias_local, 0.0)
            for i in T.unroll(0, per_lane // _VECTOR):
                start = i * _VECTOR * _WARP + lane * _VECTOR
                if start < num_experts:
                    for j in T.vectorized(_VECTOR):
                        scores[i * _VECTOR + j] = P[0, row, start + j]
                        bias_local[i * _VECTOR + j] = (
                            image_bias[start + j] if is_image else bias[start + j]
                        )
                    for s in T.serial(1, shards):
                        for j in T.vectorized(_VECTOR):
                            scores[i * _VECTOR + j] += P[s, row, start + j]
                    for j in T.unroll(_VECTOR):
                        if not T.isfinite(scores[i * _VECTOR + j]):
                            finite = False
            T.device_assert(
                finite, msg="route_partials: gate logits contain nan or inf"
            )

            for i in T.unroll(0, T.ceildiv(num_experts, _VECTOR * _WARP)):
                start = i * _VECTOR * _WARP + lane * _VECTOR
                if start < num_experts:
                    for j in T.unroll(_VECTOR):
                        x = scores[i * _VECTOR + j]
                        scores[i * _VECTOR + j] = T.sqrt(
                            T.Select(x > 20.0, x, T.log1p(T.exp(x)))
                        )
                    for j in T.vectorized(_VECTOR):
                        unbiased[token, start + j] = scores[i * _VECTOR + j]
                        scores[i * _VECTOR + j] += bias_local[i * _VECTOR + j]
                        ids[i * _VECTOR + j] = start + j
            T.sync_warp()

            for k in T.unroll(num_topk):
                best_score = -T.infinity(T.float32)
                best_idx = -1
                for i in T.unroll(0, per_lane):
                    if k != 0 and previous == ids[i]:
                        scores[i] = -T.infinity(T.float32)
                    elif scores[i] > best_score:
                        best_score = scores[i]
                        best_idx = ids[i]
                # The winner across the warp in two hardware reductions: the
                # largest score as an integer that orders like the float, then
                # the lowest expert id among the lanes holding it.
                bits = T.reinterpret(best_score, T.int32)
                key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
                top = T.warp_reduce_max(key)
                best_idx = T.warp_reduce_min(
                    T.if_then_else(key == top, best_idx, 0x7FFFFFFF)
                )
                previous = best_idx
                if lane == k:
                    chosen_idx = best_idx

            chosen_score = 0.0
            if lane < num_topk:
                chosen_score = unbiased[token, chosen_idx]
            total = chosen_score
            for i in T.unroll(0, 5):
                total += T.shfl_xor(total, 1 << (4 - i))
            total += 1e-20
            if lane < num_topk:
                topk_idx[row, lane] = chosen_idx
                topk_weights[row, lane] = chosen_score / total * routed_scaling_factor

    return router_partials


class RoutedRows(NamedTuple):
    """Top-k routing weights and expert ids computed on local SP rows and
    gathered beside ``rows``' MXFP8 cast."""

    rows: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor


_PROVIDED_ROUTES: list[RoutedRows] = []


@contextlib.contextmanager
def provide_routes(
    rows: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
):
    """Hand ``rows``' gathered top-k routing to ``TileLangGateRouter`` for the
    MoE call inside."""
    _PROVIDED_ROUTES.append(RoutedRows(rows, weights, ids))
    try:
        yield
    finally:
        _PROVIDED_ROUTES.pop()


def provided_routes(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The handed-over ``(weights, ids)`` routing of ``x``, or None."""
    for routes in reversed(_PROVIDED_ROUTES):
        if routes.rows.data_ptr() == x.data_ptr() and routes.rows.shape == x.shape:
            return routes.weights, routes.ids
    return None


class TileLangGateRouter(TileKernelsRouter):
    """TileKernelsRouter that also runs the gate, fusing its decode reduction.

    The fused-MoE runner hands it the hidden states in place of router
    logits (the MoE layer gives it the gate instead of the runner).
    """

    routes_hidden_states = True

    def __init__(self, *args, gate: torch.nn.Module, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate
        self._kernels: dict[tuple, object] = {}
        # Text-only layers pass no image tokens; one all-false mask serves them.
        self._no_image = None
        vllm_config = get_current_vllm_config_or_none()
        # The most rows one forward routes (the drafter sets its query rows); None
        # (kernel tests) is unbounded.
        self.max_rows: int | None = (
            None
            if vllm_config is None
            else vllm_config.scheduler_config.max_num_batched_tokens
        )
        gate.tilelang_router = self
        self._mask_variants = _routing_mask_variants(vllm_config)

    def _partials_kernels(self) -> dict[tuple, prepare.TileLangCall]:
        """``route_partials`` for each (input ids' dtype, routing mask) a decode
        step can pass, keyed as ``_compute_routing`` looks it up; none when the
        gate does not split. A vision layer reads its image tokens from the input
        ids (the runner's int32, or int64); the mask is the step's padding mask,
        a CED decoder layer's routing mask, or none."""
        gate = self.gate
        if gate.tilelang_shards == 1:
            return {}
        image = self.bias_vl is not None and self.image_sentinel_lo > 0
        ids = ("int32", "int64") if image else (None,)
        masks = []
        for mask in self._mask_variants:
            if not mask:
                masks.append(None)
                continue
            masks.append("padding")
            if getattr(self, "_ds41_is_ced_decoder", False):
                masks.append("route")
        return {
            (ids_dtype, mask): prepare.call(
                route_partials,
                self.top_k,
                gate.weight.shape[0],
                gate.tilelang_shards,
                image,
                mask is not None,
                image_ids=ids_dtype,
                image_lo=self.image_sentinel_lo if image else 0,
                image_count=self.image_sentinel_count if image else 0,
                padding=mask == "padding",
            )
            for ids_dtype in ids
            for mask in masks
        }

    def _gates_every_row(self) -> bool:
        """Whether every forward's rows take TileKernels' gate path: a gate without
        split-K decode rows, or forwards that can have more rows than they."""
        gate = self.gate
        return not (
            gate.tilelang_shards > 1
            and self.max_rows is not None
            and self.max_rows <= gate.tilelang_split_rows
        )

    def _warm(self) -> None:
        """TileKernels' top-k gate for rows past the gate's split rows, if a forward
        can have them."""
        if not self._gates_every_row():
            return
        image = self.bias_vl is not None and self.image_sentinel_lo > 0
        for mask in self._mask_variants:
            _warm_topk_gate(self.top_k, self.gate.weight.shape[0], image, mask)

    def _mask_calls(self) -> list[prepare.TileLangCall]:
        """``routing_masks_kernel`` for every mask the gate path can compute (once a
        forward): a vision layer's image tokens from the runner's int32 or int64 ids,
        and the rows to route when a padding mask can come, alone or together."""
        if not self._gates_every_row():
            return []
        image = self.bias_vl is not None and self.image_sentinel_lo > 0
        ids = ("int32", "int64", None) if image else (None,)
        routes = (False, True) if any(self._mask_variants) else (False,)
        return [
            prepare.call(routing_masks_kernel, ids_dtype, route)
            for ids_dtype in ids
            for route in routes
            if ids_dtype is not None or route
        ]

    def declare_kernels(self) -> None:
        """Declare the router's kernels (its layer calls this when preparing): the
        decode rows' ``route_partials``, the prefill rows' TileKernels gate and its
        masks. They read the gate's split, set when the gate declares its own
        kernels."""

        def bind() -> None:
            for key, (function, args, kwargs) in self._partials_kernels().items():
                self._kernels[key] = function(*args, **kwargs)

        prepare.declare(
            lambda: [*self._partials_kernels().values(), *self._mask_calls()],
            bind=bind,
            warm=self._warm,
        )

    def route_local(
        self,
        x: torch.Tensor,
        sp_rows,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route this rank's ``sp_rows.local_rows`` rows before the SP MoE gather."""
        if x.shape[0] != sp_rows.local_rows:
            raise ValueError(
                f"SP local routing expects {sp_rows.local_rows} rows, got {x.shape}"
            )
        return self._compute_routing(
            x, x, torch.int64, input_ids=input_ids, sp_rows=sp_rows
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
        sp_rows=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .linear import _partial_specs, _scratch, partials_kernel

        gate = self.gate
        x = router_logits.reshape(-1, gate.weight.shape[1])
        provided = provided_routes(x)
        if provided is not None:
            if indices_type not in (None, torch.int64):
                raise ValueError(
                    f"TileLang routing writes int64 expert ids, not {indices_type}"
                )
            return provided
        specs = _partial_specs(gate, x.shape[0])
        if not specs:
            logits, _ = gate(x)
            return super()._compute_routing(
                hidden_states,
                logits,
                indices_type,
                input_ids=input_ids,
                sp_rows=sp_rows,
            )
        if indices_type not in (None, torch.int64):
            raise ValueError(
                f"TileLang routing writes int64 expert ids, not {indices_type}"
            )
        (partials,) = _scratch(specs, None)
        partials_kernel(gate, x.shape[0])(x, gate.weight, partials)
        image = self.bias_vl is not None and self.image_sentinel_lo > 0
        if image and input_ids is None:
            raise ValueError("DeepSeek V4.1 vision routing requires input_ids")
        ced_decoder = getattr(self, "_ds41_is_ced_decoder", False)
        route_mask, padding = _partials_mask(
            x.shape[0],
            ced_decoder=ced_decoder,
            sp_rows=sp_rows,
        )
        key = (
            {torch.int32: "int32", torch.int64: "int64"}[input_ids.dtype]
            if image
            else None,
            None if route_mask is None else "padding" if padding else "route",
        )
        rows = x.shape[0]
        topk_ids = torch.empty(rows, self.top_k, dtype=torch.int64, device=x.device)
        topk_weights = torch.empty(
            rows, self.top_k, dtype=torch.float32, device=x.device
        )
        bias = self.e_score_correction_bias.data
        if (
            self._no_image is None
            or self._no_image.device != x.device
            or self._no_image.numel() < rows
        ):
            # Every row count this fused path takes: the gate's split rows.
            self._no_image = torch.zeros(
                max(rows, gate.tilelang_split_rows),
                dtype=torch.bool,
                device=x.device,
            )
        # A vision layer's kernel compares the ids with the sentinels itself; a
        # text layer's, and an absent routing mask, read nothing of these rows.
        ids = _local_mask(input_ids, rows, sp_rows) if image else self._no_image[:rows]
        image_bias, mask = (self.bias_vl.data, ids) if image else (bias, ids)
        self._kernels[key](
            partials,
            bias,
            image_bias,
            mask,
            route_mask if route_mask is not None else self._no_image[:rows],
            topk_ids,
            topk_weights,
            self.routed_scaling_factor,
        )
        return topk_weights, topk_ids
