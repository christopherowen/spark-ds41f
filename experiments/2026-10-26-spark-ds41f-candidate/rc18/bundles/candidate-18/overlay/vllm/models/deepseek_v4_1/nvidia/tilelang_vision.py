# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated V4.1 ViT and aligner on DeepSeek-V4.1's TileLang kernels.

The vision tower, over the checkpoint's parameter tree, with every launch a
kernel of ``tilelang/vision.py``. One image is evaluated per call.
Each module declares its kernels (``declare_kernels``, which the worker calls after
loading); binding them provisions a workspace for the tower's capacity
(``_capacity``), so a forward allocates nothing but an output the caller did not
pass, and a capture with outputs passed allocates nothing at all.

A block runs eight launches: the fused q/k/v projection with its bias, the 2-D
rotation of q and k in place in it, attention over the image's rows (reading
q, k and v where the projection wrote them), the output projection with its bias,
the residual add with the second norm, the gate/up projection with the SwiGLU in
its epilogue, the down projection, and the residual add with the next block's (or
the final) norm. The aligner merges each window, projects with the bias and GELU
in one launch, and projects again. Images of at most ``GEMV_ROWS`` rows (merged
rows, for the aligner) project through the SIMT GEMVs, larger ones through the
tensor-core GEMMs.
"""

import itertools
import math

import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_transfer import copy_weight
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang import vision as kernels


def _tp_geometry(
    *,
    tp_size: int | None = None,
    tp_rank: int | None = None,
    use_data_parallel: bool = False,
) -> tuple[int, int]:
    """Return ``(tp_size, tp_rank)`` for the vision tower, defaulting to ``(1, 0)``
    when data-parallel mode is requested or the TP group is not initialized."""
    if use_data_parallel:
        return 1, 0
    if tp_size is not None and tp_rank is not None:
        if tp_size <= 0 or not (0 <= tp_rank < tp_size):
            raise ValueError(f"invalid TP geometry: rank {tp_rank} of {tp_size}")
        return int(tp_size), int(tp_rank)
    from vllm import distributed

    try:
        world = (
            int(tp_size)
            if tp_size is not None
            else int(distributed.get_tensor_model_parallel_world_size())
        )
        rank = (
            int(tp_rank)
            if tp_rank is not None
            else int(distributed.get_tensor_model_parallel_rank())
        )
    except AssertionError:
        return 1, 0
    return (world, rank) if world > 1 else (1, 0)


def _local_heads(config, tp_size: int) -> int:
    """Per-rank head count, padded so ``local_heads * head_dim`` is a multiple of
    the GEMM K-tile alignment (``_K_ALIGN = 64``)."""
    if tp_size <= 1:
        return int(config.vision_n_heads)
    head_dim = config.vision_dim // config.vision_n_heads
    heads_per_tile = max(1, kernels._K_ALIGN // math.gcd(head_dim, kernels._K_ALIGN))
    quantum = tp_size * heads_per_tile
    padded_heads = -(-config.vision_n_heads // quantum) * quantum
    return padded_heads // tp_size


def _local_width(full_width: int, tp_size: int) -> int:
    """Per-rank output width for a column-sharded projection, padded to a multiple
    of ``_K_ALIGN = 64`` so every rank's shard has whole GEMM N-tiles and 128-byte
    rows for ``tensor_model_parallel_all_gather``."""
    if tp_size <= 1:
        return int(full_width)
    quantum = tp_size * kernels._K_ALIGN
    return -(-full_width // quantum) * quantum // tp_size


def _gather_columns(local: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """All-gather column-sharded ``local [rows, local_width]`` across TP ranks and
    copy the unpadded ``dst.shape[1]`` columns into ``dst``."""
    from vllm import distributed

    gathered = distributed.tensor_model_parallel_all_gather(local, dim=-1)
    dst.copy_(gathered[:, : dst.shape[1]])
    return dst


def _capacity(config) -> int:
    """Rows of the largest image the tower serves."""
    capacity = getattr(config, "vision_max_tokens", None)
    if capacity is None:
        capacity = config.vision_max_n_token * config.vision_downsample_ratio**2
    if capacity <= 0:
        raise ValueError("vision token capacity must be positive")
    return int(capacity)


def _bf16_rows(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tensor.shape != shape or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be BF16 {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


class _Linear(nn.Module):
    """A checkpoint projection: BF16 ``weight [out, in]`` and ``bias``, optionally
    tensor-parallel sharded along ``shard_dim = 0`` (``parts`` stacked column
    shards, each contiguous in the checkpoint for ``O_DIRECT`` arena loading) with
    zero-padded tail rows when ``full_features`` does not divide evenly."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        shard_dim: int | None = None,
        parts: int = 1,
        full_features: int | None = None,
    ):
        super().__init__()
        if shard_dim not in (None, 0):
            raise ValueError(f"_Linear only supports shard_dim in (None, 0), got {shard_dim}")
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.shard_dim = shard_dim
        self.parts = int(parts)
        self.full_features = full_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self._load_shard})
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False
            )
            if bias
            else None
        )
        if self.bias is not None:
            set_weight_attrs(self.bias, {"weight_loader": self._load_shard})

    def _load_shard(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if self.shard_dim is None or (
            loaded_weight.shape == param.shape and self.tp_size == 1
        ):
            copy_weight(param.data, loaded_weight)
            return
        local_part = param.shape[0] // self.parts
        full_part = (
            self.full_features
            if self.full_features is not None
            else loaded_weight.shape[0] // self.parts
        )
        start = min(self.tp_rank * local_part, full_part)
        end = min((self.tp_rank + 1) * local_part, full_part)
        valid = end - start
        for p in range(self.parts):
            if valid > 0:
                copy_weight(
                    param.data[p * local_part : p * local_part + valid],
                    loaded_weight[p * full_part + start : p * full_part + end],
                )
            if valid < local_part and not param.is_meta:
                param.data[p * local_part + valid : (p + 1) * local_part].zero_()


class _Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.bfloat16), requires_grad=False
        )
        set_weight_attrs(self.weight, {"weight_loader": copy_weight})


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = _Linear(3 * config.vision_patch_size**2, config.vision_dim)


class _Attention(nn.Module):
    def __init__(self, config, *, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        head_dim = config.vision_dim // config.vision_n_heads
        local_heads = _local_heads(config, tp_size)
        qkv_width = local_heads * head_dim
        local_hidden = _local_width(config.vision_dim, tp_size)
        shard = 0 if tp_size > 1 else None
        self.wqkv = _Linear(
            config.vision_dim,
            3 * qkv_width,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=shard,
            parts=3,
            full_features=config.vision_dim,
        )
        self.wo = _Linear(
            config.vision_dim,
            local_hidden,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=shard,
            full_features=config.vision_dim,
        )


class _MLP(nn.Module):
    def __init__(self, config, *, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        local_hidden = _local_width(config.vision_dim, tp_size)
        local_inter = _local_width(config.vision_inter_dim, tp_size)
        shard = 0 if tp_size > 1 else None
        self.w1 = _Linear(
            config.vision_dim,
            2 * local_inter,
            False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=shard,
            parts=2,
            full_features=config.vision_inter_dim,
        )
        self.w2 = _Linear(
            config.vision_inter_dim,
            local_hidden,
            False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=shard,
            full_features=config.vision_dim,
        )


class _Block(nn.Module):
    def __init__(self, config, *, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.norm1, self.norm2 = _Norm(config.vision_dim), _Norm(config.vision_dim)
        self.attn = _Attention(config, tp_size=tp_size, tp_rank=tp_rank)
        self.mlp = _MLP(config, tp_size=tp_size, tp_rank=tp_rank)


class _VisionWorkspace:
    """The ViT's capacity-sized buffers. On TP1 the local shard buffers alias the
    full-width buffers directly; under TP sharding each column-sharded stage writes
    its local shard and all-gathers into the full-width buffer so every subsequent
    GEMM/GEMV accumulates over the unsharded K dimension."""

    def __init__(
        self,
        config,
        capacity: int,
        device: torch.device,
        *,
        local_heads: int | None = None,
        local_hidden: int | None = None,
        local_inter: int | None = None,
    ):
        hidden = config.vision_dim
        inter_full = config.vision_inter_dim
        dim = hidden // config.vision_n_heads
        qkv_width = (local_heads or config.vision_n_heads) * dim
        hidden_shard = local_hidden or hidden
        inter_shard = local_inter or inter_full
        patch = 3 * config.vision_patch_size**2
        padded = kernels.padded_width(patch)

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        self.patches = empty(capacity, padded)
        self.patch_weight = empty(hidden, padded)
        self.state, self.residual, self.normalized, self.branch = (
            empty(capacity, hidden) for _ in range(4)
        )
        self.qkv = empty(capacity, 3 * qkv_width)
        self.local_attended = (
            self.normalized if qkv_width == hidden else empty(capacity, qkv_width)
        )
        self.local_branch = (
            self.branch
            if hidden_shard == hidden
            else empty(capacity, hidden_shard)
        )
        self.activated = empty(capacity, inter_full)
        self.local_activated = (
            self.activated
            if inter_shard == inter_full
            else empty(capacity, inter_shard)
        )
        # B12X's static head geometry, computed the same way (on the CPU, in FP32).
        inv = 1.0 / (
            config.vision_rope_theta
            ** (
                torch.arange(0, dim // 2, 2, dtype=torch.float32, device="cpu")
                / (dim // 2)
            )
        )
        self.inv_freq = inv.to(device=device)


def _projections(launch: dict, rows: int) -> dict:
    """The projection kernels by role for ``rows`` rows: the GEMVs (``*_rows``) up
    to ``GEMV_ROWS``, else the GEMMs."""
    suffix = "_rows" if rows <= kernels.GEMV_ROWS else ""
    return {
        name: launch[name + suffix]
        for name in launch
        if not name.endswith("_rows") and name + "_rows" in launch
    }


def _device(module: nn.Module) -> torch.device:
    device = next(module.parameters()).device
    if device.type != "cuda":
        raise ValueError("the TileLang vision tower requires CUDA weights")
    return device


class DeepseekV4ViT(nn.Module):
    """Reference 2D split-half RoPE and full bidirectional attention per image,
    tensor-parallel sharded across TP ranks along output columns (with zero-padded
    tail heads/columns when widths do not divide evenly) so every rank accumulates
    the exact unsharded K dimension and produces bit-identical BF16 activations."""

    def __init__(
        self,
        config,
        *,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        if (
            config.vision_dim % config.vision_n_heads
            or (config.vision_dim // config.vision_n_heads) % 16
        ):
            raise ValueError("vision head dimension must be a multiple of 16")
        self.config, self.capacity = config, _capacity(config)
        self.tp_size, self.tp_rank = _tp_geometry(
            tp_size=tp_size, tp_rank=tp_rank, use_data_parallel=use_data_parallel
        )
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.local_heads = _local_heads(config, self.tp_size)
        self.qkv_width = self.local_heads * self.head_dim
        self.local_hidden = _local_width(config.vision_dim, self.tp_size)
        self.local_inter = _local_width(config.vision_inter_dim, self.tp_size)
        self.rope_dim = self.head_dim // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [
                _Block(config, tp_size=self.tp_size, tp_rank=self.tp_rank)
                for _ in range(config.vision_n_layers)
            ]
        )
        self.norm = _Norm(config.vision_dim)
        self._kernels: dict | None = None
        self._workspace: _VisionWorkspace | None = None

    def kernel_calls(self) -> dict[str, prepare.TileLangCall]:
        """Every kernel the forward launches, by role."""
        config = self.config
        hidden, inter_full = config.vision_dim, config.vision_inter_dim
        heads, dim = self.local_heads, self.head_dim
        qkv_width, local_hidden, inter = (
            self.qkv_width,
            self.local_hidden,
            self.local_inter,
        )
        patch = 3 * config.vision_patch_size**2
        return {
            "pad": kernels.pad_call(patch),
            "patch": kernels.linear_call(
                hidden, kernels.padded_width(patch), bias=True
            ),
            "norm": kernels.norm_call(hidden, add=False),
            "qkv": kernels.linear_call(3 * qkv_width, hidden, bias=True),
            "rope": kernels.rope_call(heads, dim),
            "attention": kernels.attention_call(heads, dim),
            "wo": kernels.linear_call(local_hidden, hidden, bias=True),
            "add_norm": kernels.norm_call(hidden, add=True),
            "mlp_in": kernels.swiglu_call(inter, hidden),
            "mlp_out": kernels.linear_call(local_hidden, inter_full, bias=False),
            # Images of at most GEMV_ROWS rows; the patches unpadded.
            "patch_rows": kernels.gemv_call(hidden, patch, "bias"),
            "qkv_rows": kernels.gemv_call(3 * qkv_width, hidden, "bias"),
            "wo_rows": kernels.gemv_call(local_hidden, hidden, "bias"),
            "mlp_in_rows": kernels.gemv_call(2 * inter, hidden, "swiglu"),
            "mlp_out_rows": kernels.gemv_call(local_hidden, inter_full, "none"),
        }

    def declare_kernels(self) -> None:
        calls = self.kernel_calls()
        prepare.declare(lambda: list(calls.values()), bind=lambda: self._bind(calls))

    def _bind(self, calls: dict) -> None:
        """Store the prepared kernels and provision the workspace (never during a
        CUDA graph capture)."""
        device = _device(self)
        self._kernels = {name: kernels.resolve(call) for name, call in calls.items()}
        if self._workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("vision workspace cannot allocate during capture")
            self._workspace = _VisionWorkspace(
                self.config,
                self.capacity,
                device,
                local_heads=self.local_heads,
                local_hidden=self.local_hidden,
                local_inter=self.local_inter,
            )
        elif self._workspace.state.device != device:
            raise ValueError("vision moved devices after preparation")

    def forward(self, patches, n_vit_h: int, n_vit_w: int, *, out=None):
        rows = n_vit_h * n_vit_w
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or rows != patches.shape[0]
            or rows > self.capacity
        ):
            raise ValueError(
                "vision image grid exceeds capacity or does not match patches"
            )
        # Patch rows are 588 values (1176 bytes): 8-byte pieces.
        if (
            patches.dtype != torch.bfloat16
            or not patches.is_contiguous()
            or patches.data_ptr() % 8
        ):
            raise ValueError("vision patches must be contiguous 8-byte aligned BF16")
        if self._kernels is None or self._workspace is None:
            raise RuntimeError("vision kernels were not prepared")
        hidden = self.config.vision_dim
        if out is None:
            out = torch.empty(
                (rows, hidden), dtype=torch.bfloat16, device=patches.device
            )
        _bf16_rows(out, (rows, hidden), "vision output")
        launch, ws = self._kernels, self._workspace
        state, residual = ws.state[:rows], ws.residual[:rows]
        normalized, branch = ws.normalized[:rows], ws.branch[:rows]
        qkv = ws.qkv[:rows]
        local_attended = ws.local_attended[:rows]
        local_branch = ws.local_branch[:rows]
        activated = ws.activated[:rows]
        local_activated = ws.local_activated[:rows]
        proj = self.patch_embed.proj
        project = _projections(launch, rows)

        if rows <= kernels.GEMV_ROWS:
            project["patch"](patches.reshape(rows, -1), proj.weight, proj.bias, state)
        else:
            launch["pad"](patches.reshape(rows, -1), ws.patches[:rows])
            launch["pad"](proj.weight, ws.patch_weight)
            project["patch"](ws.patches[:rows], ws.patch_weight, proj.bias, state)
        # Each block ends with the next block's first norm; the last with the final.
        norms = [block.norm1.weight for block in self.blocks] + [self.norm.weight]
        launch["norm"](state, norms[0], normalized if len(self.blocks) else out)
        sharded = self.tp_size > 1
        for index, block in enumerate(self.blocks):
            attn, mlp = block.attn, block.mlp
            project["qkv"](normalized, attn.wqkv.weight, attn.wqkv.bias, qkv)
            launch["rope"](qkv, ws.inv_freq, n_vit_w)
            launch["attention"](qkv, local_attended)
            attended = (
                _gather_columns(local_attended, normalized)
                if sharded
                else local_attended
            )
            project["wo"](attended, attn.wo.weight, attn.wo.bias, local_branch)
            if sharded:
                _gather_columns(local_branch, branch)
            launch["add_norm"](state, branch, block.norm2.weight, residual, normalized)
            project["mlp_in"](normalized, mlp.w1.weight, local_activated)
            if sharded:
                _gather_columns(local_activated, activated)
            project["mlp_out"](activated, mlp.w2.weight, local_branch)
            if sharded:
                _gather_columns(local_branch, branch)
            last = index + 1 == len(self.blocks)
            launch["add_norm"](
                residual, branch, norms[index + 1], state, out if last else normalized
            )
        return out


class _AlignerWorkspace:
    def __init__(
        self,
        hidden: int,
        out_dim: int,
        local_out_dim: int,
        ratio: int,
        rows: int,
        device,
    ):
        self.merged = torch.empty(
            (rows, hidden * ratio * ratio), dtype=torch.bfloat16, device=device
        )
        self.hidden = torch.empty(
            (rows, out_dim), dtype=torch.bfloat16, device=device
        )
        self.local_out = (
            self.hidden
            if local_out_dim == out_dim
            else torch.empty(
                (rows, local_out_dim), dtype=torch.bfloat16, device=device
            )
        )


class DeepseekV4Aligner(nn.Module):
    """Channel-major padded spatial merge, biased projections, exact GELU,
    tensor-parallel sharded across TP ranks along output columns."""

    def __init__(
        self,
        config,
        *,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        if self.downsample_ratio <= 0:
            raise ValueError("vision downsample ratio must be positive")
        self.tp_size, self.tp_rank = _tp_geometry(
            tp_size=tp_size, tp_rank=tp_rank, use_data_parallel=use_data_parallel
        )
        self.out_dim = config.hidden_size
        self.local_out_dim = _local_width(self.out_dim, self.tp_size)
        self.hidden_size, self.capacity = config.vision_dim, _capacity(config)
        # Merged rows of the widest grid the capacity admits (one row of patches).
        r = self.downsample_ratio
        self.merged_capacity = (self.capacity + r - 1) // r
        shard = 0 if self.tp_size > 1 else None
        self.w1 = _Linear(
            config.vision_dim * r**2,
            self.local_out_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            shard_dim=shard,
            full_features=self.out_dim,
        )
        self.w2 = _Linear(
            self.out_dim,
            self.local_out_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            shard_dim=shard,
            full_features=self.out_dim,
        )
        self._kernels: dict | None = None
        self._workspace: _AlignerWorkspace | None = None

    def kernel_calls(self) -> dict[str, prepare.TileLangCall]:
        r = self.downsample_ratio
        merged = self.hidden_size * r * r
        return {
            "merge": kernels.merge_call(self.hidden_size, r),
            "w1": kernels.linear_call(self.local_out_dim, merged, bias=True, gelu=True),
            "w2": kernels.linear_call(
                self.local_out_dim, self.out_dim, bias=True
            ),
            "w1_rows": kernels.gemv_call(self.local_out_dim, merged, "gelu"),
            "w2_rows": kernels.gemv_call(self.local_out_dim, self.out_dim, "bias"),
        }

    def declare_kernels(self) -> None:
        calls = self.kernel_calls()
        prepare.declare(lambda: list(calls.values()), bind=lambda: self._bind(calls))

    def _bind(self, calls: dict) -> None:
        device = _device(self)
        self._kernels = {name: kernels.resolve(call) for name, call in calls.items()}
        if self._workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("aligner workspace cannot allocate during capture")
            self._workspace = _AlignerWorkspace(
                self.hidden_size,
                self.out_dim,
                self.local_out_dim,
                self.downsample_ratio,
                self.merged_capacity,
                device,
            )
        elif self._workspace.merged.device != device:
            raise ValueError("aligner moved devices after preparation")

    def forward(self, x, n_vit_h: int, n_vit_w: int, *, out=None):
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or x.shape != (n_vit_h * n_vit_w, self.hidden_size)
            or x.shape[0] > self.capacity
        ):
            raise ValueError(
                "aligner image grid exceeds capacity or does not match input"
            )
        if self._kernels is None or self._workspace is None:
            raise RuntimeError("aligner kernels were not prepared")
        _bf16_rows(x, tuple(x.shape), "aligner input")
        r = self.downsample_ratio
        rows = ((n_vit_h + r - 1) // r) * ((n_vit_w + r - 1) // r)
        if out is None:
            out = torch.empty(
                (rows, self.out_dim), dtype=torch.bfloat16, device=x.device
            )
        _bf16_rows(out, (rows, self.out_dim), "aligner output")
        launch, ws = self._kernels, self._workspace
        project = _projections(launch, rows)
        merged, hidden = ws.merged[:rows], ws.hidden[:rows]
        local_out = ws.local_out[:rows]
        launch["merge"](x, merged, n_vit_h, n_vit_w)
        if self.tp_size > 1:
            project["w1"](merged, self.w1.weight, self.w1.bias, local_out)
            _gather_columns(local_out, hidden)
            project["w2"](hidden, self.w2.weight, self.w2.bias, local_out)
            _gather_columns(local_out, out)
        else:
            project["w1"](merged, self.w1.weight, self.w1.bias, hidden)
            project["w2"](hidden, self.w2.weight, self.w2.bias, out)
        return out


def run_dp_sharded_vision_tower(vision_model, aligner, patches, vit_grid):
    """Shard images across replicated TP ranks and perform a real all-gather."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.models.vision import get_load_balance_assignment

    if not vit_grid:
        if patches.shape[0] != 0:
            raise ValueError("nonempty patches require image grids")
        return []
    sizes = [h * w for h, w in vit_grid]
    if any(h <= 0 or w <= 0 for h, w in vit_grid) or sum(sizes) != patches.shape[0]:
        raise ValueError("image grids do not partition patches")
    world, rank = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
    )
    offsets = [0, *itertools.accumulate(sizes)]
    assignments, counts, _ = get_load_balance_assignment(sizes, world)
    rank_offsets = [0, *itertools.accumulate(counts)]
    r = aligner.downsample_ratio
    output_rows = [((h + r - 1) // r) * ((w + r - 1) // r) for h, w in vit_grid]

    def images(g):
        return assignments[rank_offsets[g] : rank_offsets[g + 1]]

    max_rows = max(sum(output_rows[i] for i in images(g)) for g in range(world))
    # Rows past a rank's images are gathered but never read: no fill.
    local = torch.empty(
        (max_rows, aligner.out_dim), dtype=patches.dtype, device=patches.device
    )
    offset = 0
    for i in images(rank):
        h, w = vit_grid[i]
        encoded = vision_model(patches[offsets[i] : offsets[i + 1]], h, w)
        aligner(encoded, h, w, out=local[offset : offset + output_rows[i]])
        offset += output_rows[i]
    gathered = tensor_model_parallel_all_gather(local, dim=0) if world > 1 else local
    result = [None] * len(vit_grid)
    for g in range(world):
        offset = g * max_rows
        for i in images(g):
            result[i] = gathered[offset : offset + output_rows[i]]
            offset += output_rows[i]
    return result


__all__ = ["DeepseekV4Aligner", "DeepseekV4ViT", "run_dp_sharded_vision_tower"]
