# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 ViT and aligner on DeepSeek-V4.1's TileLang kernels.

The vision tower, over the checkpoint's parameter tree, with every launch a
kernel of ``tilelang/vision.py``. Single images or packed multi-image batches
fitting within the tower's capacity (``_capacity``) are evaluated per call.
Each module declares its kernels (``declare_kernels``, which the worker calls after
loading); binding them provisions a workspace for the tower's capacity
(``_capacity``), so a forward allocates nothing but an output the caller did not
pass, and a capture with outputs passed allocates nothing at all.

Under tensor parallelism (``--mm-encoder-tp-mode weights``), ``attn.wqkv``,
``mlp.w1``, and ``aligner.w1`` are column-parallel (``shard_dim = 0``) while
``attn.wo``, ``mlp.w2``, and ``aligner.w2`` are row-parallel (``shard_dim = 1``),
eliminating intermediate all-gathers and keeping output widths unsharded
(multiples of 128 on every TP size). Partial row-parallel outputs are reduced
across TP ranks with ``rank_order_sum``.
"""

import itertools
import math

import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs
from spark_ds41.direct_io.weight_transfer import copy_weight
from spark_ds41.tilelang import prepare
from spark_ds41.tilelang import vision as kernels


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
    """Per-rank width for a sharded projection, padded to a multiple of
    ``_K_ALIGN = 64`` so every rank's shard has whole GEMM N/K tiles."""
    if tp_size <= 1:
        return int(full_width)
    quantum = tp_size * kernels._K_ALIGN
    return -(-full_width // quantum) * quantum // tp_size


def _valid_span(local_count: int, full_count: int, tp_rank: int) -> int:
    """Valid (unpadded) items on ``tp_rank`` for a dimension of size ``full_count``
    partitioned into ``local_count``-wide shards."""
    start = min(tp_rank * local_count, full_count)
    end = min((tp_rank + 1) * local_count, full_count)
    return max(0, end - start)


def _reduce_rows(local: torch.Tensor, dst: torch.Tensor, world: int) -> torch.Tensor:
    """All-gather row-parallel partial outputs ``local [rows, width]`` across TP
    ranks and sum them in rank order into ``dst``."""
    from vllm import distributed
    from spark_ds41.tilelang.collectives import rank_order_sum

    gathered = distributed.tensor_model_parallel_all_gather(local, dim=0)
    rank_order_sum(list(gathered.view(world, *dst.shape).unbind(0)), out=dst)
    return dst


def _capacity(config) -> int:
    """Rows of the largest image batch the tower serves."""
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
    tensor-parallel sharded along ``shard_dim = 0`` (``parts`` stacked output-row
    shards) or ``shard_dim = 1`` (input-column shard, with ``bias`` loaded on rank 0
    and zeroed on nonzero ranks so cross-rank reduction adds the bias once)."""

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
        if shard_dim not in (None, 0, 1):
            raise ValueError(
                f"_Linear only supports shard_dim in (None, 0, 1), got {shard_dim}"
            )
        if shard_dim == 1 and parts != 1:
            raise ValueError("_Linear row-parallel shard_dim=1 requires parts=1")
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
        if self.shard_dim == 1:
            if param.ndim == 1:
                if self.tp_rank == 0:
                    copy_weight(param.data, loaded_weight)
                elif not param.is_meta:
                    param.data.zero_()
                return
            local_in = param.shape[1]
            full_in = (
                self.full_features
                if self.full_features is not None
                else loaded_weight.shape[1]
            )
            start = min(self.tp_rank * local_in, full_in)
            end = min((self.tp_rank + 1) * local_in, full_in)
            valid = end - start
            if valid > 0:
                copy_weight(
                    param.data[:, :valid],
                    loaded_weight[:, start:end],
                )
            if valid < local_in and not param.is_meta:
                param.data[:, valid:local_in].zero_()
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
        self.wqkv = _Linear(
            config.vision_dim,
            3 * qkv_width,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=0 if tp_size > 1 else None,
            parts=3,
            full_features=config.vision_dim,
        )
        self.wo = _Linear(
            qkv_width,
            config.vision_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=1 if tp_size > 1 else None,
            full_features=config.vision_dim,
        )


class _MLP(nn.Module):
    def __init__(self, config, *, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        local_inter = _local_width(config.vision_inter_dim, tp_size)
        self.w1 = _Linear(
            config.vision_dim,
            2 * local_inter,
            False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=0 if tp_size > 1 else None,
            parts=2,
            full_features=config.vision_inter_dim,
        )
        self.w2 = _Linear(
            local_inter,
            config.vision_dim,
            False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            shard_dim=1 if tp_size > 1 else None,
            full_features=config.vision_inter_dim,
        )


class _Block(nn.Module):
    def __init__(self, config, *, tp_size: int = 1, tp_rank: int = 0):
        super().__init__()
        self.norm1, self.norm2 = _Norm(config.vision_dim), _Norm(config.vision_dim)
        self.attn = _Attention(config, tp_size=tp_size, tp_rank=tp_rank)
        self.mlp = _MLP(config, tp_size=tp_size, tp_rank=tp_rank)


class _VisionWorkspace:
    """The ViT's capacity-sized buffers. On TP1 ``local_branch`` aliases ``branch``
    directly; under TP sharding each column-sharded stage (``wqkv``/``attention``,
    ``mlp.w1``) feeds directly into its paired row-parallel projection (``wo``,
    ``mlp.w2``), which writes its partial output into ``local_branch`` and reduces
    across ranks into ``branch``."""

    def __init__(
        self,
        config,
        capacity: int,
        device: torch.device,
        *,
        local_heads: int | None = None,
        local_inter: int | None = None,
        sharded: bool = False,
    ):
        hidden = config.vision_dim
        inter_full = config.vision_inter_dim
        dim = hidden // config.vision_n_heads
        qkv_width = (local_heads or config.vision_n_heads) * dim
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
        self.local_branch = empty(capacity, hidden) if sharded else self.branch
        self.local_activated = empty(capacity, inter_shard)
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
    tensor-parallel sharded with column-parallel ``wqkv``/``mlp.w1`` and
    row-parallel ``wo``/``mlp.w2`` reduced via ``rank_order_sum``."""

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
        self.valid_heads = _valid_span(
            self.local_heads, config.vision_n_heads, self.tp_rank
        )
        self.qkv_width = self.local_heads * self.head_dim
        self.local_hidden = config.vision_dim
        self.local_inter = _local_width(config.vision_inter_dim, self.tp_size)
        self.valid_inter = _valid_span(
            self.local_inter, config.vision_inter_dim, self.tp_rank
        )
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
        hidden = config.vision_dim
        heads, dim = self.local_heads, self.head_dim
        qkv_width, inter = self.qkv_width, self.local_inter
        patch = 3 * config.vision_patch_size**2
        calls = {
            "pad": kernels.pad_call(patch),
            "patch": kernels.linear_call(
                hidden, kernels.padded_width(patch), bias=True
            ),
            "norm": kernels.norm_call(hidden, add=False),
            "qkv": kernels.linear_call(3 * qkv_width, hidden, bias=True),
            "rope": kernels.rope_call(heads, dim),
            "attention": kernels.attention_call(heads, dim),
            "wo": kernels.linear_call(hidden, qkv_width, bias=True),
            "add_norm": kernels.norm_call(hidden, add=True),
            "mlp_in": kernels.swiglu_call(inter, hidden),
            "mlp_out": kernels.linear_call(hidden, inter, bias=False),
            # Images of at most GEMV_ROWS rows; the patches unpadded.
            "patch_rows": kernels.gemv_call(hidden, patch, "bias"),
            "qkv_rows": kernels.gemv_call(3 * qkv_width, hidden, "bias"),
            "wo_rows": kernels.gemv_call(hidden, qkv_width, "bias"),
            "mlp_in_rows": kernels.gemv_call(2 * inter, hidden, "swiglu"),
            "mlp_out_rows": kernels.gemv_call(hidden, inter, "none"),
        }

        if self.tp_size > 1:
            from spark_ds41.tilelang.collectives import (
                rank_order_sum_kernel,
            )

            calls["rank_order_sum"] = prepare.call(rank_order_sum_kernel, self.tp_size)
        return calls

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
                local_inter=self.local_inter,
                sharded=self.tp_size > 1,
            )
        elif self._workspace.state.device != device:
            raise ValueError("vision moved devices after preparation")

    def forward_batch(
        self,
        patches: torch.Tensor,
        vit_grid: list[list[int]] | list[tuple[int, int]],
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a packed batch of images whose total patch count fits within
        ``self.capacity``. Token-independent projections, norms, activations, and
        collectives execute once across all packed rows; 2D RoPE and self-attention
        execute per image slice."""
        grid = [(int(h), int(w)) for h, w in vit_grid]
        sizes = [h * w for h, w in grid]
        rows = sum(sizes)
        if (
            not grid
            or any(h <= 0 or w <= 0 for h, w in grid)
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
        spans: list[tuple[int, int, int]] = []
        offset = 0
        for (_, w), size in zip(grid, sizes):
            spans.append((offset, offset + size, w))
            offset += size

        for index, block in enumerate(self.blocks):
            attn, mlp = block.attn, block.mlp
            if self.valid_heads > 0:
                project["qkv"](normalized, attn.wqkv.weight, attn.wqkv.bias, qkv)
                for start, end, width in spans:
                    qkv_slice = qkv[start:end]
                    launch["rope"](qkv_slice, ws.inv_freq, width)
                    launch["attention"](qkv_slice, local_attended[start:end])
                project["wo"](
                    local_attended, attn.wo.weight, attn.wo.bias, local_branch
                )
            else:
                local_branch.zero_()
            if sharded:
                _reduce_rows(local_branch, branch, self.tp_size)
            launch["add_norm"](state, branch, block.norm2.weight, residual, normalized)
            if self.valid_inter > 0:
                project["mlp_in"](normalized, mlp.w1.weight, local_activated)
                project["mlp_out"](local_activated, mlp.w2.weight, local_branch)
            else:
                local_branch.zero_()
            if sharded:
                _reduce_rows(local_branch, branch, self.tp_size)
            last = index + 1 == len(self.blocks)
            launch["add_norm"](
                residual, branch, norms[index + 1], state, out if last else normalized
            )
        return out

    def forward(self, patches, n_vit_h: int, n_vit_w: int, *, out=None):
        return self.forward_batch(patches, [(n_vit_h, n_vit_w)], out=out)


class _AlignerWorkspace:
    def __init__(
        self,
        hidden: int,
        out_dim: int,
        local_out_dim: int,
        ratio: int,
        rows: int,
        device,
        *,
        sharded: bool = False,
    ):
        self.merged = torch.empty(
            (rows, hidden * ratio * ratio), dtype=torch.bfloat16, device=device
        )
        self.hidden = torch.empty(
            (rows, local_out_dim), dtype=torch.bfloat16, device=device
        )
        self.gelu = kernels.gelu_table(device)
        self.local_out = (
            torch.empty((rows, out_dim), dtype=torch.bfloat16, device=device)
            if sharded
            else None
        )


class DeepseekV4Aligner(nn.Module):
    """Channel-major padded spatial merge, biased projections, exact GELU,
    tensor-parallel sharded with column-parallel ``w1`` and row-parallel ``w2``
    reduced via ``rank_order_sum``."""

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
        self.valid_out_dim = _valid_span(
            self.local_out_dim, self.out_dim, self.tp_rank
        )
        self.hidden_size, self.capacity = config.vision_dim, _capacity(config)
        # Merged rows of the widest grid the capacity admits (one row of patches).
        r = self.downsample_ratio
        self.merged_capacity = (self.capacity + r - 1) // r
        self.w1 = _Linear(
            config.vision_dim * r**2,
            self.local_out_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            shard_dim=0 if self.tp_size > 1 else None,
            full_features=self.out_dim,
        )
        self.w2 = _Linear(
            self.local_out_dim,
            self.out_dim,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            shard_dim=1 if self.tp_size > 1 else None,
            full_features=self.out_dim,
        )
        self._kernels: dict | None = None
        self._workspace: _AlignerWorkspace | None = None

    def kernel_calls(self) -> dict[str, prepare.TileLangCall]:
        r = self.downsample_ratio
        merged = self.hidden_size * r * r
        calls = {
            "merge": kernels.merge_call(self.hidden_size, r),
            "w1": kernels.linear_call(
                self.local_out_dim, merged, bias=True, gelu=True
            ),
            "w2": kernels.linear_call(self.out_dim, self.local_out_dim, bias=True),
            "w1_rows": kernels.gemv_call(self.local_out_dim, merged, "gelu"),
            "w2_rows": kernels.gemv_call(self.out_dim, self.local_out_dim, "bias"),
        }

        if self.tp_size > 1:
            from spark_ds41.tilelang.collectives import (
                rank_order_sum_kernel,
            )

            calls["rank_order_sum"] = prepare.call(rank_order_sum_kernel, self.tp_size)
        return calls

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
                sharded=self.tp_size > 1,
            )
        elif self._workspace.merged.device != device:
            raise ValueError("aligner moved devices after preparation")

    def forward_batch(
        self,
        x: torch.Tensor,
        vit_grid: list[list[int]] | list[tuple[int, int]],
        *,
        out: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Spatial-merge each image in ``vit_grid`` into packed rows, then run
        ``w1``, ``w2``, and TP reduction once across all merged rows."""
        grid = [(int(h), int(w)) for h, w in vit_grid]
        vit_sizes = [h * w for h, w in grid]
        total_vit = sum(vit_sizes)
        r = self.downsample_ratio
        merged_sizes = [((h + r - 1) // r) * ((w + r - 1) // r) for h, w in grid]
        rows = sum(merged_sizes)
        if (
            not grid
            or any(h <= 0 or w <= 0 for h, w in grid)
            or x.shape != (total_vit, self.hidden_size)
            or total_vit > self.capacity
            or rows > self.merged_capacity
        ):
            raise ValueError(
                "aligner image grid exceeds capacity or does not match input"
            )
        if self._kernels is None or self._workspace is None:
            raise RuntimeError("aligner kernels were not prepared")
        _bf16_rows(x, tuple(x.shape), "aligner input")
        if out is None:
            out = torch.empty(
                (rows, self.out_dim), dtype=torch.bfloat16, device=x.device
            )
        _bf16_rows(out, (rows, self.out_dim), "aligner output")
        launch, ws = self._kernels, self._workspace
        project = _projections(launch, rows)
        merged, hidden = ws.merged[:rows], ws.hidden[:rows]
        vit_offset = 0
        merged_offset = 0
        for (h, w), vit_rows, m_rows in zip(grid, vit_sizes, merged_sizes):
            launch["merge"](
                x[vit_offset : vit_offset + vit_rows],
                merged[merged_offset : merged_offset + m_rows],
                h,
                w,
            )
            vit_offset += vit_rows
            merged_offset += m_rows
        if self.tp_size > 1:
            assert ws.local_out is not None
            local_out = ws.local_out[:rows]
            if self.valid_out_dim > 0:
                project["w1"](merged, self.w1.weight, self.w1.bias, ws.gelu, hidden)
                project["w2"](hidden, self.w2.weight, self.w2.bias, local_out)
            else:
                local_out.zero_()
            _reduce_rows(local_out, out, self.tp_size)
        else:
            project["w1"](merged, self.w1.weight, self.w1.bias, ws.gelu, hidden)
            project["w2"](hidden, self.w2.weight, self.w2.bias, out)
        slices: list[torch.Tensor] = []
        offset = 0
        for m_rows in merged_sizes:
            slices.append(out[offset : offset + m_rows])
            offset += m_rows
        return slices

    def forward(self, x, n_vit_h: int, n_vit_w: int, *, out=None):
        return self.forward_batch(x, [(n_vit_h, n_vit_w)], out=out)[0]


def encode_images(
    vision_model: DeepseekV4ViT,
    aligner: DeepseekV4Aligner,
    patches: torch.Tensor,
    vit_grid: list[list[int]] | list[tuple[int, int]],
    *,
    out: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Encode one or more images by packing contiguous runs of images that fit
    within the ViT and aligner workspace capacities into batched passes."""
    if not vit_grid:
        if patches.shape[0] != 0:
            raise ValueError("nonempty patches require image grids")
        return []
    grid = [(int(h), int(w)) for h, w in vit_grid]
    sizes = [h * w for h, w in grid]
    if any(h <= 0 or w <= 0 for h, w in grid) or sum(sizes) != patches.shape[0]:
        raise ValueError("image grids do not partition patches")
    r = aligner.downsample_ratio
    merged_sizes = [((h + r - 1) // r) * ((w + r - 1) // r) for h, w in grid]
    vit_cap = vision_model.capacity
    merged_cap = aligner.merged_capacity

    results: list[torch.Tensor] = []
    n = len(grid)
    idx = 0
    patch_offset = 0
    out_offset = 0
    while idx < n:
        chunk_grid: list[tuple[int, int]] = []
        chunk_patches = 0
        chunk_merged = 0
        while idx < n:
            p_rows = sizes[idx]
            m_rows = merged_sizes[idx]
            if p_rows > vit_cap or m_rows > merged_cap:
                raise ValueError(
                    "vision image grid exceeds capacity or does not match patches"
                )
            if chunk_grid and (
                chunk_patches + p_rows > vit_cap or chunk_merged + m_rows > merged_cap
            ):
                break
            chunk_grid.append(grid[idx])
            chunk_patches += p_rows
            chunk_merged += m_rows
            idx += 1
        encoded = vision_model.forward_batch(
            patches[patch_offset : patch_offset + chunk_patches],
            chunk_grid,
        )
        chunk_out = (
            out[out_offset : out_offset + chunk_merged] if out is not None else None
        )
        results.extend(aligner.forward_batch(encoded, chunk_grid, out=chunk_out))
        patch_offset += chunk_patches
        out_offset += chunk_merged
    return results


def run_dp_sharded_vision_tower(vision_model, aligner, patches, vit_grid):
    """Shard images across replicated TP ranks, pack each rank's assigned images
    into batched tower passes, and perform a real all-gather."""
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
    assigned = images(rank)
    if assigned:
        rank_patches = (
            patches[offsets[assigned[0]] : offsets[assigned[0] + 1]]
            if len(assigned) == 1
            else torch.cat([patches[offsets[i] : offsets[i + 1]] for i in assigned])
        )
        rank_grid = [vit_grid[i] for i in assigned]
        rank_out_rows = sum(output_rows[i] for i in assigned)
        encode_images(
            vision_model,
            aligner,
            rank_patches,
            rank_grid,
            out=local[:rank_out_rows],
        )
    gathered = tensor_model_parallel_all_gather(local, dim=0) if world > 1 else local
    result = [None] * len(vit_grid)
    for g in range(world):
        offset = g * max_rows
        for i in images(g):
            result[i] = gathered[offset : offset + output_rows[i]]
            offset += output_rows[i]
    return result


__all__ = [
    "DeepseekV4Aligner",
    "DeepseekV4ViT",
    "encode_images",
    "run_dp_sharded_vision_tower",
]
