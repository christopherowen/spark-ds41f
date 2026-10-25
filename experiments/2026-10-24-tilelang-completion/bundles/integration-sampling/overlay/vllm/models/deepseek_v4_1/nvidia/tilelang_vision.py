# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated V4.1 ViT and aligner on DeepSeek-V4.1's TileLang kernels.

The TileLang family's (``VLLM_DS41_KERNEL_BACKEND=tilelang``) vision tower:
``b12x_vision``'s classes, forward signatures and checkpoint parameter tree, with
every launch a kernel of ``tilelang/vision.py``. One image is evaluated per call.
Each module declares its kernels (``declare_kernels``, which the worker calls after
loading); binding them provisions a workspace for the tower's capacity
(``_capacity``), so a forward allocates nothing but an output the caller did not
pass, and a capture with outputs passed allocates nothing at all.

A block runs seven launches: the fused q/k/v projection with its bias, the 2-D
rotation of q and k in place in it, attention over the image's rows (reading
q, k and v where the projection wrote them), the output projection with its bias,
the residual add with the second norm, the gate/up projection with the SwiGLU in
its epilogue, the down projection, and the residual add with the next block's (or
the final) norm. The aligner merges each window, projects with the bias and exact
GELU in one launch, and projects again.
"""

import itertools

import torch
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang import vision as kernels


def _capacity(config) -> int:
    """Rows of the largest image the tower serves (``b12x_vision``'s)."""
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
    """A replicated checkpoint projection: BF16 ``weight [out, in]`` and ``bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False
            )
            if bias
            else None
        )


class _Norm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.bfloat16), requires_grad=False
        )


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = _Linear(3 * config.vision_patch_size**2, config.vision_dim)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wqkv = _Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = _Linear(config.vision_dim, config.vision_dim)


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = _Linear(config.vision_dim, 2 * config.vision_inter_dim, False)
        self.w2 = _Linear(config.vision_inter_dim, config.vision_dim, False)


class _Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = _Norm(config.vision_dim), _Norm(config.vision_dim)
        self.attn, self.mlp = _Attention(config), _MLP(config)


class _VisionWorkspace:
    """The ViT's capacity-sized buffers. The attention output reuses ``normalized``
    (the q/k/v projection has read it by then), and ``qkv`` holds q, k and v."""

    def __init__(self, config, capacity: int, device: torch.device):
        hidden, inter = config.vision_dim, config.vision_inter_dim
        patch = 3 * config.vision_patch_size**2
        padded = kernels.padded_width(patch)
        dim = hidden // config.vision_n_heads

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        self.patches = empty(capacity, padded)
        self.patch_weight = empty(hidden, padded)
        self.state, self.residual, self.normalized, self.branch = (
            empty(capacity, hidden) for _ in range(4)
        )
        self.qkv = empty(capacity, 3 * hidden)
        self.activated = empty(capacity, inter)
        # B12X's static head geometry, computed the same way (on the CPU, in FP32).
        inv = 1.0 / (
            config.vision_rope_theta
            ** (
                torch.arange(0, dim // 2, 2, dtype=torch.float32, device="cpu")
                / (dim // 2)
            )
        )
        self.inv_freq = inv.to(device=device)


def _device(module: nn.Module) -> torch.device:
    device = next(module.parameters()).device
    if device.type != "cuda":
        raise ValueError("the TileLang vision tower requires CUDA weights")
    return device


class DeepseekV4ViT(nn.Module):
    """Reference 2D split-half RoPE and full bidirectional attention per image."""

    def __init__(self, config):
        super().__init__()
        if (
            config.vision_dim % config.vision_n_heads
            or (config.vision_dim // config.vision_n_heads) % 16
        ):
            raise ValueError("vision head dimension must be a multiple of 16")
        self.config, self.capacity = config, _capacity(config)
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [_Block(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = _Norm(config.vision_dim)
        self._kernels: dict | None = None
        self._workspace: _VisionWorkspace | None = None

    def kernel_calls(self) -> dict[str, prepare.TileLangCall]:
        """Every kernel the forward launches, by role."""
        config = self.config
        hidden, heads = config.vision_dim, config.vision_n_heads
        inter, dim = config.vision_inter_dim, hidden // heads
        patch = 3 * config.vision_patch_size**2
        return {
            "pad": kernels.pad_call(patch),
            "patch": kernels.linear_call(
                hidden, kernels.padded_width(patch), bias=True
            ),
            "norm": kernels.norm_call(hidden, add=False),
            "qkv": kernels.linear_call(3 * hidden, hidden, bias=True),
            "rope": kernels.rope_call(heads, dim),
            "attention": kernels.attention_call(heads, dim),
            "wo": kernels.linear_call(hidden, hidden, bias=True),
            "add_norm": kernels.norm_call(hidden, add=True),
            "mlp_in": kernels.swiglu_call(inter, hidden),
            "mlp_out": kernels.linear_call(hidden, inter, bias=False),
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
            self._workspace = _VisionWorkspace(self.config, self.capacity, device)
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
        qkv, activated = ws.qkv[:rows], ws.activated[:rows]
        proj = self.patch_embed.proj

        launch["pad"](patches.reshape(rows, -1), ws.patches[:rows])
        launch["pad"](proj.weight, ws.patch_weight)
        launch["patch"](ws.patches[:rows], ws.patch_weight, proj.bias, state)
        # Each block ends with the next block's first norm; the last with the final.
        norms = [block.norm1.weight for block in self.blocks] + [self.norm.weight]
        launch["norm"](state, norms[0], normalized if len(self.blocks) else out)
        for index, block in enumerate(self.blocks):
            attn, mlp = block.attn, block.mlp
            launch["qkv"](normalized, attn.wqkv.weight, attn.wqkv.bias, qkv)
            launch["rope"](qkv, ws.inv_freq, n_vit_w)
            launch["attention"](qkv, normalized)
            launch["wo"](normalized, attn.wo.weight, attn.wo.bias, branch)
            launch["add_norm"](state, branch, block.norm2.weight, residual, normalized)
            launch["mlp_in"](normalized, mlp.w1.weight, activated)
            launch["mlp_out"](activated, mlp.w2.weight, branch)
            last = index + 1 == len(self.blocks)
            launch["add_norm"](
                residual, branch, norms[index + 1], state, out if last else normalized
            )
        return out


class _AlignerWorkspace:
    def __init__(self, hidden: int, out_dim: int, ratio: int, rows: int, device):
        self.merged = torch.empty(
            (rows, hidden * ratio * ratio), dtype=torch.bfloat16, device=device
        )
        self.hidden = torch.empty((rows, out_dim), dtype=torch.bfloat16, device=device)


class DeepseekV4Aligner(nn.Module):
    """Channel-major padded spatial merge, biased projections, exact GELU."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        if self.downsample_ratio <= 0:
            raise ValueError("vision downsample ratio must be positive")
        self.out_dim = config.hidden_size
        self.hidden_size, self.capacity = config.vision_dim, _capacity(config)
        # Merged rows of the widest grid the capacity admits (one row of patches).
        r = self.downsample_ratio
        self.merged_capacity = (self.capacity + r - 1) // r
        self.w1 = _Linear(config.vision_dim * r**2, self.out_dim)
        self.w2 = _Linear(self.out_dim, self.out_dim)
        self._kernels: dict | None = None
        self._workspace: _AlignerWorkspace | None = None

    def kernel_calls(self) -> dict[str, prepare.TileLangCall]:
        r = self.downsample_ratio
        return {
            "merge": kernels.merge_call(self.hidden_size, r),
            "w1": kernels.linear_call(
                self.out_dim, self.hidden_size * r * r, bias=True, gelu=True
            ),
            "w2": kernels.linear_call(self.out_dim, self.out_dim, bias=True),
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
        merged, hidden = ws.merged[:rows], ws.hidden[:rows]
        launch["merge"](x, merged, n_vit_h, n_vit_w)
        launch["w1"](merged, self.w1.weight, self.w1.bias, hidden)
        launch["w2"](hidden, self.w2.weight, self.w2.bias, out)
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
