# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's TileLang vision tower against its arithmetic and DeepSeek's
reference.

Bit for bit (compared as bits): the patch padding and the spatial merge (copies),
the residual add with RMSNorm, the 2-D RoPE and the bias epilogue's single
rounding, against torch models of the tower's arithmetic (``_b12x_*``, read from
the CuTe kernels the port replaced: the same reduction tree, FMAs where libNVVM
contracts, the same FP32 constants); the GELU and SwiGLU epilogues against
DeepSeek's reference activations (torch's ``F.gelu``), on every BF16 input.

Against FP64 references (DeepSeek's ``deepseek_v4/common/vision.py`` math in FP64,
rounded to BF16 where its modules return BF16), with tolerances no looser than fixed
ones nor than B12X's own error on the same inputs (``B12X_ERROR``, measured on the
tower this one replaced): the projections, attention, and the whole tower and aligner
at grids from 1x1 to the capacity, odd sizes included for the merge padding. Then
non-finite values (the reference's positions), stale capacity rows, CUDA graph replay,
the data-parallel runner and declaration coverage.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_finite_equal,
    assert_other_rows_equal,
    assert_same_nonfinite,
)
from vllm.models.deepseek_v4_1.nvidia import tilelang_vision  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import prepare, vision  # noqa: E402

DEVICE = "cuda"


def _config(**overrides) -> SimpleNamespace:
    base = dict(
        vision_patch_size=2,
        vision_dim=128,
        vision_n_heads=2,
        vision_inter_dim=192,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=256,
        vision_max_n_token=16,
    )
    return SimpleNamespace(**{**base, **overrides})


# Two blocks of narrow widths, capacity 144 rows.
SMALL = _config()
# DeepSeek-V4.1's widths (1024 wide, 16 heads of 64, 2816, 14-pixel patches, a
# 5120-wide aligner), one block and a 576-row capacity.
REAL = _config(
    vision_patch_size=14,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_n_layers=1,
    hidden_size=5120,
    vision_max_n_token=64,
)
# Grids from one patch to the capacity; odd sizes pad the 3x3 merge.
SMALL_GRIDS = [(1, 1), (1, 7), (4, 7), (5, 5), (7, 11), (12, 12), (3, 48), (1, 144)]
REAL_GRIDS = [(1, 1), (5, 7), (13, 17), (24, 24), (1, 576)]

# B12X's error at each FP64 comparison below, on the same inputs: candidate 6's vision
# test on its image (spark-ds41f experiments/2026-10-25-retire-conditionals,
# bundles/vision-b12x-error; dgx4, 2026-10-25), which measured both towers. Projections:
# (worst BF16 steps, values off the correctly rounded); attention: (relative RMS, worst
# BF16 steps); towers, by width and grid: (ViT, aligner) relative RMS.
B12X_ERROR = {
    "qkv at 1 rows": (0.49664192447845906, 0),
    "wo at 1 rows": (0.49470068153396846, 0),
    "mlp gate/up at 1 rows": (0.49794636391807, 0),
    "mlp down at 1 rows": (0.4959321068274381, 0),
    "aligner w1 at 1 rows": (0.48621282933571575, 0),
    "aligner w2 at 1 rows": (0.4934345848745577, 2),
    "qkv at 77 rows": (0.49866283029179753, 130),
    "wo at 77 rows": (0.49879117582709537, 37),
    "mlp gate/up at 77 rows": (0.49880283280869087, 198),
    "mlp down at 77 rows": (0.49676895476923505, 63),
    "aligner w1 at 77 rows": (0.49089014453706376, 2006),
    "aligner w2 at 77 rows": (0.4945250258583259, 1170),
    "qkv at 192 rows": (0.498748932429561, 333),
    "wo at 192 rows": (121.80086993074598, 80684),
    "mlp gate/up at 192 rows": (0.4989614576498799, 572),
    "mlp down at 192 rows": (0.49701668764484325, 291),
    "aligner w1 at 192 rows": (0.4913324818187672, 5229),
    "aligner w2 at 192 rows": (0.49501480496989436, 2878),
    "patch embedding at 1 rows": (0.49650757680379937, 0),
    "patch embedding at 77 rows": (0.49868921946481376, 29),
    "patch embedding at 300 rows": (0.4992916210657623, 92),
    "attention at 1 rows": (0.0, 0.0),
    "attention at 77 rows": (0.001767753885921865, 10.741378149878534),
    "attention at 300 rows": (0.0017426768323724874, 14.752147869508457),
    "attention at 576 rows": (0.0017363360945632515, 12.480541974387345),
    "tower 128 1x1": (0.0, 0.0),
    "tower 128 1x7": (0.006624815042978595, 0.0),
    "tower 128 4x7": (0.005986462518016249, 0.0),
    "tower 128 5x5": (0.006280779640718641, 0.0),
    "tower 128 7x11": (0.006553608157130729, 0.004384625869705573),
    "tower 128 12x12": (0.0063686162131065495, 0.004168449837265123),
    "tower 128 3x48": (0.006261522709414092, 0.00446008286592595),
    "tower 128 1x144": (0.006070036114104239, 0.0002476796899654311),
    "tower 1024 1x1": (0.0, 0.0),
    "tower 1024 5x7": (0.004482965789398322, 0.00067090285939057),
    "tower 1024 13x17": (0.003496343115106573, 0.0050644118823980925),
    "tower 1024 24x24": (0.0030264071165440115, 0.0009215817182229654),
    "tower 1024 1x576": (0.0030055770167633227, 0.0007788215984908864),
}


# --- Construction ---------------------------------------------------------------


@torch.no_grad()
def _init(module: torch.nn.Module, seed: int) -> None:
    """Random weights at serving scale: unit-variance projections, small biases,
    norm weights near one."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    for name, parameter in module.named_parameters():
        noise = torch.randn(parameter.shape, generator=gen, device=DEVICE)
        if "norm" in name:
            parameter.copy_(1 + 0.1 * noise)
        elif name.endswith("bias"):
            parameter.copy_(0.1 * noise)
        else:
            parameter.copy_(noise / math.sqrt(parameter.shape[1]))


def _build(config, seed: int = 411):
    with torch.device(DEVICE):
        vit = tilelang_vision.DeepseekV4ViT(config)
        aligner = tilelang_vision.DeepseekV4Aligner(config)
    _init(vit, seed)
    _init(aligner, seed + 1)
    return vit, aligner


def _towers(config, seed: int = 411):
    vit, aligner = _build(config, seed)
    vit.declare_kernels()
    aligner.declare_kernels()
    return vit, aligner


@pytest.fixture(scope="module")
def small():
    return _towers(SMALL)


@pytest.fixture(scope="module")
def real():
    return _towers(REAL)


def _kernel(module, role: str):
    return vision.resolve(module.kernel_calls()[role])


def _randn(*shape, seed: int, scale: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return (scale * torch.randn(shape, generator=gen, device=DEVICE)).bfloat16()


def _patches(config, rows: int, seed: int) -> torch.Tensor:
    p = config.vision_patch_size
    return _randn(rows, 3, p, p, seed=seed)


def _norm_weight(width: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return (1 + 0.1 * torch.randn(width, generator=gen, device=DEVICE)).bfloat16()


# --- Bits and errors --------------------------------------------------------------


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16)


def _differing(got: torch.Tensor, want: torch.Tensor) -> str | None:
    diff = _bits(got) != _bits(want)
    if not bool(diff.any()):
        return None
    return (
        f"{int(diff.sum())} of {diff.numel()} values differ, first at "
        f"{diff.nonzero()[:4].tolist()}"
    )


def _assert_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape, (what, got.shape, want.shape)
    message = _differing(got, want)
    assert message is None, f"{what}: {message}"


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The spacing of BF16 values at ``|x|`` (FP64), normals' at the least."""
    exponent = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.exp2(exponent - 7)


def _fma32(a, b, c) -> torch.Tensor:
    """Correctly rounded FP32 ``a * b + c`` of FP32 tensors (one rounding, an FMA).

    The product is exact in FP64; the sum is rounded to odd in FP64 (Knuth's
    two-sum gives the first rounding's error), and FP64 rounded to odd and then to
    FP32 is FP32 correctly rounded (53 >= 24 + 2 bits)."""
    a, b, c = torch.broadcast_tensors(a.double(), b.double(), c.double())
    product = a * b
    total = product + c
    a1 = total - c
    b1 = total - a1
    error = (product - a1) + (c - b1)
    bits = total.contiguous().view(torch.int64)
    inexact = (error != 0) & torch.isfinite(total)
    toward_zero = inexact & ((error > 0) != (total > 0))
    bits = torch.where(toward_zero, bits - 1, bits)
    bits = torch.where(inexact, bits | 1, bits)
    return bits.view(torch.float64).float()


def _round(x: torch.Tensor) -> torch.Tensor:
    """FP64 values rounded to BF16 (where the reference's modules return BF16)."""
    return x.bfloat16().double()


def _relative(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.double() - ref).norm() / ref.norm()).item()


def _max_ulps(out: torch.Tensor, ref: torch.Tensor, floor) -> float:
    return ((out.double() - ref).abs() / (_bf16_ulp(ref) + floor)).max().item()


# --- B12X's arithmetic, as torch ----------------------------------------------------


def _b12x_rmsnorm(x: torch.Tensor, weight: torch.Tensor, fused: bool = True):
    """B12X's grouped RMSNorm (one stream, not zero-centered, eps 1e-6) in its
    order: thread ``t`` of 256 adds the squares of columns ``t, t + 256, ...``
    (``fused``: by FMA, as libNVVM contracts ``square_sum += value * value``), a
    butterfly over each warp's lanes, the eight warp sums in one warp's butterfly,
    ``rsqrt(sum / width + eps)`` (``rsqrtf``: torch's CUDA rsqrt), then
    ``(x * inverse) * (0 + weight)``."""
    rows, width = x.shape
    threads = 256
    items = -(-width // threads)
    values = torch.zeros(rows, items * threads, device=x.device)
    values[:, :width] = x.float()
    total = torch.zeros(rows, threads, device=x.device)
    for item in range(items):
        v = values[:, item * threads : (item + 1) * threads]
        total = _fma32(v, v, total) if fused else total + v * v
    lanes = torch.arange(32, device=x.device)
    total = total.view(rows, threads // 32, 32)
    for level in range(5):
        total = total + total[..., lanes ^ (1 << level)]
    warps = torch.zeros(rows, 32, device=x.device)
    warps[:, : threads // 32] = total[..., 0]
    for level in range(5):
        warps = warps + warps[:, lanes ^ (1 << level)]
    inverse = torch.rsqrt(warps[:, :1] / width + 1e-6)
    return ((values[:, :width] * inverse) * (weight.float() + 0.0)).bfloat16()


def _inv_freq(config) -> torch.Tensor:
    """B12X's FP32 inverse frequencies (computed on the CPU, as its workspace)."""
    dim = config.vision_dim // config.vision_n_heads
    inv = 1.0 / (
        config.vision_rope_theta
        ** (torch.arange(0, dim // 2, 2, dtype=torch.float32) / (dim // 2))
    )
    return inv.to(DEVICE)


def _b12x_rope(qkv, height, width, inv_freq, heads, form: str = "left"):
    """B12X's 2-D split-half RoPE of q and k in ``qkv``: pair ``j`` of a head rotates
    by ``float(position) * inv_freq[j % (dim / 4)]`` (the row for ``j < dim / 4``,
    else the column), ``cosf``/``sinf`` of that FP32 product. ``form`` is how
    ``q0 * cos + sign * q1 * sin`` rounds: ``left`` (the left product fused, as
    nvcc and libNVVM contract), ``right``, or ``unfused``."""
    rows, hidden = qkv.shape[0], qkv.shape[1] // 3
    dim = hidden // heads
    half, quarter = dim // 2, dim // 4
    index = torch.arange(rows, device=qkv.device)
    j = torch.arange(half, device=qkv.device)
    position = torch.where(
        j[None, :] < quarter, (index // width)[:, None], (index % width)[:, None]
    ).float()
    angle = position * inv_freq[j % quarter][None, :]
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    out = qkv.clone()
    for part in range(2):
        x = qkv[:, part * hidden : (part + 1) * hidden].view(rows, heads, dim).float()
        x1, x2 = x[..., :half], x[..., half:]
        if form == "left":
            first = _fma32(x1, cos, -(x2 * sin))
            second = _fma32(x2, cos, x1 * sin)
        elif form == "right":
            first = _fma32(-x2, sin, x1 * cos)
            second = _fma32(x1, sin, x2 * cos)
        else:
            first = x1 * cos - x2 * sin
            second = x2 * cos + x1 * sin
        rotated = torch.cat((first, second), -1).bfloat16().view(rows, hidden)
        out[:, part * hidden : (part + 1) * hidden] = rotated
    return out


def _reference_gelu(h: torch.Tensor) -> torch.Tensor:
    """DeepSeek's reference GELU of BF16 values: ``F.gelu`` on the BF16 tensor (torch
    evaluates it in FP32 with CUDA's ``erff`` and rounds to BF16)."""
    return F.gelu(h)


def _reference_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """DeepSeek's reference MLP activation as vLLM's ``SiluAndMul`` runs it on BF16:
    SiLU rounded to BF16, times ``up``, rounded."""
    gate, up = gate_up.chunk(2, -1)
    return F.silu(gate) * up


# --- FP64 references (DeepSeek's vision math) -----------------------------------


def _linear64(x: torch.Tensor, layer) -> torch.Tensor:
    out = x.double() @ layer.weight.double().T
    if layer.bias is not None:
        out = out + layer.bias.double()
    return out


def _norm64(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight.double()


def _attention64(q, k, v):
    """``[rows, heads, dim]`` FP64 softmax attention over all rows."""
    scores = torch.einsum("thd,shd->hts", q, k) * q.shape[-1] ** -0.5
    return torch.einsum("hts,shd->thd", scores.softmax(-1), v)


def _vit_reference(vit, patches: torch.Tensor, height: int, width: int):
    """``deepseek_v4/common/vision.py``'s ViT in FP64, rounded to BF16 at each
    module's output (as the reference model's BF16 activations)."""
    config = vit.config
    heads, hidden = config.vision_n_heads, config.vision_dim
    dim, rows = hidden // heads, height * width
    rope_dim = dim // 2
    inv = 1.0 / (
        config.vision_rope_theta
        ** (torch.arange(0, rope_dim, 2, device=DEVICE).double() / rope_dim)
    )
    hpos = torch.arange(height, device=DEVICE)[:, None].expand(height, width)
    wpos = torch.arange(width, device=DEVICE)[None, :].expand(height, width)
    angles = (torch.stack((hpos, wpos), -1).reshape(-1, 2, 1).double() * inv).flatten(
        1
    )[:, None]
    cos, sin = angles.cos(), angles.sin()

    def rotate(x):
        first, second = x.chunk(2, -1)
        return _round(
            torch.cat((first * cos - second * sin, second * cos + first * sin), -1)
        )

    x = _round(_linear64(patches.reshape(rows, -1), vit.patch_embed.proj))
    for block in vit.blocks:
        qkv = _round(_linear64(_round(_norm64(x, block.norm1.weight)), block.attn.wqkv))
        q, k, v = (t.reshape(rows, heads, dim) for t in qkv.chunk(3, -1))
        attended = _round(_attention64(rotate(q), rotate(k), v)).reshape(rows, hidden)
        x = _round(x + _round(_linear64(attended, block.attn.wo)))
        gate_up = _round(
            _linear64(_round(_norm64(x, block.norm2.weight)), block.mlp.w1)
        )
        gate, up = gate_up.chunk(2, -1)
        activated = _round(_round(F.silu(gate)) * up)
        x = _round(x + _round(_linear64(activated, block.mlp.w2)))
    return _round(_norm64(x, vit.norm.weight))


def _merge64(x: torch.Tensor, height: int, width: int, ratio: int) -> torch.Tensor:
    """The reference's channel-major padded unfold (FP64)."""
    image = x.double().reshape(height, width, -1).permute(2, 0, 1)
    image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
    return F.unfold(image[None], ratio, stride=ratio)[0].T


def _aligner_reference(aligner, x: torch.Tensor, height: int, width: int):
    merged = _merge64(x, height, width, aligner.downsample_ratio)
    hidden = _round(F.gelu(_round(_linear64(merged, aligner.w1))))
    return _round(_linear64(hidden, aligner.w2))


# --- Copies: padding and merge --------------------------------------------------


@pytest.mark.parametrize("rows", [1, 7, 300])
def test_pad_columns_copies_and_zero_fills(rows):
    """588-wide patch rows at an 8-byte (not 16-byte) address become 640-wide rows
    with zero tails; rows past the call stay untouched."""
    kernel = vision.resolve(vision.pad_call(588))
    source = _randn(rows + 1, 588, seed=rows)[1:]
    assert source.data_ptr() % 16 == 8
    out = torch.full((rows + 3, 640), NAN, dtype=torch.bfloat16, device=DEVICE)
    kernel(source, out[:rows])
    _assert_bits(out[:rows, :588], source, "padded rows")
    assert bool((out[:rows, 588:] == 0).all()) and not bool(
        torch.signbit(out[:rows, 588:]).any()
    )
    assert bool(out[rows:].isnan().all()), "rows past the call were written"


@pytest.mark.parametrize("channels", [128, 1024])
@pytest.mark.parametrize("height, width", [(1, 1), (1, 5), (4, 7), (7, 11), (9, 9)])
def test_spatial_merge_matches_unfold(channels, height, width):
    kernel = vision.resolve(vision.merge_call(channels, 3))
    x = _randn(height * width, channels, seed=channels + height * width)
    rows = -(-height // 3) * -(-width // 3)
    out = torch.full((rows + 2, channels * 9), NAN, dtype=torch.bfloat16, device=DEVICE)
    kernel(x, out[:rows], height, width)
    _assert_bits(out[:rows], _merge64(x, height, width, 3).bfloat16(), "merge")
    assert bool(out[rows:].isnan().all())


# --- The residual add and RMSNorm -----------------------------------------------


def _add_norm(width, x, branch, weight):
    out = torch.empty_like(x)
    if branch is None:
        vision.resolve(vision.norm_call(width, add=False))(x, weight, out)
        return None, out
    summed = torch.empty_like(x)
    vision.resolve(vision.norm_call(width, add=True))(x, branch, weight, summed, out)
    return summed, out


@pytest.mark.parametrize("config", [SMALL, REAL], ids=["narrow", "v41"])
@pytest.mark.parametrize("rows", [1, 3, 257])
@pytest.mark.parametrize("add", [False, True])
def test_add_norm_is_b12x_arithmetic(config, rows, add):
    width = config.vision_dim
    x = _randn(rows, width, seed=rows, scale=3.0)
    branch = _randn(rows, width, seed=rows + 1, scale=2.0) if add else None
    weight = _norm_weight(width, seed=rows)
    weight[3] = -0.0  # B12X's (0 + weight) turns it positive
    summed, out = _add_norm(width, x, branch, weight)
    total = x
    if add:
        total = (x.float() + branch.float()).bfloat16()
        _assert_bits(summed, total, "residual sum")
    _assert_bits(out, _b12x_rmsnorm(total, weight), "normalized rows")
    # DeepSeek's FP64 norm of the same rounded sum: within one BF16 step.
    ref = _norm64(total, weight)
    assert _max_ulps(out, ref, 0.0) <= 1.0


def test_add_norm_nonfinite_like_the_reference():
    width, rows = REAL.vision_dim, 6
    x = _randn(rows, width, seed=5)
    branch = _randn(rows, width, seed=6)
    weight = _norm_weight(width, seed=7)
    clean_sum, clean = _add_norm(width, x, branch, weight)
    x[1, 7], x[4, 900] = NAN, INF
    summed, out = _add_norm(width, x, branch, weight)
    total = (x.float() + branch.float()).bfloat16()
    assert_same_nonfinite(summed, total, "sum: ")
    ref = _norm64(total, weight)
    assert_same_nonfinite(out, ref, "norm: ")
    assert_other_rows_equal(out, clean, [1, 4], "norm: ")
    assert_other_rows_equal(summed, clean_sum, [1, 4], "sum: ")


# --- RoPE -------------------------------------------------------------------------


@pytest.mark.parametrize("config", [SMALL, REAL], ids=["narrow", "v41"])
@pytest.mark.parametrize("height, width", [(1, 1), (3, 5), (5, 3), (7, 11), (24, 24)])
def test_rope_is_b12x_arithmetic(config, height, width):
    heads, hidden = config.vision_n_heads, config.vision_dim
    rows = height * width
    kernel = vision.resolve(vision.rope_call(heads, hidden // heads))
    inv = _inv_freq(config)
    buffer = torch.full(
        (rows + 5, 3 * hidden), NAN, dtype=torch.bfloat16, device=DEVICE
    )
    qkv = _randn(rows, 3 * hidden, seed=rows + hidden, scale=4.0)
    buffer[:rows] = qkv
    kernel(buffer[:rows], inv, width)
    _assert_bits(buffer[:rows], _b12x_rope(qkv, height, width, inv, heads), "rope")
    _assert_bits(buffer[:rows, 2 * hidden :], qkv[:, 2 * hidden :], "v moved")
    assert bool(buffer[rows:].isnan().all()), "rows past the call were written"
    # DeepSeek's FP64 rotation (true cos and sin of the exact angle): within one
    # BF16 step, plus 2^-12 for FP32's angle (position * inv_freq, rounded) and
    # cos/sin where the two products cancel.
    dim = hidden // heads
    angles = (
        torch.stack(
            (
                torch.arange(rows, device=DEVICE) // width,
                torch.arange(rows, device=DEVICE) % width,
            ),
            -1,
        ).double()[:, :, None]
        * inv.double()[None, None, :]
    )
    angles = angles.flatten(1)[:, None]
    for part in range(2):
        x = qkv[:, part * hidden : (part + 1) * hidden].double().view(rows, heads, dim)
        first, second = x.chunk(2, -1)
        ref = torch.cat(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            -1,
        ).view(rows, hidden)
        got = buffer[:rows, part * hidden : (part + 1) * hidden]
        assert _max_ulps(got, ref, 2.0**-12) <= 1.0


def test_rope_nonfinite_like_the_reference():
    heads, hidden, height, width = 16, 1024, 3, 4
    rows = height * width
    kernel = vision.resolve(vision.rope_call(heads, hidden // heads))
    inv = _inv_freq(REAL)
    qkv = _randn(rows, 3 * hidden, seed=3)
    clean = qkv.clone()
    kernel(clean, inv, width)
    qkv[2, 5] = NAN  # q head 0, first half
    qkv[7, hidden + 64 + 40] = INF  # k head 1, second half
    poisoned = qkv.clone()
    kernel(poisoned, inv, width)
    model = _b12x_rope(qkv, height, width, inv, heads)
    assert_same_nonfinite(poisoned, model)
    assert_finite_equal(poisoned, model)
    assert_other_rows_equal(poisoned, clean, [2, 7])


# --- Projections ------------------------------------------------------------------


def _projection_error(out, a, weight, bias):
    """Max ``|out - ref|`` in BF16 steps at the FP64 reference plus FP32 accumulation
    noise (``sqrt(K) 2^-24 sum |a w|``), and how many outputs differ from the
    correctly rounded reference, of how many."""
    a64, w64 = a.double(), weight.double()
    ref, scale = a64 @ w64.T, a64.abs() @ w64.abs().T
    if bias is not None:
        ref, scale = ref + bias.double(), scale + bias.double().abs()
    noise = scale * math.sqrt(a.shape[1]) * 2.0**-24
    error = ((out.double() - ref).abs() / (_bf16_ulp(ref) + noise)).max().item()
    return error, int((out != ref.bfloat16()).sum()), out.numel()


def _check_projection(name, ours):
    """Ours within one BF16 step of the reference (plus accumulation noise), and off
    the correctly rounded value no more often than for 1% of the outputs nor than
    B12X (1.5x), plus four values for sampling noise."""
    error, off, total = ours
    b12x_error, b12x_off = B12X_ERROR[name]
    message = (
        f"{name}: ours {error:.3f} steps, {off} of {total} off; "
        f"B12X {b12x_error:.3f} steps, {b12x_off} off"
    )
    assert error <= 1.0, message
    assert off <= min(0.01 * total, 1.5 * b12x_off) + 4, message


def _projection_cases(vit, aligner, rows: int):
    """Each projection at ``rows`` rows: the TileLang kernel the tower launches (the
    GEMV up to GEMV_ROWS rows, else the GEMM; for the two with an epilogue, the
    plain projection their epilogues start from), its layer, and its module path
    """
    block, out_dim = vit.blocks[0], aligner.out_dim
    small = rows <= vision.GEMV_ROWS

    def declared(module, role):
        return _kernel(module, role + "_rows" if small else role)

    def plain(n, k, bias):
        if small:
            return vision.resolve(vision.gemv_call(n, k, "bias" if bias else "none"))
        return vision.resolve(vision.linear_call(n, k, bias=bias))

    return {
        "qkv": (declared(vit, "qkv"), block.attn.wqkv, "blocks.0.attn.wqkv"),
        "wo": (declared(vit, "wo"), block.attn.wo, "blocks.0.attn.wo"),
        "mlp gate/up": (
            plain(*block.mlp.w1.weight.shape, False),
            block.mlp.w1,
            "blocks.0.mlp.w1",
        ),
        "mlp down": (declared(vit, "mlp_out"), block.mlp.w2, "blocks.0.mlp.w2"),
        "aligner w1": (
            plain(out_dim, aligner.w1.weight.shape[1], True),
            aligner.w1,
            "w1",
        ),
        "aligner w2": (declared(aligner, "w2"), aligner.w2, "w2"),
    }


def _project(kernel, layer, a: torch.Tensor) -> torch.Tensor:
    out = torch.empty(
        a.shape[0], layer.weight.shape[0], dtype=torch.bfloat16, device=DEVICE
    )
    if layer.bias is None:
        kernel(a, layer.weight, out)
    else:
        kernel(a, layer.weight, layer.bias, out)
    return out


# The aligner's merged capacity is 192 rows here.
@pytest.mark.parametrize("rows", [1, 77, 192])
def test_projections_against_fp64(real, rows):
    vit, aligner = real
    for name, (kernel, layer, _) in _projection_cases(vit, aligner, rows).items():
        a = _randn(rows, layer.weight.shape[1], seed=rows + layer.weight.shape[0])
        out = _project(kernel, layer, a)
        ours = _projection_error(out, a, layer.weight, layer.bias)
        _check_projection(f"{name} at {rows} rows", ours)


def _patch_embedding(vit, patches: torch.Tensor) -> torch.Tensor:
    """The tower's patch embedding of ``patches``: the GEMV on the 588-wide rows up
    to GEMV_ROWS rows, else the GEMM on rows and weight padded to 640."""
    proj, rows = vit.patch_embed.proj, patches.shape[0]
    flat = patches.reshape(rows, -1)
    if rows <= vision.GEMV_ROWS:
        return _project(_kernel(vit, "patch_rows"), proj, flat)
    padded = torch.empty(rows, 640, dtype=torch.bfloat16, device=DEVICE)
    weight = torch.empty(proj.weight.shape[0], 640, dtype=torch.bfloat16, device=DEVICE)
    _kernel(vit, "pad")(flat, padded)
    _kernel(vit, "pad")(proj.weight, weight)
    out = torch.empty(rows, proj.weight.shape[0], dtype=torch.bfloat16, device=DEVICE)
    _kernel(vit, "patch")(padded, weight, proj.bias, out)
    return out


@pytest.mark.parametrize("rows", [1, 77, 300])
def test_patch_embedding_against_fp64(real, rows):
    """The patch embedding: the GEMV on the 588-wide rows for one row, the GEMM on
    rows and weight padded to 640 (zero tails) for more."""
    vit, _ = real
    proj = vit.patch_embed.proj
    patches = _patches(REAL, rows + 1, seed=rows)[1:]
    flat = patches.reshape(rows, -1)
    out = _patch_embedding(vit, patches)
    ours = _projection_error(out, flat, proj.weight, proj.bias)
    _check_projection(f"patch embedding at {rows} rows", ours)


@pytest.mark.parametrize("role, k", [("patch", 64), ("patch_rows", 12)])
def test_bias_is_added_before_the_one_rounding(small, role, k):
    """``acc = 257`` (exact in FP32) plus a bias of 0.5 rounds to 258 in BF16; rounding
    the accumulator first (257 ties to 256) and then adding would give 256. The GEMM
    (K padded to 64) and the GEMV (the 12-wide patches)."""
    vit, _ = small
    kernel = _kernel(vit, role)  # [128, k] with bias
    a = torch.zeros(5, k, dtype=torch.bfloat16, device=DEVICE)
    a[:, :3] = 1
    weight = torch.zeros(128, k, dtype=torch.bfloat16, device=DEVICE)
    weight[:, 0], weight[:, 1], weight[:, 2] = 128, 128, 1
    bias = torch.full((128,), 0.5, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty(5, 128, dtype=torch.bfloat16, device=DEVICE)
    kernel(a, weight, bias, out)
    assert bool((out == 258).all()), out.unique().tolist()


@pytest.mark.parametrize("rows", [5, 9])
def test_projection_nonfinite_rows(small, rows):
    vit, _ = small
    layer = vit.blocks[0].attn.wqkv
    kernel = _kernel(vit, "qkv_rows" if rows <= vision.GEMV_ROWS else "qkv")
    a = _randn(rows, layer.weight.shape[1], seed=rows)
    clean = _project(kernel, layer, a)
    a[4, 3] = NAN
    out = _project(kernel, layer, a)
    assert bool(out[4].isnan().all())
    assert_other_rows_equal(out, clean, [4])


# --- Epilogues: GELU and SwiGLU ---------------------------------------------------


def _aligner_w1(aligner, merged: torch.Tensor, gelu: bool) -> torch.Tensor:
    """The aligner's first projection of ``merged`` on the tower's path for its rows
    (the GEMV up to GEMV_ROWS): with the GELU epilogue (the declared kernels) or
    without (biased, rounded)."""
    rows, k = merged.shape
    if rows <= vision.GEMV_ROWS:
        call = vision.gemv_call(aligner.out_dim, k, "gelu" if gelu else "bias")
    else:
        call = vision.linear_call(aligner.out_dim, k, bias=True, gelu=gelu)
    out = torch.empty(rows, aligner.out_dim, dtype=torch.bfloat16, device=DEVICE)
    vision.resolve(call)(merged, aligner.w1.weight, aligner.w1.bias, out)
    return out


@pytest.mark.parametrize("rows", [1, 33, 144])
def test_gelu_epilogue_is_the_reference(small, rows):
    """The GELU epilogue is DeepSeek's reference GELU (torch's ``F.gelu``) of the
    rounded biased projection, bit for bit, on the GEMV and the GEMM; a NaN row
    stays NaN."""
    _, aligner = small
    merged = _randn(rows, aligner.w1.weight.shape[1], seed=rows)
    activated = _aligner_w1(aligner, merged, gelu=True)
    hidden = _aligner_w1(aligner, merged, gelu=False)
    _assert_bits(activated, _reference_gelu(hidden), "GELU epilogue")
    merged[rows // 2, 0] = NAN
    poisoned = _aligner_w1(aligner, merged, gelu=True)
    assert bool(poisoned[rows // 2].isnan().all())
    assert_other_rows_equal(poisoned, activated, [rows // 2])


def _every_bf16() -> torch.Tensor:
    """Every BF16 value that is zero or normal, but negative zero: the inputs GELU
    sees from a projection (sums of zeros are positive; subnormal inputs depend on
    the tensor cores' denormal handling, not on GELU)."""
    values = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    values = values.view(torch.bfloat16).float()
    keep = torch.isfinite(values) & ((values.abs() >= 2.0**-126) | (values == 0))
    keep &= ~torch.signbit(values) | (values != 0)
    return values[keep].bfloat16().to(DEVICE)


def _gelu_table(aligner, values: torch.Tensor) -> torch.Tensor:
    """``values`` laid out as whole rows of the aligner's width, zeros after."""
    n = aligner.out_dim
    table = torch.zeros(-(-values.numel() // n) * n, dtype=torch.bfloat16)
    table = table.to(DEVICE)
    table[: values.numel()] = values
    return table.view(-1, n)


def _gelu_through_epilogue(aligner, table: torch.Tensor, gemv: bool) -> torch.Tensor:
    """GELU of each value of ``table`` through the fused epilogue: one-hot activation
    rows select a weight column holding a row of values, the bias is zero, so each
    output is GELU of a value exactly; the GEMV takes GEMV_ROWS rows a launch."""
    n, k = aligner.out_dim, aligner.w1.weight.shape[1]
    call = (
        vision.gemv_call(n, k, "gelu")
        if gemv
        else vision.linear_call(n, k, bias=True, gelu=True)
    )
    kernel, bias = vision.resolve(call), torch.zeros(n, device=DEVICE).bfloat16()
    step = vision.GEMV_ROWS if gemv else table.shape[0]
    outs = []
    for start in range(0, table.shape[0], step):
        rows = table[start : start + step]
        a = torch.zeros(rows.shape[0], k, dtype=torch.bfloat16, device=DEVICE)
        a[torch.arange(rows.shape[0]), torch.arange(rows.shape[0])] = 1
        weight = torch.zeros(n, k, dtype=torch.bfloat16, device=DEVICE)
        weight[:, : rows.shape[0]] = rows.T
        out = torch.empty(rows.shape, dtype=torch.bfloat16, device=DEVICE)
        kernel(a, weight, bias, out)
        outs.append(out)
    return torch.cat(outs)


@pytest.mark.parametrize("gemv", [False, True], ids=["gemm", "gemv"])
def test_gelu_epilogue_is_the_reference_everywhere(real, gemv):
    """Every BF16 input a projection can produce, through each fused epilogue,
    against torch's ``F.gelu``."""
    _, aligner = real
    table = _gelu_table(aligner, _every_bf16())
    got = _gelu_through_epilogue(aligner, table, gemv)
    _assert_bits(got, _reference_gelu(table), "GELU on every BF16 input")


def _mlp_in(vit, rows: int, seed: int):
    """The MLP's fused gate/up projection with SwiGLU, and the unfused projection's
    BF16 gate/up of the same rows (the GEMVs up to GEMV_ROWS rows, else the GEMMs:
    the fused and unfused accumulate alike)."""
    mlp = vit.blocks[0].mlp
    inter2, k = mlp.w1.weight.shape
    x = _randn(rows, k, seed=seed)
    activated = _mlp_activated(vit, x)
    gate_up = torch.empty(rows, inter2, dtype=torch.bfloat16, device=DEVICE)
    if rows <= vision.GEMV_ROWS:
        call = vision.gemv_call(inter2, k, "none")
    else:
        call = vision.linear_call(inter2, k, bias=False)
    vision.resolve(call)(x, mlp.w1.weight, gate_up)
    return x, activated, gate_up


def _mlp_activated(vit, x: torch.Tensor) -> torch.Tensor:
    """The tower's fused gate/up projection and SwiGLU of ``x``."""
    mlp, rows = vit.blocks[0].mlp, x.shape[0]
    role = "mlp_in_rows" if rows <= vision.GEMV_ROWS else "mlp_in"
    activated = torch.empty(
        rows, mlp.w1.weight.shape[0] // 2, dtype=torch.bfloat16, device=DEVICE
    )
    _kernel(vit, role)(x, mlp.w1.weight, activated)
    return activated


@pytest.mark.parametrize("rows", [1, 33, 144])
def test_swiglu_epilogue_against_the_reference(small, rows):
    """SwiGLU of the rounded gate/up, as the reference rounds them. ``__expf`` (B12X's
    fast exponential) and its reciprocal may move SiLU's last FP32 bits, so a SiLU
    may round to the neighbouring BF16 value; times ``up`` and rounded again, that
    is at most two BF16 steps, and rare."""
    vit, _ = small
    _, activated, gate_up = _mlp_in(vit, rows, rows)
    ref = _reference_swiglu(gate_up)
    steps = (activated.double() - ref.double()).abs() / _bf16_ulp(ref.double())
    assert steps.max().item() <= 2.0
    assert (activated != ref).float().mean().item() <= 2e-3


@pytest.mark.parametrize("rows", [5, 9])
def test_swiglu_epilogue_nonfinite_rows(small, rows):
    vit, _ = small
    x, clean, _ = _mlp_in(vit, rows, rows)
    x[2, 1] = NAN
    out = _mlp_activated(vit, x)
    assert bool(out[2].isnan().all())
    assert_other_rows_equal(out, clean, [2])


# --- Attention --------------------------------------------------------------------


def _attention(vit, qkv: torch.Tensor) -> torch.Tensor:
    hidden = vit.config.vision_dim
    out = torch.empty(qkv.shape[0], hidden, dtype=torch.bfloat16, device=DEVICE)
    _kernel(vit, "attention")(qkv, out)
    return out


def _split64(qkv: torch.Tensor, heads: int):
    rows, hidden = qkv.shape[0], qkv.shape[1] // 3
    return [
        t.double().reshape(rows, heads, hidden // heads) for t in qkv.split(hidden, -1)
    ]


@pytest.mark.parametrize("rows", [1, 77, 300, 576])
def test_attention_against_fp64(real, rows):
    """Relative RMS error at most 2^-8 and 1.25x B12X's, and the worst element at most
    one BF16 step past B12X's worst (FP64 softmax attention, in BF16 steps of the
    reference plus 2^-12 of its RMS)."""
    vit, _ = real
    heads = REAL.vision_n_heads
    qkv = _randn(rows, 3 * REAL.vision_dim, seed=rows, scale=2.0)
    ref = _attention64(*_split64(qkv, heads)).reshape(rows, -1)
    floor = 2.0**-12 * ref.square().mean().sqrt()
    out = _attention(vit, qkv)
    ours = (_relative(out, ref), _max_ulps(out, ref, floor))
    theirs = B12X_ERROR[f"attention at {rows} rows"]
    message = (
        f"{rows} rows: ours {ours[0]:.3e} relative, {ours[1]:.2f} steps; "
        f"B12X {theirs[0]:.3e} relative, {theirs[1]:.2f} steps"
    )
    assert ours[0] <= min(2.0**-8, 1.25 * theirs[0]), message
    assert ours[1] <= theirs[1] + 1.0, message


def test_attention_ignores_stale_capacity_rows(real):
    """Rows past the live ones hold NaN: no output changes."""
    vit, _ = real
    rows, hidden = 77, REAL.vision_dim
    qkv = _randn(rows, 3 * hidden, seed=1)
    clean = _attention(vit, qkv)
    buffer = torch.full(
        (vit.capacity, 3 * hidden), NAN, dtype=torch.bfloat16, device=DEVICE
    )
    buffer[:rows] = qkv
    _assert_bits(_attention(vit, buffer[:rows]), clean, "stale rows")


def test_attention_nonfinite_like_the_reference(real):
    """A NaN query poisons its row; a NaN key every row; a NaN value its column."""
    vit, _ = real
    rows, hidden, heads = 40, REAL.vision_dim, REAL.vision_n_heads
    qkv = _randn(rows, 3 * hidden, seed=2)
    clean = _attention(vit, qkv)
    for target, column in ((5, 3), (11, hidden + 70), (17, 2 * hidden + 130)):
        poisoned = qkv.clone()
        poisoned[target, column] = NAN
        out = _attention(vit, poisoned)
        ref = _attention64(*_split64(poisoned, heads)).reshape(rows, -1)
        assert_same_nonfinite(out, ref, f"NaN at ({target}, {column}): ")
        if column < hidden:
            assert_other_rows_equal(out, clean, [target])


# --- The tower and aligner ----------------------------------------------------------


def _tower_case(vit, aligner, height, width, seed):
    rows = height * width
    patches = _patches(vit.config, rows, seed)
    encoded = vit(patches, height, width)
    aligned = aligner(encoded, height, width)
    vit_ref = _vit_reference(vit, patches, height, width)
    aligner_ref = _aligner_reference(aligner, encoded, height, width)
    ours = (_relative(encoded, vit_ref), _relative(aligned, aligner_ref))
    theirs = B12X_ERROR[f"tower {vit.config.vision_dim} {height}x{width}"]
    message = (
        f"{height}x{width}: ViT {ours[0]:.3e}, aligner {ours[1]:.3e} relative to FP64; "
        f"B12X {theirs[0]:.3e}, {theirs[1]:.3e}"
    )
    # Both round the same activations to BF16; their errors come from FP32
    # accumulation orders and attention, so ours may differ by sampling noise.
    for stage in (0, 1):
        assert ours[stage] <= min(2.0**-7, 1.5 * theirs[stage] + 2.0**-14), message


@pytest.mark.parametrize("height, width", SMALL_GRIDS)
def test_small_tower_against_reference(small, height, width):
    _tower_case(*small, height, width, seed=height * 1000 + width)


@pytest.mark.parametrize("height, width", REAL_GRIDS)
def test_v41_tower_against_reference(real, height, width):
    _tower_case(*real, height, width, seed=height * 1000 + width)


def test_images_do_not_contaminate_each_other(small):
    """Every capacity buffer is reused: an image's output does not depend on the
    image before it (bits), and the tower's output equals a fresh run's."""
    vit, aligner = small
    patches = _patches(SMALL, 28, seed=3)
    first = aligner(vit(patches, 4, 7), 4, 7)
    big = _patches(SMALL, 144, seed=4) * 4
    aligner(vit(big, 12, 12), 12, 12)
    _assert_bits(aligner(vit(patches, 4, 7), 4, 7), first, "after a larger image")


def test_graph_replay_allocates_nothing(small):
    vit, aligner = small
    height, width = 7, 11
    rows = height * width
    patches = _patches(SMALL, rows, seed=5)
    encoded = torch.empty(rows, SMALL.vision_dim, dtype=torch.bfloat16, device=DEVICE)
    merged = -(-height // 3) * -(-width // 3)
    out = torch.empty(merged, SMALL.hidden_size, dtype=torch.bfloat16, device=DEVICE)
    vit(patches, height, width, out=encoded)
    aligner(encoded, height, width, out=out)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            vit(patches, height, width, out=encoded)
            aligner(encoded, height, width, out=out)
    torch.cuda.current_stream().wait_stream(side)
    patches.copy_(_patches(SMALL, rows, seed=6))
    encoded.fill_(NAN)
    out.fill_(NAN)
    pointers = (encoded.data_ptr(), out.data_ptr())
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert (encoded.data_ptr(), out.data_ptr()) == pointers
    expected = aligner(vit(patches, height, width), height, width)
    _assert_bits(out, expected, "graph replay")
    graph.reset()


def test_data_parallel_runner_on_one_rank(small, monkeypatch):
    """The sharded runner keeps each image's rows and order (one rank: no gather)."""
    import vllm.distributed as distributed

    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: 0)
    vit, aligner = small
    grids = [[4, 7], [1, 1], [7, 11], [2, 5]]
    images = [_patches(SMALL, h * w, seed=10 + i) for i, (h, w) in enumerate(grids)]
    result = tilelang_vision.run_dp_sharded_vision_tower(
        vit, aligner, torch.cat(images), grids
    )
    assert len(result) == len(grids)
    for image, (h, w), got in zip(images, grids, result):
        _assert_bits(got, aligner(vit(image, h, w), h, w), f"image {h}x{w}")


def test_rejects_unprepared_and_misaligned_inputs():
    vit, aligner = _build(SMALL, seed=1)
    patches = _patches(SMALL, 4, seed=1)
    with pytest.raises(RuntimeError, match="not prepared"):
        vit(patches, 2, 2)
    with pytest.raises(ValueError, match="capacity"):
        vit(_patches(SMALL, 145, seed=1), 1, 145)
    x = torch.zeros(4, SMALL.vision_dim, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError, match="not prepared"):
        aligner(x, 2, 2)


# --- Declarations -----------------------------------------------------------------


class _Recording:
    def __init__(self, kernel, name: str, launched: set):
        self.kernel, self.name, self.launched = kernel, name, launched

    def __call__(self, *args):
        self.launched.add(self.name)
        return self.kernel(*args)


def test_declarations_cover_every_launch(monkeypatch):
    """The worker's entry point declares exactly the calls the modules bind (after
    the kernel cache freezes, an undeclared one raises KernelNotPreparedError), and
    a small image (the GEMVs) and a large one (the GEMMs) launch every bound kernel
    and nothing else."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    vit, aligner = _build(SMALL, seed=2)
    registry = JitWarmupRegistry(None)
    with registry.activate():
        prepare.declare_modules(vit, aligner)
    assert vit._kernels is None and vit._workspace is None, "bound before preparing"
    declared = {prepare._CallKey(*call) for call in prepare.declared_calls(registry)}
    expected = {
        prepare._CallKey(*call)
        for module in (vit, aligner)
        for call in module.kernel_calls().values()
    }
    assert declared == expected
    resolved = set()
    original = vision.resolve

    def record(call):
        resolved.add(prepare._CallKey(*call))
        return original(call)

    monkeypatch.setattr(vision, "resolve", record)
    prepare.prepare_declared(registry)
    prepare.bind_declared(registry)
    assert resolved == declared
    launched: set = set()
    for module in (vit, aligner):
        module._kernels = {
            name: _Recording(kernel, f"{type(module).__name__}.{name}", launched)
            for name, kernel in module._kernels.items()
        }
    for height, width in ((2, 3), (12, 12)):
        rows = height * width
        aligner(vit(_patches(SMALL, rows, seed=2), height, width), height, width)
    assert launched == {
        f"{type(module).__name__}.{name}"
        for module in (vit, aligner)
        for name in module.kernel_calls()
    }


def _build_tp_shards(ref_vit, ref_aligner, tp_size: int):
    """Build ``tp_size`` sharded ViT and aligner instances and load their shards
    from ``ref_vit`` and ``ref_aligner`` via each parameter's ``weight_loader``."""
    config = ref_vit.config
    ref_vit_params = dict(ref_vit.named_parameters())
    ref_aligner_params = dict(ref_aligner.named_parameters())
    vits, aligners = [], []
    for rank in range(tp_size):
        with torch.device(DEVICE):
            vit = tilelang_vision.DeepseekV4ViT(config, tp_size=tp_size, tp_rank=rank)
            aligner = tilelang_vision.DeepseekV4Aligner(
                config, tp_size=tp_size, tp_rank=rank
            )
        # Fill with NaN first so any unzeroed TP padding is caught immediately.
        for module, ref_params in ((vit, ref_vit_params), (aligner, ref_aligner_params)):
            for name, param in module.named_parameters():
                param.data.fill_(NAN)
                param.weight_loader(param, ref_params[name])
                assert bool(torch.isfinite(param).all()), f"rank {rank} {name} has non-finite values"
        vit.declare_kernels()
        aligner.declare_kernels()
        vits.append(vit)
        aligners.append(aligner)
    return vits, aligners


@pytest.mark.parametrize("tp_size", [3, 4])
@pytest.mark.parametrize("height, width", [(1, 1), (5, 7), (13, 17)])
def test_tp_sharded_tower_against_reference(real, monkeypatch, tp_size, height, width):
    """Tensor-parallel ViT + aligner at TP3 (16 -> 18 heads, 2816 -> 2880 MLP,
    5120 -> 5184 aligner zero-padded) and TP4 (exact 4-way shards) match the
    FP64 reference within the tower's error bound and emit identical outputs on
    every rank."""
    import threading
    import vllm.distributed as distributed

    ref_vit, ref_aligner = real
    vits, aligners = _build_tp_shards(ref_vit, ref_aligner, tp_size)
    barrier = threading.Barrier(tp_size)
    partials: list[torch.Tensor | None] = [None] * tp_size
    local_rank = threading.local()

    def sync_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        r = local_rank.rank
        partials[r] = tensor.clone()
        barrier.wait()
        acc = partials[0].clone()
        for other in partials[1:]:
            acc = (acc.float() + other.float()).bfloat16()
        barrier.wait()
        return acc

    monkeypatch.setattr(distributed, "tensor_model_parallel_all_reduce", sync_all_reduce)
    rows = height * width
    patches = _patches(REAL, rows, seed=height * 1000 + width)
    encoded_ranks: list[torch.Tensor | None] = [None] * tp_size
    aligned_ranks: list[torch.Tensor | None] = [None] * tp_size
    errors: list[BaseException] = []

    def worker(rank: int) -> None:
        try:
            local_rank.rank = rank
            enc = vits[rank](patches, height, width)
            aln = aligners[rank](enc, height, width)
            encoded_ranks[rank] = enc
            aligned_ranks[rank] = aln
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(tp_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    for r in range(1, tp_size):
        _assert_bits(encoded_ranks[r], encoded_ranks[0], f"TP{tp_size} ViT rank {r}")
        _assert_bits(aligned_ranks[r], aligned_ranks[0], f"TP{tp_size} aligner rank {r}")

    vit_ref = _vit_reference(ref_vit, patches, height, width)
    aligner_ref = _aligner_reference(ref_aligner, encoded_ranks[0], height, width)
    ours = (_relative(encoded_ranks[0], vit_ref), _relative(aligned_ranks[0], aligner_ref))
    # RowParallel projections (wo, mlp.w2, aligner.w2) round each rank's partial
    # sum to BF16 before tensor_model_parallel_all_reduce, so compare against the
    # tower's 2^-7 BF16 relative RMS ceiling rather than TP1's single-accumulator GEMV error.
    for stage in (0, 1):
        assert ours[stage] <= 2.0**-7


def test_tp_sharding_weight_padding_and_data_parallel_override(real):
    """Verify TP3 zero-padding of tail heads/MLP/aligner columns, TP4 exact 4-way
    memory reduction, and ``use_data_parallel=True`` replication."""
    ref_vit, ref_aligner = real
    vits3, aligners3 = _build_tp_shards(ref_vit, ref_aligner, 3)
    # TP3: 16 heads -> 6 heads/rank (384 cols); rank 2 has 4 valid heads (256 cols) + 128 zero cols.
    assert vits3[2].local_heads == 6 and vits3[2].qkv_width == 384
    wqkv2 = vits3[2].blocks[0].attn.wqkv
    for p in range(3):
        assert bool((wqkv2.weight[p * 384 + 256 : (p + 1) * 384] == 0).all())
        assert bool((wqkv2.bias[p * 384 + 256 : (p + 1) * 384] == 0).all())
    wo2 = vits3[2].blocks[0].attn.wo
    assert bool((wo2.weight[:, 256:] == 0).all())
    assert bool((wo2.bias == 0).all())
    # TP3 MLP: 2816 -> 960/rank; rank 2 has 896 valid + 64 zero cols.
    w1_2 = vits3[2].blocks[0].mlp.w1
    for p in range(2):
        assert bool((w1_2.weight[p * 960 + 896 : (p + 1) * 960] == 0).all())
    assert bool((vits3[2].blocks[0].mlp.w2.weight[:, 896:] == 0).all())
    # TP3 aligner: 5120 -> 1728/rank; rank 2 has 1664 valid + 64 zero cols.
    assert bool((aligners3[2].w1.weight[1664:] == 0).all())
    assert bool((aligners3[2].w1.bias[1664:] == 0).all())
    assert bool((aligners3[2].w2.weight[:, 1664:] == 0).all())
    assert bool((aligners3[2].w2.bias == 0).all())

    # TP4: exact 1/4 sharding of attention, MLP, and aligner weights.
    vits4, aligners4 = _build_tp_shards(ref_vit, ref_aligner, 4)
    assert vits4[0].local_heads == 4 and vits4[0].local_inter == 704
    assert aligners4[0].local_out_dim == 1280
    assert aligners4[0].w1.weight.nbytes * 4 == ref_aligner.w1.weight.nbytes
    assert aligners4[0].w2.weight.nbytes * 4 == ref_aligner.w2.weight.nbytes

    # use_data_parallel=True keeps full replicated weights even when tp_size is passed.
    with torch.device("meta"):
        dp_vit = tilelang_vision.DeepseekV4ViT(
            REAL, tp_size=4, tp_rank=2, use_data_parallel=True
        )
        dp_aligner = tilelang_vision.DeepseekV4Aligner(
            REAL, tp_size=4, tp_rank=2, use_data_parallel=True
        )
    assert dp_vit.tp_size == 1 and dp_vit.local_heads == 16
    assert dp_aligner.tp_size == 1 and dp_aligner.local_out_dim == 5120
