# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TileLang quantization helpers and per-token cast kernels for DeepSeek-V4.1."""

import functools
import math
from dataclasses import dataclass
from typing import Optional, Union

import tilelang
import torch
from tilelang import language as T

QuantTensor = tuple[torch.Tensor, torch.Tensor]

SMALL_BATCH_THRESHOLD = 4096

_DTYPE_TO_STR = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "e4m3",
    torch.int8: "e2m1",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}

_DEFAULT_CLAMP_MIN_VALUES = {
    ("e4m3", False): 1e-4,
    ("e4m3", True): float(T.max_value(T.float8_e4m3fn)) * (2**-9),
    ("e2m1", False): float(T.max_value(T.float4_e2m1fn) * (2**-126)),
    ("e2m1", True): float(T.max_value(T.float4_e2m1fn)) * (2**-9),
}
_clamp_min_values = _DEFAULT_CLAMP_MIN_VALUES.copy()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length()


def dtype_to_str(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported: {list(_DTYPE_TO_STR.keys())}"
        )
    return _DTYPE_TO_STR[dtype]


def str_to_dtype(fmt: str) -> torch.dtype:
    if fmt not in _STR_TO_DTYPE:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported: {list(_STR_TO_DTYPE.keys())}"
        )
    return _STR_TO_DTYPE[fmt]


def _validate_quant_fmt(fmt: str) -> None:
    if fmt not in ("e4m3", "e2m1"):
        raise ValueError(f"Unsupported quant fmt {fmt}")


def set_amax_clamp_for_quant(
    fmt: str, use_e4m3_sf: bool, clamp_min_value: float
) -> None:
    _validate_quant_fmt(fmt)
    _clamp_min_values[(fmt, use_e4m3_sf)] = clamp_min_value


def get_amax_clamp_for_quant(fmt: str, use_e4m3_sf: bool) -> float:
    _validate_quant_fmt(fmt)
    return _clamp_min_values[(fmt, use_e4m3_sf)]


@functools.lru_cache(maxsize=8)
def get_best_vectorize_size(dtype: T.dtype) -> int:
    major, _ = torch.cuda.get_device_capability()
    return (16 if major < 10 else 32) // dtype.bytes


@dataclass(frozen=True)
class BaseCastConfig:
    torch_dtype: torch.dtype = torch.float8_e4m3fn
    sf_block: tuple[int, int] = (1, 1)
    use_tma_aligned_col_major_sf: bool = False
    use_packed_ue8m0: bool = False
    use_e4m3_sf: bool = False

    @property
    def with_sf(self) -> bool:
        return self.torch_dtype in [torch.float8_e4m3fn, torch.int8, torch.uint32]

    @property
    def dtype(self) -> T.dtype:
        return (
            T.dtype(self.torch_dtype)
            if self.torch_dtype != torch.int8
            else T.float4_e2m1fn
        )

    @property
    def sf_torch_dtype(self) -> torch.dtype:
        if self.use_packed_ue8m0:
            return torch.uint8
        elif self.use_e4m3_sf:
            return torch.float8_e4m3fn
        else:
            return torch.float32

    @property
    def sf_dtype(self) -> T.dtype:
        return T.dtype(self.sf_torch_dtype)

    @property
    def sf_col_pack(self) -> bool:
        return self.use_packed_ue8m0 and self.sf_block[1] == 1


@dataclass(frozen=True)
class CastInputConfig(BaseCastConfig):
    pass


@dataclass(frozen=True)
class CastOutputConfig(BaseCastConfig):
    round_sf: bool = False
    clamp_min_value: Optional[float] = None


def get_sf_inv_dtype(value_dtype: T.dtype, out_config: CastOutputConfig) -> T.dtype:
    if value_dtype == T.bfloat16 and out_config.round_sf:
        return T.bfloat16
    return T.float32


def get_packed_ue8m0_pack_factor() -> int:
    return 4


def get_packed_ue8m0_torch_dtype() -> torch.dtype:
    return torch.int32


@functools.lru_cache
def get_cast_input_config_impl(
    dtype: torch.dtype,
    sf_block: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
) -> CastInputConfig:
    if sf_block is None:
        assert dtype in (torch.bfloat16, torch.float32)
        return CastInputConfig(dtype)
    else:
        assert dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8)
        return CastInputConfig(
            dtype,
            sf_block,
            use_tma_aligned_col_major_sf,
            use_packed_ue8m0,
            use_e4m3_sf,
        )


def get_cast_input_and_config(
    x: Union[torch.Tensor, QuantTensor],
    sf_block: Optional[tuple[int, int]],
) -> tuple[torch.Tensor, Optional[torch.Tensor], CastInputConfig]:
    if isinstance(x, tuple):
        x, x_sf = x
        assert (
            isinstance(sf_block, tuple)
            and isinstance(x, torch.Tensor)
            and isinstance(x_sf, torch.Tensor)
        )
        assert x_sf.dtype in (
            torch.float32,
            torch.float8_e4m3fn,
            get_packed_ue8m0_torch_dtype(),
        )

        use_tma_aligned_col_major_sf = False
        use_packed_ue8m0 = False
        use_e4m3_sf = False
        if x_sf.stride(0) == 1 and x_sf.stride(1) != 1:
            use_tma_aligned_col_major_sf = True
            x_sf = x_sf.T
        else:
            assert x_sf.stride(1) == 1
        if x_sf.dtype == get_packed_ue8m0_torch_dtype():
            use_packed_ue8m0 = True
            x_sf = x_sf.view(torch.uint8)
        elif x_sf.dtype == torch.float8_e4m3fn:
            use_e4m3_sf = True

        config = get_cast_input_config_impl(
            x.dtype,
            sf_block,
            use_tma_aligned_col_major_sf,
            use_packed_ue8m0,
            use_e4m3_sf,
        )
        return x, x_sf, config
    else:
        assert sf_block is None and isinstance(x, torch.Tensor)
        config = get_cast_input_config_impl(x.dtype)
        return x, None, config


@functools.lru_cache
def get_cast_output_config_impl(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,
    use_e4m3_sf: bool,
    clamp_min_value: Optional[float],
) -> CastOutputConfig:
    assert fmt in ("fp32", "bf16", "e4m3", "e2m1")
    assert not use_packed_ue8m0 or round_sf
    assert not (round_sf and use_e4m3_sf)
    return CastOutputConfig(
        torch_dtype=str_to_dtype(fmt),
        sf_block=sf_block,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        clamp_min_value=clamp_min_value,
    )


def get_cast_output_config(
    fmt: str,
    sf_block: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
) -> CastOutputConfig:
    if fmt in ("e4m3", "e2m1"):
        clamp_min_value = get_amax_clamp_for_quant(fmt, use_e4m3_sf)
    else:
        clamp_min_value = None
    return get_cast_output_config_impl(
        fmt,
        sf_block,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        use_e4m3_sf,
        clamp_min_value,
    )


def get_logical_hidden(hidden: int, dtype: torch.dtype) -> int:
    """Compute hidden size when ``torch.int8`` is used for packing FP4."""
    return hidden if dtype != torch.int8 else hidden * 2


def get_physical_hidden(hidden: int, dtype: torch.dtype) -> int:
    """Compute hidden size when ``torch.int8`` is used for packing FP4."""
    return hidden if dtype != torch.int8 else hidden // 2


def get_sf_shape(shape: tuple[int, int], config: BaseCastConfig) -> tuple[int, int]:
    num_block_m = ceil_div(shape[0], config.sf_block[0])
    num_block_k = ceil_div(shape[1], config.sf_block[1])

    if not config.sf_col_pack:
        if config.use_packed_ue8m0:
            num_block_k = ceil_div(num_block_k, get_packed_ue8m0_pack_factor())
        if config.use_tma_aligned_col_major_sf:
            num_block_m, num_block_k = num_block_k, num_block_m
        if config.use_packed_ue8m0:
            num_block_k *= get_packed_ue8m0_pack_factor()
    else:
        num_block_m = ceil_div(num_block_m, get_packed_ue8m0_pack_factor())
        num_block_k *= get_packed_ue8m0_pack_factor()

    return num_block_m, num_block_k


def alloc_scaling_factors(
    shape: tuple[int, int],
    out_config: BaseCastConfig,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Allocate scaling factors for quantization."""
    sf_shape = get_sf_shape(shape, out_config)
    if out_config.use_tma_aligned_col_major_sf:
        aligned_sf_shape_1 = align(sf_shape[1], 16 // out_config.sf_dtype.bytes)
        scaling_factor = torch.empty(
            size=(sf_shape[0], aligned_sf_shape_1),
            dtype=out_config.sf_torch_dtype,
            device=device,
        )
        if aligned_sf_shape_1 == 0:
            scaling_factor.as_strided_(
                scaling_factor.shape, (aligned_sf_shape_1, 1)
            )
        return scaling_factor[:, : sf_shape[1]]
    else:
        return torch.empty(
            size=sf_shape,
            dtype=out_config.sf_torch_dtype,
            device=device,
        )


def cast_epilogue(out_sf: torch.Tensor, config: BaseCastConfig) -> torch.Tensor:
    """Post-process the sf-factor tensor after a cast kernel launch."""
    if config.use_packed_ue8m0:
        out_sf = out_sf.view(dtype=get_packed_ue8m0_torch_dtype())
    return out_sf.T if config.use_tma_aligned_col_major_sf else out_sf


IMPORT_BF16X2_FP8_CVT = r"""
#ifndef VLLM_BF16X2_FP8_CVT_H_
#define VLLM_BF16X2_FP8_CVT_H_
#if (__CUDACC_VER_MAJOR__ < 13) || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ < 1)
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_fp8x2_storage_t __vllm_cvt_bfloat162_to_fp8x2(
    const __nv_bfloat162 src,
    const __nv_fp8_interpretation_t fp8_interpretation) {
  const unsigned int u = *reinterpret_cast<const unsigned int*>(&src);
  const float2 f2 = make_float2(
      __uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
  return __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, fp8_interpretation);
}
#define __tl_cvt_bfloat162_to_fp8x2 __vllm_cvt_bfloat162_to_fp8x2
#endif
#endif
"""


@T.macro
def get_sf_and_inv(
    amax: float, out_config: CastOutputConfig, sf_inv_dtype: T.dtype = T.float32
):
    clamped_amax = T.max(amax, out_config.clamp_min_value)

    max_value = T.max_value(out_config.dtype)
    sf = T.alloc_var(T.float32)
    if out_config.round_sf:
        sf = clamped_amax * (T.float32(1) / max_value)
    else:
        sf = clamped_amax / max_value
        if out_config.use_e4m3_sf:
            sf_fp8 = T.cast(sf, T.float8_e4m3fn)
            return sf_fp8, 1 / T.cast(sf_fp8, T.float32)
        else:
            return sf, max_value / clamped_amax

    bits = T.reinterpret(sf, T.uint32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    if sf_inv_dtype == T.bfloat16:
        sf_inv = T.reinterpret(T.uint16(127 - exp_sf) << 7, T.bfloat16)
    else:
        sf_inv = T.reinterpret((127 - exp_sf) << 23, T.float32)
    if out_config.use_packed_ue8m0:
        return T.uint8(exp_sf + 127), sf_inv
    else:
        return T.reinterpret((127 + exp_sf) << 23, T.float32), sf_inv


@T.macro
def get_sf_index(m_idx: int, k_idx: int, config: BaseCastConfig):
    pack_factor = get_packed_ue8m0_pack_factor()
    if config.sf_col_pack:
        return m_idx // pack_factor, k_idx * pack_factor + m_idx % pack_factor
    if config.use_packed_ue8m0 and config.use_tma_aligned_col_major_sf:
        return k_idx // pack_factor, m_idx * pack_factor + k_idx % pack_factor
    if config.use_tma_aligned_col_major_sf:
        return k_idx, m_idx
    return m_idx, k_idx


@T.macro
def load_sf(tensor: T.Tensor, m_idx: int, k_idx: int, config: BaseCastConfig):
    row, col = get_sf_index(m_idx, k_idx, config)
    return tensor[row, col]


@T.macro
def transform_sf(
    sf: Union[T.float32, T.float8_e4m3fn, T.uint8], config: BaseCastConfig
) -> T.float32:
    if config.use_packed_ue8m0:
        return T.reinterpret(T.uint32(sf) << 23, T.float32)
    elif config.use_e4m3_sf:
        return T.cast(sf, T.float32)
    else:
        return sf


@T.macro
def store_sf(
    tensor: T.Tensor,
    sf: Union[T.float32, T.float8_e4m3fn, T.uint8],
    m_idx: int,
    k_idx: int,
    config: BaseCastConfig,
):
    row, col = get_sf_index(m_idx, k_idx, config)
    tensor[row, col] = sf


def unpack_from_e2m1fn_x2(
    x: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Decode a uint8/int8 tensor of packed fp4 values to ``out_dtype``."""
    assert x.dtype == torch.int8 or x.dtype == torch.uint8

    if x.ndim == 0:
        raise ValueError("x must have at least 1 dimension so the last dim can be doubled")

    lo = (x & 0x0F).to(torch.int16)
    hi = ((x >> 4) & 0x0F).to(torch.int16)

    def decode_fp4_e2m1(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1

        sign = torch.where(
            s == 1,
            torch.tensor(-1.0, device=n.device),
            torch.tensor(1.0, device=n.device),
        )
        bias = 1
        sub = (m.to(torch.float32) * 0.5) * (2.0 ** (1 - bias))
        norm = (1.0 + m.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device),
            (e - bias).to(torch.float32),
        )
        val = torch.where(e == 0, sub, norm)
        return (val * sign).to(out_dtype)

    flo = decode_fp4_e2m1(lo)
    fhi = decode_fp4_e2m1(hi)
    return torch.stack([flo, fhi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)


@T.macro
def rbits_hash(values, quad_idx):
    h = T.alloc_var(T.uint32)
    h = (
        T.reinterpret(values[0], T.uint32) * T.uint32(0x5671D42B)
        + T.reinterpret(values[1], T.uint32) * T.uint32(0x9995E499)
        + T.reinterpret(values[2], T.uint32) * T.uint32(0xACE1B8A5)
        + T.reinterpret(values[3], T.uint32) * T.uint32(0xE153538D)
        + T.cast(quad_idx, T.uint32)
    )
    h = h ^ (h >> 23)
    h = h * T.uint32(0x7FEB352D)
    h = h ^ (h >> 16)
    h = h * T.uint32(0x846CA68B)
    h = h ^ (h >> 11)
    return h


@tilelang.jit(
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_per_token_cast_kernel_cuda(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    small_batch: bool = False,
    stochastic_cast: bool = False,
    use_pdl: bool = False,
):
    if small_batch:
        num_threads, num_elems_per_thread = 64, 32
    else:
        num_threads, num_elems_per_thread = 32, 64

    num_elems_per_block = num_threads * num_elems_per_thread
    num_per_channels = out_config.sf_block[1]

    if is_power_of_two(num_per_channels):
        block_k = max(num_per_channels, math.gcd(num_elems_per_block, hidden))
    elif hidden == num_per_channels:
        assert not in_config.with_sf and not cast_only and not sf_only
        block_k = align(
            hidden, num_threads * (2 if out_config.dtype == T.float4_e2m1fn else 1)
        )
        num_per_channels = block_k
    else:
        raise ValueError("Cast granularity must be equal to hidden or be power of two")

    assert block_k % num_per_channels == 0
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    num_groups = block_k // num_per_channels

    num_vectorize = min(
        get_best_vectorize_size(in_config.dtype),
        math.gcd(block_m * block_k // num_threads, 32),
    )
    sf_value_dtype = T.float32 if in_config.with_sf else in_config.dtype
    sf_inv_dtype = get_sf_inv_dtype(sf_value_dtype, out_config)
    if stochastic_cast:
        assert (
            num_vectorize >= 4
        ), f"stochastic_cast requires vectorize width >= 4, got {num_vectorize}"

    if in_config.with_sf:
        assert not cast_only and not sf_only
        assert block_k % num_vectorize == 0
        assert num_per_channels >= num_vectorize, (
            f"num_per_channels ({num_per_channels}) must be >= num_vectorize "
            f"({num_vectorize}); on sm>=10 with float8, num_vectorize=32, so "
            "num_per_channels=16 would cause a zero-dimension reshape"
        )
        assert (
            block_m % in_config.sf_block[0] == 0
            or in_config.sf_block[0] % block_m == 0
        )
        assert (
            block_k % in_config.sf_block[1] == 0
            or in_config.sf_block[1] % block_k == 0
        )

    num_tokens = T.dynamic("num_tokens")
    in_sf_stride = T.dynamic("in_sf_stride")
    out_sf_stride = T.dynamic("out_sf_stride")
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)

    def x_layout_fn(i: int, j: int):
        idx = i * block_k + j
        return (
            idx // num_vectorize % num_threads,
            idx // (num_vectorize * num_threads) * num_vectorize + idx % num_vectorize,
        )

    blocks_per_token = T.ceildiv(hidden, block_k)

    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        T.assume(num_tokens > 0)
        with T.Kernel(
            T.ceildiv(num_tokens, block_m) * blocks_per_token, threads=num_threads
        ) as pid:
            if sf_inv_dtype == T.bfloat16:
                T.import_source(IMPORT_BF16X2_FP8_CVT)
            pid_token = pid // blocks_per_token
            pid_hidden = pid % blocks_per_token
            if use_pdl:
                T.pdl_sync()
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((block_m, num_groups), sf_inv_dtype)
            if stochastic_cast or in_config.with_sf:
                out_fragment_or_shared = T.alloc_shared(
                    (block_m, block_k), out_config.dtype
                )
            else:
                out_fragment_or_shared = T.alloc_fragment(
                    (block_m, block_k), out_config.dtype
                )

            T.annotate_layout(
                {
                    x_fragment: T.Fragment(
                        (block_m, block_k),
                        forward_fn=x_layout_fn,
                    )
                }
            )

            T.copy(
                x[pid_token * block_m, pid_hidden * block_k],
                x_fragment,
                disable_tma=True,
            )

            if in_config.with_sf:
                num_sf_rows_per_block = T.ceildiv(block_m, in_config.sf_block[0])
                num_sf_cols_per_block = T.ceildiv(block_k, in_config.sf_block[1])
                x_sf_fragment = T.alloc_fragment(
                    (num_sf_rows_per_block, num_sf_cols_per_block), T.float32
                )
                for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                    m_idx = pid_token * block_m // in_config.sf_block[0] + i
                    k_idx = pid_hidden * block_k // in_config.sf_block[1] + j
                    x_sf_fragment[i, j] = transform_sf(
                        load_sf(x_sf, m_idx, k_idx, in_config), in_config
                    )

                stage1_amax_fragment = T.alloc_fragment(
                    (block_m, block_k // num_vectorize), T.float16
                )
                x_stage1_fragment_reshaped = T.reshape(
                    x_fragment, [block_m, block_k // num_vectorize, num_vectorize]
                )
                T.reduce_absmax(
                    x_stage1_fragment_reshaped, stage1_amax_fragment, dim=-1
                )

                stage2_amax_fragment = T.alloc_fragment(
                    (block_m, block_k // num_vectorize), T.float32
                )
                T.clear(stage2_amax_fragment)
                for i, j in T.Parallel(block_m, block_k // num_vectorize):
                    stage2_amax_fragment[i, j] = (
                        T.float32(stage1_amax_fragment[i, j])
                        * x_sf_fragment[
                            i // in_config.sf_block[0],
                            j * num_vectorize // in_config.sf_block[1],
                        ]
                    )

                stage2_amax_fragment_reshaped = T.reshape(
                    stage2_amax_fragment,
                    [block_m, num_groups, block_k // num_vectorize // num_groups],
                )
                T.reduce_max(stage2_amax_fragment_reshaped, sf_inv_fragment, dim=-1)

                for i, j in T.Parallel(block_m, num_groups):
                    sf, sf_inv = get_sf_and_inv(
                        sf_inv_fragment[i, j], out_config, sf_inv_dtype
                    )
                    m_idx = pid_token * block_m + i
                    k_idx = pid_hidden * num_groups + j
                    store_sf(out_sf, sf, m_idx, k_idx, out_config)
                    sf_inv_fragment[i, j] = sf_inv

                if not sf_only:
                    if stochastic_cast:
                        x_fp32_local = T.alloc_local((4,), T.float32)
                        for i, j in T.Parallel(block_m, block_k // 4):
                            for k in T.vectorized(4):
                                factor = (
                                    x_sf_fragment[
                                        i // in_config.sf_block[0],
                                        (j * 4 + k) // in_config.sf_block[1],
                                    ]
                                    * sf_inv_fragment[
                                        i, (j * 4 + k) // num_per_channels
                                    ]
                                )
                                x_fp32_local[k] = (
                                    T.float32(x_fragment[i, j * 4 + k]) * factor
                                )
                            rbits = rbits_hash(
                                x_fp32_local, pid_hidden * (block_k // 4) + j
                            )
                            for k in T.vectorized(4):
                                out_fragment_or_shared[i, j * 4 + k] = T.cast(
                                    x_fp32_local[k],
                                    out_config.dtype,
                                    round="rs",
                                    rbits=rbits,
                                )
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            factor = (
                                x_sf_fragment[
                                    i // in_config.sf_block[0],
                                    j // in_config.sf_block[1],
                                ]
                                * sf_inv_fragment[i, j // num_per_channels]
                            )
                            out_fragment_or_shared[i, j] = (
                                T.float32(x_fragment[i, j]) * factor
                            )

            else:
                if cast_only:
                    for i, j in T.Parallel(block_m, num_groups):
                        sf = transform_sf(
                            load_sf(
                                out_sf,
                                pid_token * block_m + i,
                                pid_hidden * num_groups + j,
                                out_config,
                            ),
                            out_config,
                        )
                        sf_inv_fragment[i, j] = 1 / sf
                else:
                    amax_fragment = T.alloc_fragment(
                        (block_m, num_groups), in_config.dtype
                    )
                    x_fragment_reshaped = T.reshape(
                        x_fragment, [block_m, num_groups, num_per_channels]
                    )
                    T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)
                    for i, j in T.Parallel(block_m, num_groups):
                        amax = T.cast(amax_fragment[i, j], T.float32)
                        sf, sf_inv = get_sf_and_inv(amax, out_config, sf_inv_dtype)

                        m_idx = pid_token * block_m + i
                        k_idx = pid_hidden * num_groups + j
                        store_sf(out_sf, sf, m_idx, k_idx, out_config)
                        sf_inv_fragment[i, j] = sf_inv

                if not sf_only:
                    if stochastic_cast:
                        x_fp32_local = T.alloc_local((4,), T.float32)
                        for i, j in T.Parallel(block_m, block_k // 4):
                            for k in T.vectorized(4):
                                x_fp32_local[k] = (
                                    x_fragment[i, j * 4 + k]
                                    * sf_inv_fragment[
                                        i, (j * 4 + k) // num_per_channels
                                    ]
                                )
                            rbits = rbits_hash(
                                x_fp32_local, pid_hidden * (block_k // 4) + j
                            )
                            for k in T.vectorized(4):
                                out_fragment_or_shared[i, j * 4 + k] = T.cast(
                                    x_fp32_local[k],
                                    out_config.dtype,
                                    round="rs",
                                    rbits=rbits,
                                )
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            out_fragment_or_shared[i, j] = (
                                x_fragment[i, j]
                                * sf_inv_fragment[i, j // num_per_channels]
                            )

            if not sf_only:
                T.copy(
                    out_fragment_or_shared,
                    out[pid_token * block_m, pid_hidden * block_k],
                    disable_tma=True,
                )

            if use_pdl:
                T.pdl_trigger()

    return per_token_cast_kernel


def per_token_cast_impl(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    sf_only: bool = False,
    stochastic_cast: bool = False,
    sf: Optional[torch.Tensor] = None,
    out_x: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> Union[torch.Tensor, QuantTensor]:
    assert fmt in ("e4m3", "e2m1")

    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        use_e4m3_sf,
    )

    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_per_channels in (16, 32, 64, 128) and hidden % 64 == 0
    assert sf is None or out_sf is None
    assert sf is None or (
        not use_tma_aligned_col_major_sf and not use_packed_ue8m0
    ), "Cast-only mode only supports row-major, non-packed SF"

    kernel = get_per_token_cast_kernel_cuda(
        hidden=hidden,
        token_stride=get_logical_hidden(x.stride(0), x.dtype),
        in_config=in_config,
        out_config=out_config,
        sf_only=sf_only,
        cast_only=(sf is not None),
        small_batch=num_tokens <= SMALL_BATCH_THRESHOLD,
        stochastic_cast=stochastic_cast,
        use_pdl=use_pdl,
    )

    if out_x is None:
        out_x = torch.empty(
            (num_tokens, get_physical_hidden(hidden, out_config.torch_dtype)),
            dtype=out_config.torch_dtype,
            device=x.device,
        )
    if out_sf is None:
        out_sf = (
            sf
            if sf is not None
            else alloc_scaling_factors(
                (num_tokens, hidden),
                out_config=out_config,
                device=x.device,
            )
        )
    else:
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
        assert not use_tma_aligned_col_major_sf, "Not supported yet"
        if use_packed_ue8m0:
            out_sf = out_sf.view(dtype=torch.uint8)
        assert out_sf.shape == sf_shape

    if num_tokens > 0:
        kernel(x, x_sf, out_x, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf

    if sf is not None:
        return out_x

    return out_x, out_sf


def per_token_cast(
    x: Union[torch.Tensor, QuantTensor],
    fmt: str,
    num_per_channels: int,
    x_block_size: Optional[tuple[int, int]] = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    stochastic_cast: bool = False,
    out: Optional[QuantTensor] = None,
    *,
    use_pdl: bool = False,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling factors."""
    out_x, out_sf = (None, None) if out is None else out
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        x_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        use_e4m3_sf,
        stochastic_cast=stochastic_cast,
        out_x=out_x,
        out_sf=out_sf,
        use_pdl=use_pdl,
    )


def per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    out_sf: Optional[torch.Tensor] = None,
    *,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token scaling, returning only the sf factors."""
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        sf_only=True,
        out_sf=out_sf,
        use_pdl=use_pdl,
    )


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    use_e4m3_sf: bool = False,
    out: Optional[torch.Tensor] = None,
    *,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-token scaling using precomputed sf factors."""
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        use_e4m3_sf=use_e4m3_sf,
        sf=sf,
        out_x=out,
        use_pdl=use_pdl,
    )
