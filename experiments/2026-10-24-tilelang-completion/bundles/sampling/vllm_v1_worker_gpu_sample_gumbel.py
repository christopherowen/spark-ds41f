# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10

# Offset salt keeping the draft's Gumbel noise disjoint from the target's.
# Verification is a probability-ratio test, not a Gumbel coupling, so a proposal
# and the residual it is resampled from must not share a noise vector.
# Positions are int64 and never approach 2**30, so the streams cannot collide.
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30) if HAS_TRITON else (1 << 30)


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    tilelang_sampling.apply_temperature(logits, expanded_idx_mapping, temperature)


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def spec_step_seed(seed, step):
    """Mix a speculative step into the high bits of a request seed."""
    return seed ^ (tl.cast(step, tl.int64) << 32)


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree.
    """
    logits = gumbel_noised_scores(
        logits,
        keys,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )
    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_noised_scores(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """The scores `gumbel_noised_argmax` takes the argmax of: the logits plus
    the draw's Gumbel noise, or the logits themselves at temp 0. In descending
    order they are sampling without replacement from the same draw."""
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # log1p keeps the winning tail at u -> 0, where fp32 resolves it.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))
    return logits


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    vocab_size,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if logits_cache_ptr is not None:
        # Store the logits *before* temperature. Dividing first would produce a
        # value that is generally not representable in the cache's dtype, forcing
        # it to be fp32. Consumers (the rejection sampler) divide by the same
        # temperature on load, which reproduces the value used below bitwise.
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + token_idx)
    if logits_cache_ptr is not None:
        seed = spec_step_seed(seed, col)
    return gumbel_noised_argmax(
        logits,
        block,
        mask,
        seed,
        pos,
        temp,
        IS_DRAFTING=IS_DRAFTING,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
    )


# Reference for ``tilelang_sampling``; the runner no longer launches it.
@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        logits_cache_ptr,
        logits_cache_stride_0,
        logits_cache_stride_1,
        logits_cache_col_ptr,
        vocab_size,
        IS_DRAFTING=IS_DRAFTING,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def register_sampling_warmups(
    *,
    logits: Sequence[tuple[torch.dtype, int]],
    processed_logits_strides: Sequence[int],
    vocab_size: int,
    use_fp64: bool,
) -> None:
    """Declare the target sampler's temperature scaling of processed FP32 logits
    and its Gumbel-max sampling without a logits cache of logits of each
    ``(dtype, row stride)``; the TileLang kernels take any stride."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    if use_fp64:
        raise ValueError(
            "use_fp64_gumbel: the V2 runner's TileLang sampler draws in FP32 only"
        )
    tilelang_sampling.declare(
        lambda: tilelang_sampling.sampling_calls(
            [dtype for dtype, _ in logits], temperature=bool(processed_logits_strides)
        )
    )


def register_drafting_warmups(
    *,
    logits_dtypes: Sequence[torch.dtype],
    cache_dtype: torch.dtype,
    use_fp64: bool,
) -> None:
    """Declare a speculator's probabilistic draft sampling: the temperature in the
    kernel, the drafting noise stream and the logits cache of ``cache_dtype``, for
    logits of each of ``logits_dtypes`` and a step column of either index type,
    one per token or shared."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    if use_fp64:
        raise ValueError(
            "use_fp64_gumbel: the V2 runner's TileLang sampler draws in FP32 only"
        )
    tilelang_sampling.declare(
        lambda: [
            call
            for dtype in dict.fromkeys(logits_dtypes)
            for col_dtype in (torch.int32, torch.int64)
            for per_token_col in (False, True)
            for call in tilelang_sampling.drafting_calls(
                dtype, cache_dtype, col_dtype, per_token_col
            )
        ]
    )


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
) -> torch.Tensor:
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    return tilelang_sampling.gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        is_drafting,
        logits_cache,
        logits_cache_col,
        use_fp64,
    )


def warmup_processed_gumbel(
    vocab_size: int, device: torch.device, *, use_fp64: bool = False
) -> None:
    """Compile target sampling of processed FP32 logits without a model forward.

    FlashInfer handles unseeded filtered warmup requests, but explicit seeds
    dispatch to Gumbel-max. Greedy warmup covers only native-dtype logits and
    therefore cannot initialize this specialization. Temporary inputs do not
    consume a serving request's RNG stream or modify request state.
    """
    logits = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
    mapping = torch.zeros(1, dtype=torch.int64, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    seed = torch.zeros(1, dtype=torch.int64, device=device)
    position = torch.zeros(1, dtype=torch.int64, device=device)
    gumbel_sample(
        logits,
        mapping,
        temperature,
        seed,
        position,
        apply_temperature=False,
        is_drafting=False,
        use_fp64=use_fp64,
    )
