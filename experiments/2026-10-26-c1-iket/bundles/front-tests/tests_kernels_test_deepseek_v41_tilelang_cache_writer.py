# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang V4.1 cache records against a torch replica of B12X's arithmetic."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

from vllm.models.deepseek_v4_1.tilelang.cache_writer import RECORD_BYTES, write_cache

DEVICE = torch.device("cuda")
BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _records(kv: torch.Tensor, kind: str) -> torch.Tensor:
    """B12X's V4.1 records, one row per token, computed in torch."""
    group = 32 if kind == "swa" else 16
    x = kv.float().cpu().reshape(kv.shape[0], 512 // group, group)
    amax = x.abs().amax(dim=-1)
    if kind == "swa":
        scale = amax.clamp_min(1e-4) * torch.tensor(1.0 / 448.0, dtype=torch.float32)
        bits = scale.view(torch.int32)
        bumped = torch.where(
            (bits & 0x7FFFFF) != 0, (bits + 0x800000) & 0x7F800000, bits
        )
        exponent = (bumped >> 23) & 0xFF
        inverse = ((254 - exponent) << 23).view(torch.float32)
        payload = (x * inverse[..., None]).to(torch.float8_e4m3fn).view(torch.uint8)
        return torch.cat((payload.reshape(-1, 512), exponent.to(torch.uint8)), dim=-1)
    scale = amax.clamp_min(6 * 2**-9) / 6.0
    code = scale.clamp(max=448.0).to(torch.float8_e4m3fn)
    q = x / code.float()[..., None]
    magnitude = q.abs()
    codes = torch.bucketize(magnitude, BOUNDARIES)
    midpoint = BOUNDARIES[codes.clamp(max=6)]
    codes += (codes < 7) & (magnitude == midpoint) & ((codes & 1) != 0)
    codes = (codes.to(torch.uint8) | (torch.signbit(q).to(torch.uint8) << 3)).reshape(
        -1, 512
    )
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat((payload, code.view(torch.uint8)), dim=-1)


def _kv(tokens: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    kv = (
        torch.randn(tokens, 512, generator=gen) * torch.logspace(-3, 2, tokens)[:, None]
    )
    kv[0, :64] = 0.0  # a zero group: the scale floors
    kv[0, 64:128] = -0.0
    kv[1, :32] = 448.0 * 2.0**-3  # amax / 448 is a power of two
    kv[1, 32:48] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2) * 0.5
    kv[2, :] *= 1e4  # E4M3 scales saturate in the indexed records
    return kv.bfloat16()


@pytest.mark.parametrize("kind", ["swa", "indexed"])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("layout", ["flat", "padded", "records"])
def test_records_match_b12x(kind, slot_dtype, layout):
    page_size, pages, tokens = 64, 6, 200
    record = RECORD_BYTES[kind]
    kv = _kv(tokens, 3)
    gen = torch.Generator().manual_seed(4)
    slots = torch.randperm(pages * page_size, generator=gen)[:tokens].to(slot_dtype)
    slots[5] = -1  # a padded CUDA-graph slot is skipped
    slots[6] = pages * page_size  # beyond the pool: skipped
    pad = 48 if layout == "padded" else 0
    storage = torch.full((pages, page_size * record + pad), 0xA5, dtype=torch.uint8)
    cache = storage.to(DEVICE)
    view = cache
    if layout == "records":
        view = cache.view(pages, page_size, record)
    elif layout == "padded":
        view = cache[:, : page_size * record]
    write_cache(
        kv.to(DEVICE), view, slots.to(DEVICE), page_size=page_size, cache_kind=kind
    )
    expected = storage.clone()
    flat = expected[:, : page_size * record].reshape(-1, record)
    live = (slots >= 0) & (slots < pages * page_size)
    flat[slots[live].long()] = _records(kv[live], kind)
    expected[:, : page_size * record] = flat.reshape(pages, -1)
    assert torch.equal(cache.cpu(), expected)


def test_strided_rows():
    # The writer reads rows of a wider buffer (row stride above 512).
    page_size, kind = 64, "swa"
    wide = _kv(70, 5).repeat(1, 2).to(DEVICE)
    kv = wide[:, :512]
    slots = torch.arange(70, device=DEVICE)
    cache = torch.zeros(
        (2, page_size * RECORD_BYTES[kind]), dtype=torch.uint8, device=DEVICE
    )
    write_cache(kv, cache, slots, page_size=page_size, cache_kind=kind)
    got = cache.view(-1, RECORD_BYTES[kind])[:70].cpu()
    assert torch.equal(got, _records(kv.cpu(), kind))
