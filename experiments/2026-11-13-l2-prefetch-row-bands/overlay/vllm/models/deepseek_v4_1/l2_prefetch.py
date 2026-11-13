# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for DeepSeek V4.1 decode on GB10.

Decode projections stream their weights from LPDDR5X, while each layer also
runs serial latency-bound work (sparse-MLA preparation, TP all-reduces, mHC,
routing) that leaves device memory idle. GB10 has a 24 MB L2, enough for one
layer's pair of attention projections. Each window issues bulk L2 prefetches
for the next projections on a side stream (the GLM-5.3 prefetcher), so those
GEMMs read L2 instead of DRAM. Cache hints only; numerics are unchanged.

At a fixed eight-row verifier, TP4 decode cycles fall from 49.36 to 47.45 ms
(4 prefetch CTAs). Issuing FFN/NEXT after their all-reduces loses the window
(49.13 ms); WO alone is nearly neutral (49.12 ms) because the fill slows the
latency-bound sparse-MLA preparation kernels it overlaps.

Windows per decoder layer, in fill order on the one prefetch stream (a window
issued while the one before it still fills starts when that one ends):
  WO    after the query up-projection: this layer's WO-A and WO-B. A layer that
        owns a compressor (and an indexer) issues it right after the compressor
        instead: issued after Q-B, its lines did not survive the compressor's
        projection, and WO-A ran at DRAM speed with its fill long done.
  COMP  an owner layer's compressor projection, after Q-B in WO's place. (The
        indexer's projections after it gained nothing in this window: the WO
        fill behind it evicted them before they ran.)
  FFN   before the attention all-reduce: router, mHC FFN mix, shared expert
  NEXT  before the MoE all-reduce: next layer's mHC attention mix, fused
        Q-A/KV and Q-B projections, and indexer projections. An Engram layer
        runs its projection before anything NEXT holds, and NEXT's lines did not
        survive it: that layer issues NEXT between its Engram projection's GEMM
        and all-gather instead.
  ENGRAM  before the MoE all-reduce ahead of an Engram layer, in NEXT's place:
        the first bytes of its Engram projection, over the all-reduce, mHC post
        and row wait.

A window issued before the lines of the one ahead of it are read evicts those
unread lines: the consumers' reads keep the lines they read, so the unread ones
are the oldest. Issuing WO right after the fused Q-A/KV or FFN between WO-A and
WO-B made forty TP4 decode layers 8-10% slower on one GB10; the owner and Engram
points made them 1.6% faster at one stream, 0.7% at eight and 0.3% at sixteen,
spin kernels standing in for the attention core and all-reduces
(experiments/2026-10-26-l2-prefetch-schedule in spark-ds41f).

A window can take a different budget in a band of row counts. WO (and COMP, which
takes WO's budget) fills while the sparse-MLA preparation kernels run, and what the
fill costs them grows with the rows while what it saves the projections does not: at
TP4 on four GB10s, 24 and 32 rows (the 32-row decode tile tier) were 2.4-2.7% and
1.5% faster with 10 MB than with 15 MB, and 1x2 rows 1.9% slower
(experiments/2026-11-13-l2-prefetch-row-bands in spark-ds41f). The forward issues the
plan its row count selects, so a captured graph holds its own row count's plan.

Environment:
  VLLM_DS41_L2_PREFETCH=0/1           default on for SM121 (GB10)
  VLLM_DS41_L2_PREFETCH_{WO,FFN,NEXT}_MB   per-window fill budgets
  VLLM_DS41_L2_PREFETCH_WO_BANDS      WO and COMP budgets by row count, as
                                      first-last=megabytes[,...]: "17-32=10"
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.models.glm5next.nvidia import l2_prefetch as _l2pf

logger = init_logger(__name__)

Segment = _l2pf.Segment


def _enabled() -> bool:
    if not torch.cuda.is_available():
        return False
    raw = os.getenv("VLLM_DS41_L2_PREFETCH")
    if raw is not None:
        return raw != "0"
    return torch.cuda.get_device_capability() == (12, 1)


ENABLED = _enabled()
BUDGET_WO = _l2pf._mb("VLLM_DS41_L2_PREFETCH_WO_MB", "20")
BUDGET_FFN = _l2pf._mb("VLLM_DS41_L2_PREFETCH_FFN_MB", "16")
BUDGET_NEXT = _l2pf._mb("VLLM_DS41_L2_PREFETCH_NEXT_MB", "20")


def parse_row_bands(raw: str | None) -> tuple[tuple[int, int, int], ...]:
    """``"17-32=10,65-96=12.5"`` as ``(first row count, last, budget in bytes)``
    bands, in row order. A band may name one row count (``"24=10"``); megabytes are
    decimal, as the window budgets are. Bands may not overlap."""
    bands = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            rows, megabytes = item.split("=")
            first, _, last = rows.partition("-")
            band = (int(first), int(last or first), int(float(megabytes) * 1e6))
        except ValueError:
            raise ValueError(
                f"L2 prefetch row band {item!r}: expected first-last=megabytes"
            ) from None
        if not 1 <= band[0] <= band[1] or band[2] < 0:
            raise ValueError(
                f"L2 prefetch row band {item!r}: rows or budget out of range"
            )
        bands.append(band)
    bands.sort()
    for (_, last, _), (first, _, _) in zip(bands, bands[1:]):
        if first <= last:
            raise ValueError(f"L2 prefetch row bands overlap at {first} rows: {raw!r}")
    return tuple(bands)


# A malformed value stops the worker at import: a band that silently did not apply
# would look like a budget that does nothing.
WO_BANDS = parse_row_bands(os.getenv("VLLM_DS41_L2_PREFETCH_WO_BANDS"))


class RowBandPlans:
    """One window's plans by the forward's row count: ``default`` outside every band."""

    __slots__ = ("default", "bands")

    def __init__(self, default, bands):
        self.default = default
        self.bands = tuple(bands)

    def select(self, rows: int):
        for first, last, plan in self.bands:
            if first <= rows <= last:
                return plan
        return self.default

    def describe(self) -> str:
        text = self.default.describe() if self.default is not None else "-"
        for first, last, plan in self.bands:
            size = f"{plan.total_bytes / 1e6:.1f} MB" if plan is not None else "-"
            text += f" [{first}-{last} rows: {size}]"
        return text


def linear_segments(name: str, linear: torch.nn.Module | None) -> list[Segment]:
    """Bytes read by a V4.1 linear at decode: its weight (or TP decode shard), and
    its scale words."""
    if linear is None:
        return []
    weight = getattr(linear, "shard_weight", getattr(linear, "weight", None))
    weight_sf = getattr(
        linear, "shard_weight_sf", getattr(linear, "tilelang_weight_sf", None)
    )
    segments = [
        _l2pf.tensor_segment(f"{name}.weight", weight),
        # Scale bytes beside the weight: packed 32-row block words, or WO-A's
        # raw 32x32 exponent grid. The GEMM indexes either with row // 32.
        _l2pf.tensor_segment(
            f"{name}.tilelang_weight_sf",
            weight_sf,
            min_bytes=0,
        ),
    ]
    return [segment for segment in segments if segment is not None]


def param_segments(name: str, owner: object, attrs: tuple[str, ...]) -> list[Segment]:
    out = []
    for attr in attrs:
        segment = _l2pf.tensor_segment(
            f"{name}.{attr}", getattr(owner, attr, None), min_bytes=0
        )
        if segment is not None:
            out.append(segment)
    return out


def make_plan(segments: list[Segment], budget: int, device: torch.device):
    seen: set[int] = set()
    unique = []
    for segment in segments:
        if segment[1] not in seen:
            seen.add(segment[1])
            unique.append(segment)
    plan, _ = _l2pf.make_plan(unique, budget, device)
    return plan


def make_wo_plan(segments: list[Segment], device: torch.device):
    """The WO or COMP window's plan: ``BUDGET_WO``, and each row band's budget where
    ``VLLM_DS41_L2_PREFETCH_WO_BANDS`` names one. Without bands this is ``make_plan``'s
    plan itself."""
    default = make_plan(segments, BUDGET_WO, device)
    if not WO_BANDS:
        return default
    return RowBandPlans(
        default,
        [
            (first, last, make_plan(segments, budget, device))
            for first, last, budget in WO_BANDS
        ],
    )


def warmup() -> bool:
    return ENABLED and _l2pf._get_launcher() is not None


def kernel_calls() -> list:
    """The TileLang prefetch kernel's call, as ``warmup`` makes it, when the windows
    prefetch through it (``VLLM_L2_PREFETCH_KERNELS=tilelang``)."""
    if not ENABLED or _l2pf._KERNELS != "tilelang":
        return []
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel

    return [
        prepare.call(l2_prefetch_kernel, _l2pf._GRID, _l2pf._BLOCK, _l2pf._CHUNK_BYTES)
    ]


def issue(plan: _l2pf.L2PrefetchPlan | RowBandPlans | None, num_tokens: int) -> None:
    if isinstance(plan, RowBandPlans):
        # Chosen as the window is issued: under capture, once for the graph's rows.
        plan = plan.select(num_tokens)
    if ENABLED and plan is not None and plan.segs is not None:
        _l2pf.L2Prefetcher.get(plan.segs.device).issue(plan, num_tokens)


def join() -> None:
    if not ENABLED:
        return
    for prefetcher in list(_l2pf.L2Prefetcher._instances.values()):
        prefetcher.join()
