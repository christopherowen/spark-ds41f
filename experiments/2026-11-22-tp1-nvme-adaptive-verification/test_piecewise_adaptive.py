import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    resolve_adaptive_cudagraph_mode,
)
from vllm.models.deepseek_v4_1.expert_slots.cache import (
    ExpertSlotCache,
    RouterLookaheadPredictor,
)


def test_resolve_adaptive_cudagraph_mode_preserves_piecewise_and_none():
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.PIECEWISE, piecewise_capture_available=True
    ) == CUDAGraphMode.PIECEWISE
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.PIECEWISE, piecewise_capture_available=False
    ) == CUDAGraphMode.PIECEWISE
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.NONE, piecewise_capture_available=False
    ) == CUDAGraphMode.NONE
    # Unchanged for FULL modes
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.FULL_AND_PIECEWISE, piecewise_capture_available=True
    ) == CUDAGraphMode.FULL_AND_PIECEWISE
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.FULL, piecewise_capture_available=False
    ) == CUDAGraphMode.FULL_DECODE_ONLY
    assert resolve_adaptive_cudagraph_mode(
        CUDAGraphMode.FULL_DECODE_ONLY, piecewise_capture_available=True
    ) == CUDAGraphMode.FULL_DECODE_ONLY


def test_expert_slot_cache_lookahead_skips_dead_padding_rows():
    seen_rows = []

    class DummyPredictor:
        def predict_experts(self, x):
            seen_rows.append(x.detach().cpu().clone())
            return [3, 4]

    class DummyNextCache:
        def __init__(self):
            self.prefetched = []

        def prefetch_async(self, exps):
            self.prefetched.append(list(exps))

    class DummyPlan:
        def run(self, x, topk_ids, topk_weights, weights, out, **kwargs):
            return out

    class _StubCache(ExpertSlotCache):
        weights = None
    cache = object.__new__(_StubCache)
    cache.prefill_protect_rows = 24
    cache.expert_to_slot_gpu = None
    cache._partition_windows = lambda arr: [(0, arr.shape[0], [0, 1])]
    cache.ensure_active = lambda active, is_prefill=False: None

    x = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    # Rows 0,1,2 live; rows 3,4,5 dead/padding (-1)
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [0, 2], [-1, -1], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    topk_w = torch.ones((6, 2), dtype=torch.float32)
    next_cache = DummyNextCache()
    cache.run(
        DummyPlan(),
        x,
        topk_ids,
        topk_w,
        None,
        next_lookahead=(DummyPredictor(), next_cache),
        lookahead_hidden=x,
    )
    assert len(seen_rows) == 1
    assert torch.equal(seen_rows[0], x[:3])
    assert next_cache.prefetched == [[3, 4]]

    # All dead rows -> empty slice passed, no crash
    seen_rows.clear()
    all_dead = torch.full((4, 2), -1, dtype=torch.int32)
    cache.run(
        DummyPlan(),
        x[:4],
        all_dead,
        topk_w[:4],
        None,
        next_lookahead=(DummyPredictor(), next_cache),
        lookahead_hidden=x[:4],
    )
    assert seen_rows[0].shape == (0, 4)
