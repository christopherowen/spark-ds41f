
class _FakeRegistry:
    def __init__(self, calls):
        self._calls = calls
    def registrations(self):
        return [(tl_prepare.TILELANG, [((lambda: self._calls, None, None), {})])]

import json
import os
from pathlib import Path
import pytest
import torch
from torch import nn
import tilelang
import tilelang.cache as tl_cache
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.models.deepseek_v4_1.tilelang import prepare as tl_prepare
from vllm.models.deepseek_v4_1.tilelang import vocab


def _make_vocab_layer():
    layer = nn.Module()
    g = torch.Generator(device="cuda").manual_seed(7)
    w = (torch.randn(4096, 256, generator=g, device="cuda") * 0.05).bfloat16()
    packed = vocab.pack_bf16_weight(w)
    layer.weight = nn.Parameter(packed.values, requires_grad=False)
    layer.weight_exception_rows = packed.exception_rows
    layer.weight_exception_entries = packed.exception_entries
    layer.weight_row_offsets = packed.row_offsets
    layer.packed_bf16_exponent_base = packed.exponent_base
    layer.packed_bf16_in_features = 256
    layer.tilelang_max_rows = 32
    layer.tilelang_markov_head = True
    return layer


def _clear_in_memory(calls):
    kc, *_ = tl_cache._resolve_cache_dispatch("cuda", None, "tvm_ffi", False)
    kc._memory_cache.clear()
    for impl, _, _ in calls:
        impl._kernel_cache.clear()
        impl._prepared_kernel_cache.clear()


def test_warm_index_populates_loads_bit_exact_and_enforces_freeze(monkeypatch):
    layer = _make_vocab_layer()
    calls = list(vocab._packed_head_kernels(layer, torch.bfloat16).values())
    m_k = vocab._packed_head_markov_greedy_kernels(layer, torch.bfloat16)
    calls += list(m_k["partials"].values()) + list(m_k["finishes"].values())

    reg = _FakeRegistry(calls)

    # 1. Cold boot populates index on disk
    n = tl_prepare.prepare_declared(reg)
    assert n == len(calls)
    kc, *_ = tl_cache._resolve_cache_dispatch("cuda", None, "tvm_ffi", False)
    idx_files = list(Path(kc._get_cache_path("")).parent.glob("prepare-index-*.json"))
    assert len(idx_files) == 1
    data = json.loads(idx_files[0].read_text())
    assert len(data["entries"]) >= len(set((id(c[0]), repr(c[1]), repr(c[2])) for c in calls))

    # Execute normal-compiled kernel for reference
    x = (torch.randn(16, 256, device="cuda") * 0.25).bfloat16()
    out_ref = torch.empty((16, 4096), dtype=torch.bfloat16, device="cuda")
    k_ref = calls[0][0](*calls[0][1], **calls[0][2])
    k_ref(x, layer.weight, layer.weight_exception_rows, layer.weight_exception_entries, layer.weight_row_offsets, out_ref)
    src_ref = k_ref.get_kernel_source()

    # 2. Clear in-memory caches and assert tilelang.prepare is NOT called on warm hit!
    _clear_in_memory(calls)
    called_compile = []
    orig_prepare = tilelang.prepare
    def _fail_prepare(*a, **kw):
        called_compile.append(a)
        return orig_prepare(*a, **kw)
    monkeypatch.setattr(tilelang, "prepare", _fail_prepare)

    reg2 = _FakeRegistry(calls)
    tl_prepare.prepare_declared(reg2)
    assert called_compile == [], f"Warm index should have hit 100%, but compiled {len(called_compile)}"

    # 3. Freeze and verify prepared kernel executes bit-exact + undeclared kernel refuses!
    tl_prepare.freeze("test-freeze")
    try:
        out_fast = torch.empty_like(out_ref)
        k_fast = calls[0][0](*calls[0][1], **calls[0][2])
        k_fast(x, layer.weight, layer.weight_exception_rows, layer.weight_exception_entries, layer.weight_row_offsets, out_fast)
        torch.cuda.synchronize()
        assert torch.equal(out_ref, out_fast)
        assert k_fast.get_kernel_source() == src_ref

        # Undeclared shape MUST still raise KernelNotPreparedError under freeze!
        with pytest.raises(Exception):
            vocab.packed_vocab_projection(8192, 256, 120, block_M=16, out_dtype=tilelang.language.bfloat16)
    finally:
        tl_cache.thaw()


def test_warm_index_invalidates_on_env_change_and_self_heals_missing_so(monkeypatch):
    layer = _make_vocab_layer()
    calls = list(vocab._packed_head_kernels(layer, torch.bfloat16).values())[:1]
    kc, *_ = tl_cache._resolve_cache_dispatch("cuda", None, "tvm_ffi", False)

    # 1. Changing a VLLM_DS41_* env var changes provenance digest
    p1 = tl_prepare._prepare_index_provenance(kc._get_cache_path(""))
    monkeypatch.setenv("VLLM_DS41_TEST_PROVENANCE_SALT", "1")
    p2 = tl_prepare._prepare_index_provenance(kc._get_cache_path(""))
    assert p1 != p2
    monkeypatch.delenv("VLLM_DS41_TEST_PROVENANCE_SALT")

    # 2. Deleting cached .so on disk self-heals cleanly via tilelang.prepare fallback
    _clear_in_memory(calls)
    idx_files = list(Path(kc._get_cache_path("")).parent.glob("prepare-index-*.json"))
    data = json.loads(idx_files[0].read_text())
    sig = tl_prepare._call_index_sig(calls[0][0], calls[0][1], calls[0][2])
    ckey = data["entries"][sig]["cache_key"]
    so_path = Path(kc._get_cache_path(ckey)) / kc.kernel_lib_path
    assert so_path.is_file()
    so_path.unlink()

    reg = _FakeRegistry(calls)
    tl_prepare.prepare_declared(reg)
    data2 = json.loads(idx_files[0].read_text())
    ckey2 = data2["entries"][sig]["cache_key"]
    assert (Path(kc._get_cache_path(ckey2)) / kc.kernel_lib_path).is_file()  # Self-healed!


def test_warm_index_bypassed_when_cache_disabled(monkeypatch):
    layer = _make_vocab_layer()
    calls = list(vocab._packed_head_kernels(layer, torch.bfloat16).values())[:1]
    _clear_in_memory(calls)
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    called = []
    orig = tilelang.prepare
    def _track(*a, **kw):
        called.append(True)
        return orig(*a, **kw)
    monkeypatch.setattr(tilelang, "prepare", _track)
    reg = _FakeRegistry(calls)
    tl_prepare.prepare_declared(reg)
    assert called == [True]
