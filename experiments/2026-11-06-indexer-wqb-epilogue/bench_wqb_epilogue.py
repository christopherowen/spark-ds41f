import argparse
import json
import time
from types import SimpleNamespace
import torch
from torch import nn

import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1 import kernels
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.tilelang import pdl, prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_ROWS,
    DECODE_TILE_ROWS,
    SF_BLOCK,
    SF_WORD_K,
    TILE_CONTIGUOUS_FP8,
    TILE_WEIGHT_K,
    TILE_WEIGHT_N,
    _e2m1_code,
    block_scale_words,
    decode_tile_rows,
    fp8_decode_config,
    mxfp8_gemm_decode,
    mxfp8_gemm_rope_mxfp4_decode,
    pack_scale_words,
    pack_weight_tiles,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4
from vllm.models.deepseek_v4_1.tilelang.linear import (
    MXFP8Rows,
    TileLangBlock32Rows,
    TileLangScaledLinearMethod,
    cast_mxfp8,
    project_block32_rope_mxfp4_decode,
    provide_mxfp8,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.worker.workspace import use_preallocated_workspace
from tests.kernels.deepseek_v41_pdl_sources import violations as pdl_violations


def _build_832_c_shared_decode(
    N: int,
    K: int,
    head_dim: int = 128,
    rope_dim: int = 64,
    block_M: int = 16,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 3,
    threads: int = 128,
    padded_rows: bool = True,
    tile_contiguous: bool = True,
):
    """#832 Component 3's 1-thread-per-32-col-group C_shared epilogue (for comparison)."""
    k_blocks = K // block_K
    M = T.dynamic('M')
    Mp = T.dynamic('Mp') if padded_rows else M
    max_pos = T.dynamic('max_pos')
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)
    k_tiles = K // TILE_WEIGHT_K
    b_shape = (N * k_tiles, TILE_WEIGHT_K) if tile_contiguous else (N, K)
    half_n = N // 2
    scale_n = N // SF_BLOCK
    rope_start = head_dim - rope_dim
    rope_half = rope_dim // 2
    groups_per_tile = block_N // SF_BLOCK
    half = SF_BLOCK // 2

    @T.prim_func
    def gemm_mxfp8_rope_mxfp4(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor(b_shape, in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N // 32, sf_words), T.uint32),
        Positions: T.Tensor((M,), T.int64),
        CosSin: T.Tensor((max_pos, rope_dim), T.float32),
        Q: T.Tensor((M, half_n), T.uint8),
        S: T.Tensor((M, scale_n), T.uint8),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[
                        T.min(by * block_M + i, M - 1),
                        ko * block_K + k,
                    ]
                row0 = (
                    ((bx * block_N // TILE_WEIGHT_N) * k_tiles + ko) * TILE_WEIGHT_N
                    + (bx * block_N % TILE_WEIGHT_N)
                )
                T.copy(B[row0, 0], B_shared)
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    row = T.min(bx * block_N + j, N - 1)
                    SFB_shared[j, w] = SFB[
                        row // 32,
                        T.min(word0 + w, sf_words - 1),
                    ]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=ko * block_K % SF_WORD_K,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )

            T.sync_threads()
            T.copy(C_local, C_shared)
            for idx in T.Parallel(block_M * groups_per_tile):
                row_local = idx // groups_per_tile
                g = idx % groups_per_tile
                m = by * block_M + row_local
                if m < M:
                    col0 = bx * block_N + g * SF_BLOCK
                    in_head = col0 % head_dim
                    vals = T.alloc_local((SF_BLOCK,), T.bfloat16)
                    for c in T.serial(SF_BLOCK):
                        vals[c] = C_shared[row_local, g * SF_BLOCK + c]
                    if in_head >= rope_start:
                        pos = Positions[m]
                        pair_base = (in_head - rope_start) // 2
                        for i in T.serial(half):
                            first = T.cast(vals[2 * i], T.float32)
                            second = T.cast(vals[2 * i + 1], T.float32)
                            cosine = CosSin[pos, pair_base + i]
                            sine = CosSin[pos, rope_half + pair_base + i]
                            vals[2 * i] = T.cast(
                                T.ieee_fmaf(first, cosine, -(second * sine)),
                                T.bfloat16,
                            )
                            vals[2 * i + 1] = T.cast(
                                T.ieee_fmaf(first, sine, second * cosine),
                                T.bfloat16,
                            )
                    amax = T.alloc_var(T.float32)
                    amax = 6.0 * 2.0**-126
                    for c in T.serial(SF_BLOCK):
                        amax = T.max(amax, T.abs(T.cast(vals[c], T.float32)))
                    bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                    sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                        (bits & 0x7FFFFF) != 0, T.int32
                    )
                    scale = T.reinterpret(
                        T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                    )
                    q_base = col0 // 2
                    for p in T.serial(half):
                        lo = _e2m1_code(T.fmul(T.cast(vals[2 * p], T.float32), scale))
                        hi = _e2m1_code(
                            T.fmul(T.cast(vals[2 * p + 1], T.float32), scale)
                        )
                        Q[m, q_base + p] = T.cast(lo | (hi << 4), T.uint8)
                    S[m, col0 // SF_BLOCK] = T.cast(sf, T.uint8)

    return gemm_mxfp8_rope_mxfp4


mxfp8_832_c_shared_decode = tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
)(_build_832_c_shared_decode)


def make_block32_weight(n: int, k: int, seed: int, device: str = 'cuda'):
    gen = torch.Generator(device=device).manual_seed(seed)
    w_bf16 = torch.randn((n, k), generator=gen, device=device, dtype=torch.bfloat16)
    w_fp8 = w_bf16.to(torch.float8_e4m3fn)
    exps = torch.randint(
        120, 132, (n // 32, k // 32), generator=gen, device=device, dtype=torch.uint8
    )
    return w_fp8, exps


def measure_us(fn, iters=200, warmup=30, wash_buf=None):
    for _ in range(warmup):
        if wash_buf is not None:
            wash_buf.add_(1)
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if wash_buf is not None:
            wash_buf.add_(1)
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    n = len(times)
    med = times[n // 2]
    iqr = times[(3 * n) // 4] - times[n // 4]
    return round(med, 2), round(iqr, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boot', type=int, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda'
    N, K = 4096, 1280
    heads, head_dim, rope_dim = 32, 128, 64
    hidden_size = 5120
    batch_sizes = [1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 64]
    bench_sizes = [1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48]

    # Build wq_b and weights_proj layers
    w_fp8, exps = make_block32_weight(N, K, 101, device=device)
    base_layer = nn.Module()
    base_layer.weight, base_layer.weight_scale_inv = w_fp8, exps

    wqb_fused = TileLangBlock32Rows(base_layer, 0)
    wqb_fused.tilelang_mxfp8_input = True
    wqb_fused.tilelang_rope_mxfp4 = (head_dim, rope_dim)

    wqb_unfused = TileLangBlock32Rows(base_layer, 0)
    wqb_unfused.tilelang_mxfp8_input = True

    gen = torch.Generator(device=device).manual_seed(202)
    w_proj_layer = nn.Module()
    w_proj_layer.weight = nn.Parameter(
        torch.randn((heads, hidden_size), generator=gen, device=device, dtype=torch.bfloat16),
        requires_grad=False,
    )
    w_proj_method = TileLangScaledLinearMethod((heads * head_dim) ** -0.5)
    w_proj_method.process_weights_after_loading(w_proj_layer)

    with set_current_vllm_config(VllmConfig()):
        wqb_fused.declare_kernels()
        wqb_unfused.declare_kernels()
        w_proj_method.declare_layer_kernels(w_proj_layer)
        # Also warm quantize_mxfp4
        dummy_iq = torch.zeros((1, heads, head_dim), dtype=torch.bfloat16, device=device)
        dummy_q = torch.empty((1, heads, head_dim // 2), dtype=torch.uint8, device=device)
        dummy_s = torch.empty((1, heads, head_dim // 32), dtype=torch.uint8, device=device)
        dummy_pos = torch.zeros((1,), dtype=torch.int64, device=device)
        dummy_cs = torch.zeros((256, rope_dim), dtype=torch.float32, device=device)
        quantize_mxfp4(dummy_iq, dummy_q, dummy_s, positions=dummy_pos, cos_sin=dummy_cs)

    # Compile #832 c_shared kernels for comparison
    kernels_832 = {}
    for bm in DECODE_TILE_ROWS:
        cfg = fp8_decode_config(N, K, bm)
        kernels_832[bm] = mxfp8_832_c_shared_decode(
            N, K, head_dim=head_dim, rope_dim=rope_dim, **cfg, padded_rows=True, tile_contiguous=True
        )

    # Check PDL source violations on our fused kernels
    src = wqb_fused.tilelang_decode_rope_mxfp4.get_kernel_source()
    v = pdl_violations(src, warp_specialized=False)
    assert not v, f'PDL violation in decode_rope_mxfp4: {v}'
    pdl_checks = {'16x128_s2_t256': v}

    # Construct mock attention objects for end-to-end _index_query calls
    cos_sin = torch.randn((4096, rope_dim), generator=gen, device=device, dtype=torch.float32)

    def make_attn_stub(wqb_obj):
        stub = SimpleNamespace()
        stub.indexer = SimpleNamespace(
            heads=heads,
            wq_b=wqb_obj,
            weights_proj=lambda h: w_proj_method.apply(w_proj_layer, h),
        )
        stub.rotary_emb = SimpleNamespace(cos_sin_cache=cos_sin)
        # Put _quantize_index_query on class, not instance __dict__, matching DeepseekV41TileLangAttention
        class StubCls:
            def _quantize_index_query(self, mode, iq, iq_data, iq_scale, positions=None):
                quantize_mxfp4(iq, iq_data, iq_scale, positions=positions, cos_sin=cos_sin)
            _index_query = DeepseekV4Attention._index_query
        s = StubCls()
        s.indexer = stub.indexer
        s.rotary_emb = stub.rotary_emb
        return s

    attn_fused = make_attn_stub(wqb_fused)
    attn_unfused = make_attn_stub(wqb_unfused)
    swa_decode = SimpleNamespace(is_decode=True)

    qr_full = torch.randn((64, K), generator=gen, device=device, dtype=torch.bfloat16)
    hidden_full = torch.randn((64, hidden_size), generator=gen, device=device, dtype=torch.bfloat16)
    pos_full = torch.randint(0, 4096, (64,), generator=gen, device=device, dtype=torch.int64)
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=device)

    bit_exact_results = {}
    with set_current_vllm_config(VllmConfig()), use_preallocated_workspace(scratch):
        for rows in batch_sizes:
            qr = qr_full[:rows]
            hidden = hidden_full[:rows]
            pos = pos_full[:rows]
            q_cast = cast_mxfp8(qr)
            if rows < q_cast.values.shape[0]:
                q_cast.values[rows:].fill_(float('nan'))
                q_cast.scales[rows:].fill_(0xFF)

            # 1. Eager bit-exactness
            iq_d_ref, iq_s_ref, w_ref = attn_unfused._index_query(
                qr, pos, hidden, None, 0, rows, swa_decode, q_cast
            )
            iq_d_fus, iq_s_fus, w_fus = attn_fused._index_query(
                qr, pos, hidden, None, 0, rows, swa_decode, q_cast
            )
            eager_q_eq = bool(torch.equal(iq_d_fus, iq_d_ref))
            eager_s_eq = bool(torch.equal(iq_s_fus, iq_s_ref))
            eager_w_eq = bool(torch.equal(w_fus, w_ref))
            assert eager_q_eq and eager_s_eq and eager_w_eq, f'Eager mismatch at rows={rows}'

            # 2. CUDAGraph capture & replay bit-exactness
            g_ref = torch.cuda.CUDAGraph()
            g_fus = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_ref):
                gd_ref, gs_ref, gw_ref = attn_unfused._index_query(
                    qr, pos, hidden, None, 0, rows, swa_decode, q_cast
                )
            with torch.cuda.graph(g_fus):
                gd_fus, gs_fus, gw_fus = attn_fused._index_query(
                    qr, pos, hidden, None, 0, rows, swa_decode, q_cast
                )
            gd_ref.zero_(); gs_ref.zero_(); gw_ref.zero_()
            gd_fus.zero_(); gs_fus.zero_(); gw_fus.zero_()
            g_ref.replay()
            g_fus.replay()
            torch.cuda.synchronize()
            cg_q_eq = bool(torch.equal(gd_fus, iq_d_ref) and torch.equal(gd_fus, gd_ref))
            cg_s_eq = bool(torch.equal(gs_fus, iq_s_ref) and torch.equal(gs_fus, gs_ref))
            cg_w_eq = bool(torch.equal(gw_fus, w_ref) and torch.equal(gw_fus, gw_ref))
            assert cg_q_eq and cg_s_eq and cg_w_eq, f'CUDAGraph mismatch at rows={rows}'

            # Also verify #832 c_shared produces identical output
            q_832 = torch.empty_like(iq_d_ref)
            s_832 = torch.empty_like(iq_s_ref)
            bm = decode_tile_rows(rows, {})
            kernels_832[bm](
                q_cast.values,
                wqb_fused.tilelang_weight,
                q_cast.scales.view(torch.uint32),
                wqb_fused.tilelang_weight_sf,
                pos,
                cos_sin,
                q_832.view(rows, -1),
                s_832.view(rows, -1),
            )
            assert torch.equal(q_832, iq_d_ref) and torch.equal(s_832, iq_s_ref)

            bit_exact_results[str(rows)] = {
                'eager_iq_data_equal': eager_q_eq,
                'eager_iq_scale_equal': eager_s_eq,
                'eager_weights_equal': eager_w_eq,
                'cudagraph_iq_data_equal': cg_q_eq,
                'cudagraph_iq_scale_equal': cg_s_eq,
                'cudagraph_weights_equal': cg_w_eq,
            }

        # Verify ratio=2 bit-exactness
        k_ratio2 = mxfp8_gemm_rope_mxfp4_decode(
            N, K, head_dim=head_dim, rope_dim=rope_dim,
            **fp8_decode_config(N, K, 16), padded_rows=True, tile_contiguous=True, ratio=2
        )
        qr16 = qr_full[:16]
        pos16 = pos_full[:16]
        qc16 = cast_mxfp8(qr16)
        with provide_mxfp8(qc16):
            iq16_ref = wqb_unfused(qr16).view(16, heads, head_dim)
        q16_ref = torch.empty((16, heads, head_dim // 2), dtype=torch.uint8, device=device)
        s16_ref = torch.empty((16, heads, head_dim // 32), dtype=torch.uint8, device=device)
        quantize_mxfp4(iq16_ref, q16_ref, s16_ref, positions=pos16, cos_sin=cos_sin, ratio=2)
        q16_out = torch.empty_like(q16_ref)
        s16_out = torch.empty_like(s16_ref)
        k_ratio2(
            qc16.values, wqb_fused.tilelang_weight, qc16.scales.view(torch.uint32),
            wqb_fused.tilelang_weight_sf, pos16, cos_sin, q16_out.view(16, -1), s16_out.view(16, -1)
        )
        assert torch.equal(q16_out, q16_ref) and torch.equal(s16_out, s16_ref)

    print(f'[Boot {args.boot}] All bit-exactness checks PASSED across {len(batch_sizes)} row counts (eager + CUDAGraph + ratio=2 + PDL)!')

    # 3. Latency Benchmarks
    wash_buf = torch.empty((128 * 1024 * 1024 // 4,), dtype=torch.int32, device=device)
    bench_results = {}

    with set_current_vllm_config(VllmConfig()), use_preallocated_workspace(scratch):
        for rows in bench_sizes:
            qr = qr_full[:rows]
            hidden = hidden_full[:rows]
            pos = pos_full[:rows]
            q_cast = cast_mxfp8(qr)
            bm = decode_tile_rows(rows, {})

            iq_buf = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
            q_buf = torch.empty((rows, heads, head_dim // 2), dtype=torch.uint8, device=device)
            s_buf = torch.empty((rows, heads, head_dim // 32), dtype=torch.uint8, device=device)

            gemm_unfused = wqb_unfused.tilelang_decode[bm]
            gemm_832 = kernels_832[bm]
            gemm_shfl16 = wqb_fused.tilelang_decode_rope_mxfp4
            sf_u32 = q_cast.scales.view(torch.uint32)
            w_t = wqb_fused.tilelang_weight
            wsf_t = wqb_fused.tilelang_weight_sf

            def run_standalone_unfused():
                gemm_unfused(q_cast.values, w_t, sf_u32, wsf_t, iq_buf.view(rows, -1))
                quantize_mxfp4(iq_buf, q_buf, s_buf, positions=pos, cos_sin=cos_sin)

            def run_standalone_832():
                gemm_832(q_cast.values, w_t, sf_u32, wsf_t, pos, cos_sin, q_buf.view(rows, -1), s_buf.view(rows, -1))

            def run_standalone_shfl16():
                gemm_shfl16(q_cast.values, w_t, sf_u32, wsf_t, pos, cos_sin, q_buf.view(rows, -1), s_buf.view(rows, -1))

            # Warm L2 standalone
            uf_warm_med, uf_warm_iqr = measure_us(run_standalone_unfused, iters=200, warmup=30)
            c832_warm_med, c832_warm_iqr = measure_us(run_standalone_832, iters=200, warmup=30)
            shfl_warm_med, shfl_warm_iqr = measure_us(run_standalone_shfl16, iters=200, warmup=30)

            # Cold 128MiB wash standalone
            uf_cold_med, uf_cold_iqr = measure_us(run_standalone_unfused, iters=60, warmup=10, wash_buf=wash_buf)
            c832_cold_med, c832_cold_iqr = measure_us(run_standalone_832, iters=60, warmup=10, wash_buf=wash_buf)
            shfl_cold_med, shfl_cold_iqr = measure_us(run_standalone_shfl16, iters=60, warmup=10, wash_buf=wash_buf)

            # End-to-end _index_query (Warm L2, Cold 128MiB wash, and CUDAGraph replay)
            def run_e2e_unfused():
                return attn_unfused._index_query(qr, pos, hidden, None, 0, rows, swa_decode, q_cast)

            def run_e2e_fused():
                return attn_fused._index_query(qr, pos, hidden, None, 0, rows, swa_decode, q_cast)

            e2e_uf_warm_med, e2e_uf_warm_iqr = measure_us(run_e2e_unfused, iters=200, warmup=30)
            e2e_fus_warm_med, e2e_fus_warm_iqr = measure_us(run_e2e_fused, iters=200, warmup=30)

            e2e_uf_cold_med, e2e_uf_cold_iqr = measure_us(run_e2e_unfused, iters=60, warmup=10, wash_buf=wash_buf)
            e2e_fus_cold_med, e2e_fus_cold_iqr = measure_us(run_e2e_fused, iters=60, warmup=10, wash_buf=wash_buf)

            g_e2e_uf = torch.cuda.CUDAGraph()
            g_e2e_fus = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_e2e_uf):
                run_e2e_unfused()
            with torch.cuda.graph(g_e2e_fus):
                run_e2e_fused()
            e2e_uf_cg_med, e2e_uf_cg_iqr = measure_us(g_e2e_uf.replay, iters=200, warmup=30)
            e2e_fus_cg_med, e2e_fus_cg_iqr = measure_us(g_e2e_fus.replay, iters=200, warmup=30)

            bench_results[str(rows)] = {
                'tile_block_M': bm,
                'standalone_warm_us': {
                    'unfused_med': uf_warm_med, 'unfused_iqr': uf_warm_iqr,
                    'c_shared_832_med': c832_warm_med, 'c_shared_832_iqr': c832_warm_iqr,
                    'warp_shfl16_med': shfl_warm_med, 'warp_shfl16_iqr': shfl_warm_iqr,
                    'delta_vs_unfused_us': round(shfl_warm_med - uf_warm_med, 2),
                    'delta_vs_832_us': round(shfl_warm_med - c832_warm_med, 2),
                },
                'standalone_cold_128MiB_us': {
                    'unfused_med': uf_cold_med, 'unfused_iqr': uf_cold_iqr,
                    'c_shared_832_med': c832_cold_med, 'c_shared_832_iqr': c832_cold_iqr,
                    'warp_shfl16_med': shfl_cold_med, 'warp_shfl16_iqr': shfl_cold_iqr,
                    'delta_vs_unfused_us': round(shfl_cold_med - uf_cold_med, 2),
                    'delta_vs_832_us': round(shfl_cold_med - c832_cold_med, 2),
                },
                'e2e_index_query_warm_us': {
                    'unfused_med': e2e_uf_warm_med, 'unfused_iqr': e2e_uf_warm_iqr,
                    'fused_med': e2e_fus_warm_med, 'fused_iqr': e2e_fus_warm_iqr,
                    'delta_us': round(e2e_fus_warm_med - e2e_uf_warm_med, 2),
                    'step_42layers_saved_ms': round((e2e_uf_warm_med - e2e_fus_warm_med) * 42.0 / 1000.0, 3),
                },
                'e2e_index_query_cold_128MiB_us': {
                    'unfused_med': e2e_uf_cold_med, 'unfused_iqr': e2e_uf_cold_iqr,
                    'fused_med': e2e_fus_cold_med, 'fused_iqr': e2e_fus_cold_iqr,
                    'delta_us': round(e2e_fus_cold_med - e2e_uf_cold_med, 2),
                    'step_42layers_saved_ms': round((e2e_uf_cold_med - e2e_fus_cold_med) * 42.0 / 1000.0, 3),
                },
                'e2e_index_query_cudagraph_us': {
                    'unfused_med': e2e_uf_cg_med, 'unfused_iqr': e2e_uf_cg_iqr,
                    'fused_med': e2e_fus_cg_med, 'fused_iqr': e2e_fus_cg_iqr,
                    'delta_us': round(e2e_fus_cg_med - e2e_uf_cg_med, 2),
                    'step_42layers_saved_ms': round((e2e_uf_cg_med - e2e_fus_cg_med) * 42.0 / 1000.0, 3),
                },
            }
            print(
                f'[Boot {args.boot}] rows={rows:2d} (bm={bm:2d}): '
                f'standalone warm uf={uf_warm_med:5.2f} 832={c832_warm_med:5.2f} shfl16={shfl_warm_med:5.2f} us | '
                f'e2e warm uf={e2e_uf_warm_med:5.2f} -> fus={e2e_fus_warm_med:5.2f} us | '
                f'e2e cold uf={e2e_uf_cold_med:5.2f} -> fus={e2e_fus_cold_med:5.2f} us | '
                f'e2e cg uf={e2e_uf_cg_med:5.2f} -> fus={e2e_fus_cg_med:5.2f} us'
            )

    payload = {
        'boot': args.boot,
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'gpu': torch.cuda.get_device_name(0),
        'pdl_violations': pdl_checks,
        'bit_exact_results': bit_exact_results,
        'bench_results': bench_results,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[Boot {args.boot}] Saved receipt to {args.out}')


if __name__ == '__main__':
    main()
