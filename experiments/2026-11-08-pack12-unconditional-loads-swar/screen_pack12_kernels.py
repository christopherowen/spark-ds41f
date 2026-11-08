import json
import sys
import torch
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang.vocab import (
    pack_bf16_weight,
    packed_vocab_projection,
    packed_vocab_greedy_partials,
    finish_vocab_greedy_pairs,
    tile_config,
    row_tile,
)
import vllm.models.deepseek_v4_1.tilelang.gemm as gemm_mod
import vllm.models.deepseek_v4_1.tilelang.vocab as vocab_mod
from vllm.models.deepseek_v4_1.tilelang import pdl

def bench(fn, warmup=25, rep=150):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return round(s.elapsed_time(e) * 1000.0 / rep, 2)

# Run in either "baseline" or "candidate" mode, or compare in-process with distinct JIT functions
mode = sys.argv[1]  # "boot1" or "boot2"

# Apply patch in-place to get candidate module functions, while keeping baseline definitions for comparison
from pathlib import Path
vllm_root = Path("/opt/spark-ds41f/candidate/vllm")

results = {
    "boot": mode,
    "device": torch.cuda.get_device_name(0),
    "packed_bf16_gemm": {},
    "packed_vocab_projection": {},
    "packed_vocab_greedy": {},
}

# 1. Benchmark packed_bf16_gemm (baseline vs candidate)
HEADER_GEMM_OPT = gemm_mod._PACKED_BF16_HEADER.replace(
"""  if (num_exceptions <= 0) {
    *out_start = 0;
    *out_end = 0;
    return;
  }
  int start = __ldg(&row_offsets[row]);
  int end = __ldg(&row_offsets[row + 1]);""",
"""  const bool has_exc = (num_exceptions > 0);
  int start = has_exc ? __ldg(&row_offsets[row]) : 0;
  int end = has_exc ? __ldg(&row_offsets[row + 1]) : 0;"""
) + r"""
__device__ __forceinline__ void vllm_packed_bf16_load_row_bounds(
    int num_exceptions,
    const int* __restrict__ row_offsets,
    int row,
    int* __restrict__ out_start,
    int* __restrict__ out_end) {
  const bool has_exc = (num_exceptions > 0);
  *out_start = has_exc ? __ldg(&row_offsets[row]) : 0;
  *out_end = has_exc ? __ldg(&row_offsets[row + 1]) : 0;
}
"""

_PACKED_GROUP = 128
_PACKED_HALF = 64
_PACKED_VEC = 16

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def packed_bf16_gemm_opt(
    N: int, K: int, exponent_base: int, block_M: int = 64, block_N: int = 64,
    num_stages: int = 2, threads: int = 128, out_dtype: T.dtype = T.bfloat16,
    shards: int = 1, blocked: bool = False,
):
    k_groups = K // _PACKED_GROUP
    shard_groups = k_groups // shards
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = _PACKED_HALF // _PACKED_VEC
    items = block_N * slices
    iters = tilelang.cdiv(items, threads)
    single_b = blocked
    pipe_stages = min(num_stages, 2, k_groups)
    n_copies = 4 if blocked else 3
    n_compute = ((8 if single_b else 7) if blocked else 2) + (1 if shards > 1 else 0)
    n_stmts = n_copies + n_compute
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def gemm_packed_bf16_cand(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            T.import_source(HEADER_GEMM_OPT)
            pdl.chain()
            if blocked:
                A0_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                A1_shared = T.alloc_shared((block_M, _PACKED_HALF), T.bfloat16)
                B0_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                if not single_b:
                    B1_shared = T.alloc_shared((block_N, _PACKED_HALF), T.bfloat16)
                C_block = T.alloc_fragment((block_M, block_N), T.float32)
            else:
                A_shared = T.alloc_shared((block_M, _PACKED_GROUP), T.bfloat16)
                B_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, _PACKED_GROUP), T.uint8)
            code_shared = T.alloc_shared((block_N, _PACKED_HALF), T.uint8)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_total = T.alloc_fragment((block_M, block_N) if shards > 1 else (1, 1), T.float32)
            low_a = T.alloc_local((_PACKED_VEC,), T.uint8)
            low_b = T.alloc_local((_PACKED_VEC,), T.uint8)
            codes = T.alloc_local((_PACKED_VEC,), T.uint8)
            out_a = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            out_b = T.alloc_local((iters, _PACKED_VEC), T.bfloat16)
            exc_s = T.alloc_local((iters,), T.int32)
            exc_e = T.alloc_local((iters,), T.int32)
            tx = T.get_thread_binding()
            first = bx * block_N
            for step in T.serial(iters):
                item = step * threads + tx
                if items % threads == 0 or item < items:
                    j = item // slices
                    r_idx = T.min(first + j, N - 1)
                    T.call_extern(
                        "handle",
                        "vllm_packed_bf16_load_row_bounds",
                        E,
                        T.address_of(row_offsets[0]),
                        r_idx,
                        T.address_of(exc_s[step]),
                        T.address_of(exc_e[step]),
                    )
                else:
                    exc_s[step] = 0
                    exc_e[step] = 0
            T.clear(C_local)
            if shards > 1:
                T.clear(C_total)
            last = pipe_stages - 1
            for kg in (
                T.serial(k_groups)
                if pipe_stages <= 1
                else T.Pipelined(
                    k_groups,
                    order=list(range(n_stmts)),
                    stage=[0] * n_copies + [last] * n_compute,
                )
            ):
                if blocked:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A0_shared)
                    T.copy(A[by * block_M, kg * _PACKED_GROUP + _PACKED_HALF], A1_shared)
                else:
                    T.copy(A[by * block_M, kg * _PACKED_GROUP], A_shared)
                T.copy(W[bx * block_N, kg * _PACKED_GROUP], low_shared)
                T.copy(W[bx * block_N, K + kg * _PACKED_HALF], code_shared)
                for step in T.serial(iters):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // slices
                        c = item % slices * _PACKED_VEC
                        for v in T.vectorized(_PACKED_VEC):
                            low_a[v] = low_shared[j, c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            low_b[v] = low_shared[j, _PACKED_HALF + c + v]
                        for v in T.vectorized(_PACKED_VEC):
                            codes[v] = code_shared[j, c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_packed_bf16_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[step, 0]),
                            T.address_of(out_b[step, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(exc_s[step]),
                            exc_e[step],
                            kg * _PACKED_GROUP + c,
                        )
                        if blocked:
                            for v in T.vectorized(_PACKED_VEC):
                                B0_shared[j, c + v] = out_a[step, v]
                            if not single_b:
                                for v in T.vectorized(_PACKED_VEC):
                                    B1_shared[j, c + v] = out_b[step, v]
                        else:
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, c + v] = out_a[step, v]
                            for v in T.vectorized(_PACKED_VEC):
                                B_shared[j, _PACKED_HALF + c + v] = out_b[step, v]
                if blocked:
                    T.clear(C_block)
                    T.gemm(A0_shared, B0_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                    if single_b:
                        for step in T.serial(iters):
                            item = step * threads + tx
                            if items % threads == 0 or item < items:
                                j = item // slices
                                c = item % slices * _PACKED_VEC
                                for v in T.vectorized(_PACKED_VEC):
                                    B0_shared[j, c + v] = out_b[step, v]
                        T.clear(C_block)
                        T.gemm(A1_shared, B0_shared, C_block, transpose_B=True)
                    else:
                        T.clear(C_block)
                        T.gemm(A1_shared, B1_shared, C_block, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local[i, j] += C_block[i, j]
                else:
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                if shards > 1:
                    if (kg + 1) % shard_groups == 0:
                        for i, j in T.Parallel(block_M, block_N):
                            C_total[i, j] = T.if_then_else(
                                kg + 1 == shard_groups,
                                C_local[i, j],
                                C_total[i, j] + C_local[i, j],
                            )
                            C_local[i, j] = T.float32(0)
            if shards > 1:
                T.copy(C_total, C[by * block_M, bx * block_N])
            else:
                T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_packed_bf16_cand

gemm_shapes = [
    ("compressor_down_512x10240", 512, 10240),
    ("compressor_up_5120x512", 5120, 512),
    ("indexer_wq_b_4096x1280", 4096, 1280),
    ("indexer_wk_128x5120", 128, 5120),
]

for name, N, K in gemm_shapes:
    torch.manual_seed(42)
    W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
    W.view(-1)[::290] = 1e-6
    p = pack_bf16_weight(W)
    results["packed_bf16_gemm"][name] = {}
    for M in [1, 6, 16, 32, 64, 128, 512]:
        bm = 16 if M <= 16 else (32 if M <= 32 else 64)
        k_base = gemm_mod.packed_bf16_gemm(N, K, p.exponent_base, block_M=bm, block_N=64, num_stages=2, threads=128)
        k_cand = packed_bf16_gemm_opt(N, K, p.exponent_base, block_M=bm, block_N=64, num_stages=2, threads=128)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        C_base = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        C_cand = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        k_base(A, p.values, p.exception_entries, p.row_offsets, C_base)
        k_cand(A, p.values, p.exception_entries, p.row_offsets, C_cand)
        exact = bool(torch.equal(C_base.view(torch.uint16), C_cand.view(torch.uint16)))
        assert exact
        t_base = bench(lambda: k_base(A, p.values, p.exception_entries, p.row_offsets, C_base))
        t_cand = bench(lambda: k_cand(A, p.values, p.exception_entries, p.row_offsets, C_cand))
        results["packed_bf16_gemm"][name][str(M)] = {
            "baseline_us": t_base,
            "candidate_us": t_cand,
            "speedup": round(t_base / t_cand, 3),
            "exact_0_ulp": exact,
        }
        print(f"[gemm] {name} M={M:3d}: {t_base:.2f} -> {t_cand:.2f} us ({t_base/t_cand:.2f}x) exact={exact}")

# 2. Benchmark vocab projection and greedy argmax
# First compile baseline kernels
vocab_shapes = [
    ("tp4_32320x5120", 32320, 5120),
    ("tp3_43096x5120", 43096, 5120),
]

for name, N, K in vocab_shapes:
    torch.manual_seed(42)
    W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
    W.view(-1)[::300] = 1e-6
    p = pack_bf16_weight(W)
    base_proj = {}
    base_greedy = {}
    for bm in [16, 32, 64, 128]:
        cfg = tile_config(bm, K)
        base_proj[bm] = vocab_mod.packed_vocab_projection(N, K, p.exponent_base, block_M=bm, **cfg)
        base_greedy[bm] = vocab_mod.packed_vocab_greedy_partials(N, K, p.exponent_base, block_M=bm, **cfg)

    # Now patch vocab.py on disk and reload under a fresh module or compile candidate
    results["packed_vocab_projection"][name] = {}
    results["packed_vocab_greedy"][name] = {}

    for M in [1, 6, 12, 16, 24, 32, 48, 64, 96, 107, 128, 512, 2048]:
        bm = row_tile(M)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        C_base = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        k_b = base_proj[bm]
        k_b(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C_base)
        rep = 50 if M <= 128 else 20
        t_base = bench(lambda: k_b(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C_base), warmup=10, rep=rep)
        results["packed_vocab_projection"][name][str(M)] = {
            "baseline_us": t_base,
            "ref_hash": int(C_base.view(torch.uint16).to(torch.int64).sum().item()),
        }
        if M <= 128:
            cfg = tile_config(bm, K)
            nt = (N + cfg["block_N"] - 1) // cfg["block_N"]
            pvals_b = torch.empty((M, nt), device="cuda", dtype=torch.float32)
            pidxs_b = torch.empty((M, nt), device="cuda", dtype=torch.int32)
            pairs_b = torch.empty((M, 4), device="cuda", dtype=torch.float32)
            kg_b = base_greedy[bm]
            kf = finish_vocab_greedy_pairs(nt)
            kg_b(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pvals_b, pidxs_b, N)
            kf(pvals_b, pidxs_b, pairs_b, 0)
            tg_base = bench(lambda: (
                kg_b(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pvals_b, pidxs_b, N),
                kf(pvals_b, pidxs_b, pairs_b, 0)
            ), warmup=10, rep=50)
            results["packed_vocab_greedy"][name][str(M)] = {
                "baseline_us": tg_base,
                "ref_pairs": pairs_b[:, :2].cpu().tolist(),
            }

# Now patch vocab.py in-place using make_patch_0762 logic and reload vocab_mod
import importlib
vocab_path = Path("/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/vocab.py")
patched_vocab = Path("/tmp/pack12_exp/vllm-1107-exact/vllm/models/deepseek_v4_1/tilelang/vocab.py").read_text()
vocab_path.write_text(patched_vocab)
importlib.reload(vocab_mod)

for name, N, K in vocab_shapes:
    torch.manual_seed(42)
    W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
    W.view(-1)[::300] = 1e-6
    p = pack_bf16_weight(W)
    cand_proj = {}
    cand_greedy = {}
    for bm in [16, 32, 64, 128]:
        cfg = tile_config(bm, K)
        cand_proj[bm] = vocab_mod.packed_vocab_projection(N, K, p.exponent_base, block_M=bm, **cfg)
        cand_greedy[bm] = vocab_mod.packed_vocab_greedy_partials(N, K, p.exponent_base, block_M=bm, **cfg)

    for M in [1, 6, 12, 16, 24, 32, 48, 64, 96, 107, 128, 512, 2048]:
        bm = row_tile(M)
        A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        C_cand = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        k_c = cand_proj[bm]
        k_c(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C_cand)
        rep = 50 if M <= 128 else 20
        t_cand = bench(lambda: k_c(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C_cand), warmup=10, rep=rep)
        rec = results["packed_vocab_projection"][name][str(M)]
        cand_hash = int(C_cand.view(torch.uint16).to(torch.int64).sum().item())
        exact = (cand_hash == rec.pop("ref_hash"))
        assert exact
        rec["candidate_us"] = t_cand
        rec["speedup"] = round(rec["baseline_us"] / t_cand, 3)
        rec["exact_0_ulp"] = exact
        print(f"[vocab_proj] {name} M={M:4d}: {rec['baseline_us']:.2f} -> {t_cand:.2f} us ({rec['speedup']:.2f}x) exact={exact}")

        if M <= 128:
            cfg = tile_config(bm, K)
            nt = (N + cfg["block_N"] - 1) // cfg["block_N"]
            pvals_c = torch.empty((M, nt), device="cuda", dtype=torch.float32)
            pidxs_c = torch.empty((M, nt), device="cuda", dtype=torch.int32)
            pairs_c = torch.empty((M, 4), device="cuda", dtype=torch.float32)
            kg_c = cand_greedy[bm]
            kf = vocab_mod.finish_vocab_greedy_pairs(nt)
            kg_c(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pvals_c, pidxs_c, N)
            kf(pvals_c, pidxs_c, pairs_c, 0)
            tg_cand = bench(lambda: (
                kg_c(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pvals_c, pidxs_c, N),
                kf(pvals_c, pidxs_c, pairs_c, 0)
            ), warmup=10, rep=50)
            grec = results["packed_vocab_greedy"][name][str(M)]
            g_exact = (pairs_c[:, :2].cpu().tolist() == grec.pop("ref_pairs"))
            assert g_exact
            grec["candidate_us"] = tg_cand
            grec["speedup"] = round(grec["baseline_us"] / tg_cand, 3)
            grec["exact_0_ulp"] = g_exact
            print(f"[vocab_greedy] {name} M={M:4d}: {grec['baseline_us']:.2f} -> {tg_cand:.2f} us ({grec['speedup']:.2f}x) exact={g_exact}")

Path(f"/tmp/pack12_exp/{mode}.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"Wrote /tmp/pack12_exp/{mode}.json")
