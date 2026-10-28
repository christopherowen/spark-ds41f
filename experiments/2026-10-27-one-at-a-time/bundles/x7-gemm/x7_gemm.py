#!/usr/bin/env python3
"""X7 weights decoded in the MXFP8 decode GEMM's tile load: bit-exact against the
production kernel, timed on cold weights.

X7 (experiments/2026-10-26-decode-bytes/x7.py) stores an E4M3 weight in 7 bits an
element: sign and mantissa as a nibble, the exponent as a 3-bit code (c < 7 is
exponent 8 + c; c = 7 escapes to a 4-bit exponent in the cluster's escape stream).
This bundle builds three decode GEMMs with the production MMA
(``T.mma_gemm_blockscaled``, MXFP8, UE8M0 words):

- ``prod``: vLLM's ``mxfp8_gemm_decode`` at the production decode tile, E4M3 weights
  and per-row scale words (``pack_scale_words(rows=N)``), as serving runs it;
- ``x8``: a hand-pipelined kernel (cp.async stages, as tilelang/moe.py) reading the
  same E4M3 tiles but the 32x32 block scales themselves, one word a 32-row block;
- ``x7``: the same pipeline reading X7 planes (7/8 of the bytes) and block scales;
  each stage's tile is decoded into the MMA's shared B tile after it lands. One warp
  takes a 16-row cluster: a lane counts its half-row's escapes, a warp scan gives
  each lane its first escape, and the tile decodes element by element;
- ``x7w``: ``x7`` decoding eight elements a step (an unescaped byte is sign, 1, code,
  mantissa, built four at a time from a nibble word and three code bytes), then a pass
  that rewrites only the escaped elements.

On real weights of DeepSeek-V4.1-Flash-1017 (layer 10, a TP4 rank's slices, read from
the node's hub) it checks that ``x8`` and ``x7`` give ``prod``'s output bit for bit at
1-16 rows, then times each kernel per call on cold weights: a CUDA graph of CALLS
launches on as many copies of the weight, the L2 flushed before each replay. One JSON
line a case. Exit status 0 only when every output matched.
"""
import json
import statistics
import sys

import tilelang
import tilelang.language as T
import torch
from safetensors import safe_open

from vllm.models.deepseek_v4_1.tilelang import gemm as prod

SNAPSHOT = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
            "1bae0e1088993b9b82de08f59de5be5a34e42170")
BASE = 8  # X7's exponent window is 8..14
CLUSTER = 16  # rows a warp decodes; the escape stream is kept per cluster
BLOCK_K = 128
CALLS, REPEATS = 16, 9
ROWS = (1, 2, 4, 6, 8, 16)
DEVICE = "cuda"


def popc(x):
    return T.call_extern("int32", "__popc", x)


def u32(v):
    return T.uint32(v)


def _gemm(N, K, x7, block_M=16, block_N=64, num_stages=3, word=False):
    """``C[M, N] = (A * SFA) @ (W * SFB)^T`` with W as E4M3 tiles (x7=False) or X7
    planes (x7=True) and SFB the [N / 32, K / 128] block-scale words. ``word``: the X7
    tile decodes eight elements a step from one nibble word and three code bytes, then
    patches the escapes."""
    threads = 2 * block_N  # two lanes a row, one per 64-element half
    assert K % BLOCK_K == 0 and N % block_N == 0 and block_N % CLUSTER == 0
    k_blocks = K // BLOCK_K
    sf_words = K // 128
    stages = num_stages
    M = T.dynamic("M")
    Mp = T.dynamic("Mp")
    esc_words = T.dynamic("esc_words")
    a_segments = block_M * 8  # 16-byte copies of one stage's activation rows
    a_copies = tilelang.cdiv(a_segments, threads)
    p_segments = block_N * 7  # 16-byte copies of one stage's X7 planes (28 words a row)
    p_copies = tilelang.cdiv(p_segments, threads)
    b_segments = block_N * 8  # 16-byte copies of one stage's E4M3 tile
    b_copies = tilelang.cdiv(b_segments, threads)
    word_items = block_N * 16  # (row, nibble word): eight elements each
    group_items = block_N * 4  # (row, 32-element group): one escape mask each

    @T.macro
    def load(A, SFA, W, P, SFB, A_shared, B_stage, P_shared, SFA_shared, SFB_shared,
             bx, by, tx, ko, s):  # fmt: skip
        for c in T.unroll(a_copies):
            e = c * threads + tx
            if e < a_segments:
                T.ptx_cp_async(
                    T.access_ptr(A_shared[s, e // 8, e % 8 * 16], "w", 16),
                    T.access_ptr(A[by * block_M + e // 8, ko * BLOCK_K + e % 8 * 16], "r", 16),
                    16,
                )
        if x7:
            for c in T.unroll(p_copies):
                e = c * threads + tx
                if e < p_segments:
                    T.ptx_cp_async(
                        T.access_ptr(P_shared[s, e // 7, e % 7 * 4], "w", 4),
                        T.access_ptr(P[ko, bx * block_N + e // 7, e % 7 * 4], "r", 4),
                        4,
                    )
        else:
            for c in T.unroll(b_copies):
                e = c * threads + tx
                if e < b_segments:
                    T.ptx_cp_async(
                        T.access_ptr(B_stage[s, e // 8, e % 8 * 16], "w", 16),
                        T.access_ptr(W[bx * block_N + e // 8, ko * BLOCK_K + e % 8 * 16], "r", 16),
                        16,
                    )
        if tx < block_M:
            T.ptx_cp_async(
                T.access_ptr(SFA_shared[s, tx, 0], "w", 1),
                T.access_ptr(SFA[by * block_M + tx, ko], "r", 1),
                1,
            )
        if tx < block_N:
            # A 32-row block's scale word serves each of its rows.
            T.ptx_cp_async(
                T.access_ptr(SFB_shared[s, tx, 0], "w", 1),
                T.access_ptr(SFB[(bx * block_N + tx) // 32, ko], "r", 1),
                1,
            )
        T.ptx_commit_group()

    @T.macro
    def body(A, SFA, W, P, ESC, OFF, SFB, C):
        with T.Kernel(N // block_N, T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((stages, block_M, BLOCK_K), T.float8_e4m3fn)
            SFA_shared = T.alloc_shared((stages, block_M, 1), T.uint32)
            SFB_shared = T.alloc_shared((stages, block_N, 1), T.uint32)
            if x7:
                P_shared = T.alloc_shared((stages, block_N, 28), T.uint32)
                start = T.alloc_shared((block_N, 4), T.int32)
                B_tile = T.alloc_shared((block_N, BLOCK_K), T.float8_e4m3fn)
                B_stage = None
            else:
                P_shared = None
                B_stage = T.alloc_shared((stages, block_N, BLOCK_K), T.float8_e4m3fn)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            count = T.alloc_var(T.int32)
            scan = T.alloc_var(T.int32)
            tx = T.get_thread_binding()
            lane = tx % 32
            T.clear(C_local)
            for p in T.unroll(stages - 1):
                if p < k_blocks:
                    load(A, SFA, W, P, SFB, A_shared, B_stage, P_shared, SFA_shared,
                         SFB_shared, bx, by, tx, p, p)  # fmt: skip
                else:
                    T.ptx_commit_group()
            for ko in T.serial(k_blocks):
                nxt = ko + stages - 1
                if nxt < k_blocks:
                    load(A, SFA, W, P, SFB, A_shared, B_stage, P_shared, SFA_shared,
                         SFB_shared, bx, by, tx, nxt, nxt % stages)  # fmt: skip
                else:
                    T.ptx_commit_group()
                T.ptx_wait_group(stages - 1)
                T.sync_threads()
                s = ko % stages
                if x7:
                    # Lane (row i, half h) counts the escapes of groups 2h and 2h + 1;
                    # the warp (a 16-row cluster) scans the counts in element order.
                    i = tx // 2
                    h = tx % 2
                    m0 = (P_shared[s, i, 16 + 6 * h] & P_shared[s, i, 17 + 6 * h]
                          & P_shared[s, i, 18 + 6 * h])
                    m1 = (P_shared[s, i, 19 + 6 * h] & P_shared[s, i, 20 + 6 * h]
                          & P_shared[s, i, 21 + 6 * h])
                    count = popc(m0) + popc(m1)
                    scan = count
                    for step in T.unroll(5):
                        prior = T.shfl_up(scan, 1 << step)
                        scan = scan + T.if_then_else(lane >= (1 << step), prior, 0)
                    first = OFF[ko, (bx * block_N + i) // CLUSTER] * 8 + scan - count
                    start[i, 2 * h] = first
                    start[i, 2 * h + 1] = first + popc(m0)
                    T.sync_threads()
                    if word:
                        # An unescaped exponent is 8 + code, so a byte is sign, 1, the
                        # code and the mantissa: four bytes at once from four nibbles
                        # and their code bits (bit q of a spread lands in byte q).
                        for c in T.unroll(word_items // threads):
                            e = c * threads + tx
                            i2 = e // 16
                            w = e % 16
                            nibw = P_shared[s, i2, w]
                            shift = T.cast(w % 4 * 8, T.uint32)
                            for h in T.unroll(2):
                                n4 = (nibw >> u32(16 * h)) & u32(0xFFFF)
                                n4 = (n4 | (n4 << u32(8))) & u32(0x00FF00FF)
                                n4 = (n4 | (n4 << u32(4))) & u32(0x0F0F0F0F)
                                packed = (((n4 & u32(0x08080808)) << u32(4)) | u32(0x40404040)
                                          | (n4 & u32(0x07070707)))
                                for b in T.unroll(3):
                                    bits = (P_shared[s, i2, 16 + 3 * (w // 4) + b] >> shift
                                            >> u32(4 * h)) & u32(15)
                                    packed = packed | (((bits * u32(0x00204081)) & u32(0x01010101))
                                                       << u32(3 + b))
                                for q in T.unroll(4):
                                    B_tile[i2, w * 8 + h * 4 + q] = T.reinterpret(
                                        T.cast((packed >> u32(8 * q)) & u32(255), T.uint8),
                                        T.float8_e4m3fn,
                                    )
                        T.sync_threads()
                        for c in T.unroll(group_items // threads):
                            e = c * threads + tx
                            i2 = e // 4
                            g = e % 4
                            mask = (P_shared[s, i2, 16 + 3 * g] & P_shared[s, i2, 17 + 3 * g]
                                    & P_shared[s, i2, 18 + 3 * g])
                            if mask != u32(0):
                                for t in T.serial(32):
                                    if (mask >> T.cast(t, T.uint32)) & u32(1) != u32(0):
                                        rank = start[i2, g] + popc(mask & ((u32(1) << T.cast(t, T.uint32)) - u32(1)))
                                        exponent = (ESC[rank >> 3] >> T.cast((rank & 7) * 4, T.uint32)) & u32(15)
                                        j = g * 32 + t
                                        nib = (P_shared[s, i2, j // 8] >> T.cast(j % 8 * 4, T.uint32)) & u32(15)
                                        B_tile[i2, j] = T.reinterpret(
                                            T.cast(((nib >> u32(3)) << u32(7)) | (exponent << u32(3))
                                                   | (nib & u32(7)), T.uint8),
                                            T.float8_e4m3fn,
                                        )
                    else:
                        for i2, j in T.Parallel(block_N, BLOCK_K):
                            g = j // 32
                            t = T.cast(j % 32, T.uint32)
                            nib = (P_shared[s, i2, j // 8] >> T.cast(j % 8 * 4, T.uint32)) & u32(15)
                            w0 = P_shared[s, i2, 16 + 3 * g]
                            w1 = P_shared[s, i2, 17 + 3 * g]
                            w2 = P_shared[s, i2, 18 + 3 * g]
                            code = (((w0 >> t) & u32(1)) | (((w1 >> t) & u32(1)) << u32(1))
                                    | (((w2 >> t) & u32(1)) << u32(2)))
                            rank = start[i2, g] + popc((w0 & w1 & w2) & ((u32(1) << t) - u32(1)))
                            # Signed test: TVM's bounds solver negates the constant of a
                            # guarded read's condition, and -7 has no uint32 literal.
                            exponent = T.if_then_else(
                                T.cast(code, T.int32) == 7,
                                (ESC[rank >> 3]
                                 >> T.cast((rank & 7) * 4, T.uint32)) & u32(15),
                                code + u32(BASE),
                            )
                            B_tile[i2, j] = T.reinterpret(
                                T.cast(((nib >> u32(3)) << u32(7)) | (exponent << u32(3))
                                       | (nib & u32(7)), T.uint8),
                                T.float8_e4m3fn,
                            )
                    T.sync_threads()
                    T.mma_gemm_blockscaled(
                        A_shared[s, :, :], B_tile, C_local,
                        SFA_shared[s, :, :], SFB_shared[s, :, :],
                        transpose_B=True, k_start=0,
                        sf_a_granularity_k=32, sf_b_granularity_k=32,
                    )
                else:
                    T.mma_gemm_blockscaled(
                        A_shared[s, :, :], B_stage[s, :, :], C_local,
                        SFA_shared[s, :, :], SFB_shared[s, :, :],
                        transpose_B=True, k_start=0,
                        sf_a_granularity_k=32, sf_b_granularity_k=32,
                    )
                # Every warp is done with this stage (and the decoded tile) before
                # the next iteration's loads and decode reuse them.
                T.sync_threads()
            T.copy(C_local, C[by * block_M, bx * block_N])

    if x7:

        @T.prim_func
        def gemm_x7(
            A: T.Tensor((Mp, K), T.float8_e4m3fn),
            SFA: T.Tensor((Mp, sf_words), T.uint32),
            P: T.Tensor((k_blocks, N, 28), T.uint32),
            ESC: T.Tensor((esc_words,), T.uint32),
            OFF: T.Tensor((k_blocks, N // CLUSTER), T.int32),
            SFB: T.Tensor((N // 32, sf_words), T.uint32),
            C: T.Tensor((M, N), T.bfloat16),
        ):
            body(A, SFA, None, P, ESC, OFF, SFB, C)

        return gemm_x7

    @T.prim_func
    def gemm_x8(
        A: T.Tensor((Mp, K), T.float8_e4m3fn),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        W: T.Tensor((N, K), T.float8_e4m3fn),
        SFB: T.Tensor((N // 32, sf_words), T.uint32),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        body(A, SFA, W, None, None, None, SFB, C)

    return gemm_x8


gemm = tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})(_gemm)


# ------------------------------------------------------------------ X7 packing (torch)


def to_u32(x):
    """int64 holding 32-bit patterns -> a uint32 view of the same bits."""
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32).view(torch.uint32)


def pack(w):
    """E4M3 bytes [N, K] (uint8, CUDA) -> (planes [K/128, N, 28], escapes, offsets)."""
    n, k = w.shape
    kb = k // 128
    w = w.to(torch.int64)
    e = (w >> 3) & 15
    nib = ((w >> 7) << 3) | (w & 7)
    code = torch.where((e >= BASE) & (e <= BASE + 6), e - BASE, torch.full_like(e, 7))
    seg = lambda a: a.view(n, kb, 128).transpose(0, 1).contiguous()  # [K/128, N, 128]
    nib_s, code_s, e_s = seg(nib), seg(code), seg(e)
    nw = nib_s.view(kb, n, 16, 8)
    nib_words = sum(nw[..., i] << (4 * i) for i in range(8))
    cw = code_s.view(kb, n, 4, 32)
    code_words = torch.stack(
        [sum(((cw[..., j] >> b) & 1) << j for j in range(32)) for b in range(3)], dim=-1)
    planes = to_u32(torch.cat([nib_words, code_words.reshape(kb, n, 12)], dim=2)).contiguous()
    mask = (code_s == 7).view(kb, n // CLUSTER, CLUSTER * 128)
    vals = e_s.view(kb, n // CLUSTER, CLUSTER * 128)[mask]
    cnt = mask.sum(dim=2).flatten()
    words = (cnt + 7) // 8
    offsets = (torch.cumsum(words, 0) - words).to(torch.int32).view(kb, n // CLUSTER)
    rank = torch.arange(vals.numel(), device=w.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    nibbles = torch.zeros(max(int(words.sum()), 1) * 8, dtype=torch.int64, device=w.device)
    nibbles[torch.repeat_interleave(offsets.flatten().long() * 8, cnt) + rank] = vals
    escapes = (nibbles[0::2] | (nibbles[1::2] << 4)).to(torch.uint8).view(torch.int32).view(torch.uint32)
    return planes, escapes.contiguous(), offsets.contiguous(), int(cnt.sum())


def unpack(planes, escapes, offsets, n, k):
    """The reference decoder (what the tile load computes), for the round-trip check."""
    kb = k // 128
    p = planes.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    nib = torch.stack([(p[..., :16] >> (4 * i)) & 15 for i in range(8)], dim=-1).reshape(kb, n, 128)
    cw = p[..., 16:].reshape(kb, n, 4, 3)
    code = sum(torch.stack([(cw[..., b] >> j) & 1 for j in range(32)], dim=-1) << b for b in range(3))
    code = code.reshape(kb, n, 128)
    e = code + BASE
    by = escapes.view(torch.uint8).to(torch.int64)
    nibbles = torch.stack([by & 15, by >> 4], dim=1).flatten()
    mask = (code == 7).view(kb, n // CLUSTER, CLUSTER * 128)
    cnt = mask.sum(dim=2).flatten()
    rank = torch.arange(int(cnt.sum()), device=planes.device) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    e = e.view(kb, n // CLUSTER, CLUSTER * 128)
    e[mask] = nibbles[torch.repeat_interleave(offsets.flatten().long() * 8, cnt) + rank]
    e = e.view(kb, n, 128)
    b = ((nib >> 3) << 7) | (e << 3) | (nib & 7)
    return b.transpose(0, 1).reshape(n, k).to(torch.uint8)


# ------------------------------------------------------------------ weights and runs


def tensor(name):
    index = json.load(open(f"{SNAPSHOT}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{SNAPSHOT}/{index[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def weight(names, rows):
    """A TP4 rank's E4M3 rows of the named tensors, stacked, with their 32x32 scales."""
    ws, ss = [], []
    for name, (r0, r1) in zip(names, rows):
        w = tensor(name + ".weight").view(torch.uint8)
        s = tensor(name + ".scale").view(torch.uint8)
        ws.append(w[r0:r1])
        ss.append(s[r0 // 32:r1 // 32])
    return torch.cat(ws).to(DEVICE).contiguous(), torch.cat(ss).to(DEVICE).contiguous()


SHAPES = {  # TP4 rank 0, layer 10
    "Q-B (8192, 1280)": (["layers.10.attn.wq_b"], [(0, 8192)]),
    "fused Q-A/KV (1792, 5120)": (["layers.10.attn.wq_a", "layers.10.attn.wkv"], [(0, 1280), (0, 512)]),
    "shared gate/up (1152, 5120)": (["layers.10.ffn.shared_experts.w1", "layers.10.ffn.shared_experts.w3"],
                                     [(0, 576), (0, 576)]),
    "indexer Q-B (4096, 1280)": (["layers.8.attn.indexer.wq_b"], [(0, 4096)]),
    "Engram wkv (6400, 6144)": (["layers.14.engram.wkv"], [(0, 6400)]),
}


def timed(calls, flush):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for call in calls[:2]:
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    samples = []
    for _ in range(REPEATS):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    return statistics.median(samples), min(samples)


def activations(rows_padded, k, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randint(0, 256, (rows_padded, k), generator=gen, device=DEVICE, dtype=torch.uint8)
    a = torch.where((a & 0x7F) == 0x7F, a & 0xF0, a)  # no NaN codes
    exps = torch.randint(118, 128, (rows_padded, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    return a.view(torch.float8_e4m3fn), prod.pack_scale_words(exps)


def main():
    torch.cuda.set_device(0)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    ok = True
    for label, (names, rows) in SHAPES.items():
        w, scales = weight(names, rows)
        n, k = w.shape
        planes, escapes, offsets, n_esc = pack(w)
        exact = torch.equal(unpack(planes, escapes, offsets, n, k), w)
        ok &= exact
        stored = n * k + n * 4 * (k // 128)
        x7_bytes = planes.numel() * 4 + escapes.numel() * 4 + offsets.numel() * 4 + (n // 32) * (k // 128) * 4
        print(json.dumps({"case": "pack", "shape": label, "round_trip_exact": exact,
                          "escape_share": round(n_esc / (n * k), 4),
                          "bytes_prod": stored, "bytes_x8": n * k + (n // 32) * (k // 128) * 4,
                          "bytes_x7": x7_bytes, "x7_vs_prod": round(x7_bytes / stored, 4)}), flush=True)
        sfb_rows = prod.pack_scale_words(scales, rows=n)
        sfb_blocks = prod.pack_scale_words(scales)
        copies = [(w.clone(), sfb_rows.clone(), planes.clone(), escapes.clone(), offsets.clone(),
                   sfb_blocks.clone()) for _ in range(CALLS)]
        wf = [c[0].view(torch.float8_e4m3fn) for c in copies]
        for m in ROWS:
            mp = 16
            a, sfa = activations(mp, k, 7 * m + k)
            ref_cfg = prod.fp8_decode_config(n, k, mp)
            ref = prod.mxfp8_gemm_decode(n, k, **ref_cfg, padded_rows=True)
            outs = {}
            c_ref = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
            ref(a, wf[0], sfa, copies[0][1], c_ref)
            best = {}
            for kind in ("x8", "x7", "x7w"):
                for block_n in (16, 32, 64):
                    for stages in (2, 3):
                        if n % block_n:
                            continue
                        kern = gemm(n, k, kind != "x8", 16, block_n, stages, kind == "x7w")
                        c = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
                        if kind != "x8":
                            kern(a, sfa, copies[0][2], copies[0][3], copies[0][4], copies[0][5], c)
                        else:
                            kern(a, sfa, wf[0], copies[0][5], c)
                        torch.cuda.synchronize()
                        same = torch.equal(c.view(torch.int16), c_ref.view(torch.int16))
                        ok &= same
                        if kind != "x8":
                            calls = [lambda c_=copies[i], o=c: kern(a, sfa, c_[2], c_[3], c_[4], c_[5], o)
                                     for i in range(CALLS)]
                        else:
                            calls = [lambda i=i, o=c: kern(a, sfa, wf[i], copies[i][5], o) for i in range(CALLS)]
                        us, us_min = timed(calls, flush)
                        line = {"case": "gemm", "shape": label, "rows": m, "kernel": kind,
                                "block_N": block_n, "stages": stages, "bit_exact": same,
                                "us": round(us, 2), "us_min": round(us_min, 2)}
                        print(json.dumps(line), flush=True)
                        if same and (kind not in best or us < best[kind]["us"]):
                            best[kind] = line
            calls = [lambda i=i: ref(a, wf[i], sfa, copies[i][1], c_ref) for i in range(CALLS)]
            us, us_min = timed(calls, flush)
            summary = {"case": "summary", "shape": label, "rows": m, "prod_us": round(us, 2),
                       "prod_config": ref_cfg,
                       "prod_gbs": round(stored / us / 1e3, 1)}
            for kind, line in best.items():
                summary[f"{kind}_us"] = line["us"]
                summary[f"{kind}_config"] = [line["block_N"], line["stages"]]
                summary[f"{kind}_vs_prod"] = round(line["us"] / us, 3)
            if n * k <= 12 << 20:
                # Warm: every launch reads the same copy, which stays in L2, as the
                # prefetched projections are read in serving.
                summary["prod_warm_us"] = round(timed([lambda: ref(a, wf[0], sfa, copies[0][1], c_ref)] * CALLS,
                                                      flush)[0], 2)
                for kind, line in best.items():
                    kern = gemm(n, k, kind != "x8", 16, line["block_N"], line["stages"], kind == "x7w")
                    o = torch.empty((m, n), dtype=torch.bfloat16, device=DEVICE)
                    if kind != "x8":
                        call = lambda: kern(a, sfa, copies[0][2], copies[0][3], copies[0][4], copies[0][5], o)
                    else:
                        call = lambda: kern(a, sfa, wf[0], copies[0][5], o)
                    summary[f"{kind}_warm_us"] = round(timed([call] * CALLS, flush)[0], 2)
            print(json.dumps(summary), flush=True)
        del copies, wf
        torch.cuda.empty_cache()
    print(json.dumps({"case": "verdict", "all_exact": bool(ok)}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
