#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B benchmark for 64 KiB chunked, %globaltimer-paced L2 weight prefetch on GB10.

Measures three levels of hardware evidence in one deterministic pass:
1. Standalone 16 MiB weight prefetch + consumer read latency and effective L2
   hit rate across off, orig_tp4_g2_4k (grid=2, block=128, chunk=4096),
   orig_tp3_g4_4k (grid=4, block=128, chunk=4096), cand_tl_64k_p200ns (TileLang),
   and cand_cute_64k_p200ns (CuTe DSL).
2. 4-layer heterogeneous decode chain (3 SWA + 1 Indexer, 20 real TileLang kernels per
   layer + real WO, FFN, NEXT L2 prefetch plans) with zero-memory pure-wait windows
   across M = 1, 2, 4, 6, 8, 16, 24, 32 (interleaved round-robin CUDA graph replays,
   bit-exact digest verification).
3. Same 4-layer heterogeneous decode chain with concurrent memory-active window kernels
   across M = 1, 2, 4, 6, 8, 16, 24, 32 (interleaved round-robin CUDA graph replays,
   bit-exact digest verification).
"""

import argparse
import json
import math
import statistics
import sys

import tilelang
import tilelang.language as T
import torch

from tests.kernels.deepseek_v41_pdl_chains import LayerChain
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1 import l2_prefetch as ds_l2_prefetch
from vllm.models.deepseek_v4_1.tilelang import linear
from vllm.models.glm5next.nvidia import l2_prefetch as glm_l2_prefetch
from vllm.models.glm5next.nvidia.tilelang import l2_prefetch as tl_l2_prefetch

DEVICE = "cuda"

_ORIG_HEADER = r"""
__device__ __forceinline__ unsigned long long orig_l2_prefetch_policy_evict_last() {
    unsigned long long policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ void orig_l2_prefetch_bulk_prefetch(long long address, int bytes,
                                                               unsigned long long policy) {
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                 :: "l"(address), "r"(bytes), "l"(policy));
}
"""


@tilelang.jit(target="cuda")
def orig_l2_prefetch_kernel(grid: int, block: int, chunk: int):
    nseg2 = T.dynamic("nseg2")

    @T.prim_func
    def orig_l2_prefetch(segs: T.Tensor((nseg2,), T.int64)):
        with T.Kernel(grid, threads=block) as bx:
            T.import_source(_ORIG_HEADER)
            tx = T.get_thread_binding()
            tid = T.cast(bx, T.int64) * block + T.cast(tx, T.int64)
            stride = T.int64(grid * block)
            policy = T.alloc_local((1,), T.uint64)
            policy[0] = T.call_extern("orig_l2_prefetch_policy_evict_last", dtype=T.uint64)
            for s in T.serial(nseg2 // 2):
                base = segs[2 * s]
                nbytes = segs[2 * s + 1]
                nchunks = (nbytes + (chunk - 1)) // chunk
                for k in T.serial(
                    T.max((nchunks - tid + (stride - 1)) // stride, T.int64(0))
                ):
                    off = (tid + k * stride) * chunk
                    size = T.min(nbytes - off, T.int64(chunk)) // 16 * 16
                    if size > 0:
                        T.call_extern(
                            "orig_l2_prefetch_bulk_prefetch",
                            base + off,
                            T.cast(size, T.int32),
                            policy[0],
                            dtype="handle",
                        )

    return orig_l2_prefetch


def compile_orig_launcher(grid: int, block: int = 128, chunk: int = 4096):
    kernel = orig_l2_prefetch_kernel(grid, block, chunk)

    def launch(segs: torch.Tensor, stream_handle: int) -> None:
        with torch.cuda.stream(
            torch.cuda.ExternalStream(stream_handle, device=segs.device)
        ):
            kernel(segs)

    return launch


_WIN_HEADER = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void pure_wait_ns(int wait_ns, uint32_t* sink, int tid) {
    uint64_t t0, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    uint64_t target = t0 + (uint64_t)wait_ns;
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now < target);
    if (tid == 0) sink[0] = (uint32_t)now;
}

__device__ __forceinline__ uint32_t sim_window_work(
    const int* __restrict__ data, int mask, int iters, int tid) {
    uint32_t idx = (tid * 257) & mask;
    for (int i = 0; i < iters; ++i) {
        idx = (idx + (uint32_t)data[idx] + i * 32 + 1) & mask;
    }
    return idx;
}

__device__ __forceinline__ void read_16mb_vec4(
    const uint32_t* __restrict__ ptr,
    int n_vec4,
    uint32_t* __restrict__ out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    const uint4* p4 = reinterpret_cast<const uint4*>(ptr);
    uint32_t acc = 0;
    for (int i = tid; i < n_vec4; i += stride) {
        uint4 v;
        asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(&p4[i]));
        acc ^= (v.x + v.y) ^ (v.z + v.w);
    }
    if (acc == 0x12345678U) out[threadIdx.x] = acc;
}
"""


@tilelang.jit(target="cuda")
def make_pure_wait(wait_ns: int):
    @T.prim_func
    def wait_k(sink: T.Tensor((64,), T.uint32)):
        with T.Kernel(6, threads=128) as bx:
            T.import_source(_WIN_HEADER)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                "pure_wait_ns",
                wait_ns,
                T.address_of(sink[0]),
                bx * 128 + tx,
            )

    return wait_k


@tilelang.jit(target="cuda")
def make_mem_window_kernel(grid: int, iters: int):
    @T.prim_func
    def win_k(
        data: T.Tensor((262144,), T.int32), out: T.Tensor((grid * 128,), T.uint32)
    ):
        with T.Kernel(grid, threads=128) as bx:
            T.import_source(_WIN_HEADER)
            tx = T.get_thread_binding()
            tid = bx * 128 + tx
            out[tid] = T.call_extern(
                "uint32",
                "sim_window_work",
                T.address_of(data[0]),
                262143,
                iters,
                tid,
            )

    return win_k


@tilelang.jit(target="cuda")
def make_reader(n_u32: int):
    @T.prim_func
    def reader_k(
        buf: T.Tensor((n_u32,), T.uint32), out: T.Tensor((256,), T.uint32)
    ):
        with T.Kernel(40, threads=256) as _bx:
            T.import_source(_WIN_HEADER)
            T.call_extern(
                "handle",
                "read_16mb_vec4",
                T.address_of(buf[0]),
                T.int32(n_u32 // 4),
                T.address_of(out[0]),
            )

    return reader_k


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    mean = statistics.mean(s)
    stdev = statistics.stdev(s) if n > 1 else 0.0
    ci95 = 1.96 * stdev / math.sqrt(n) if n > 1 else 0.0
    q = statistics.quantiles(s, n=10) if n >= 10 else [s[0]] * 9
    return {
        "n": n,
        "median_us": round(statistics.median(s), 2),
        "mean_us": round(mean, 2),
        "stdev_us": round(stdev, 2),
        "ci95_us": round(ci95, 2),
        "p10_us": round(q[0], 2),
        "p90_us": round(q[8], 2),
        "min_us": round(s[0], 2),
    }


def build_launchers() -> dict:
    glm_l2_prefetch._compiled = None
    glm_l2_prefetch._compile_failed = False
    glm_l2_prefetch._KERNELS = "cute"
    cute_cand = glm_l2_prefetch._get_launcher()

    tl_cand = tl_l2_prefetch.compile_launcher(
        glm_l2_prefetch._GRID, glm_l2_prefetch._BLOCK, glm_l2_prefetch._CHUNK_BYTES
    )

    return {
        "off": None,
        "orig_tp4_g2_4k": compile_orig_launcher(2, 128, 4096),
        "orig_tp3_g4_4k": compile_orig_launcher(4, 128, 4096),
        "cand_tl_64k_p200ns": tl_cand,
        "cand_cute_64k_p200ns": cute_cand,
    }


def build_layer_plans(
    chain: LayerChain, next_chain: LayerChain, device: torch.device
):
    wo_segs = chain.wo.l2_segments()
    plan_wo = ds_l2_prefetch.make_plan(wo_segs, ds_l2_prefetch.BUDGET_WO, device)

    ffn_segs = [
        (
            "mhc.hc_ffn_fn",
            chain.ffn_entry.fn.data_ptr(),
            chain.ffn_entry.fn.numel() * chain.ffn_entry.fn.element_size(),
        ),
        (
            "mhc.hc_ffn_scale",
            chain.ffn_entry.scale.data_ptr(),
            chain.ffn_entry.scale.numel() * chain.ffn_entry.scale.element_size(),
        ),
        (
            "mhc.hc_ffn_base",
            chain.ffn_entry.base.data_ptr(),
            chain.ffn_entry.base.numel() * chain.ffn_entry.base.element_size(),
        ),
        (
            "ffn_norm.weight",
            chain.ffn_entry.weight.data_ptr(),
            chain.ffn_entry.weight.numel() * chain.ffn_entry.weight.element_size(),
        ),
    ]
    ffn_segs += ds_l2_prefetch.linear_segments("gate_up_proj", chain.gate_up)
    ffn_segs += ds_l2_prefetch.linear_segments("down_proj", chain.down)
    plan_ffn = ds_l2_prefetch.make_plan(ffn_segs, ds_l2_prefetch.BUDGET_FFN, device)

    next_segs = [
        (
            "mhc.hc_attn_fn",
            next_chain.attention_entry.fn.data_ptr(),
            next_chain.attention_entry.fn.numel()
            * next_chain.attention_entry.fn.element_size(),
        ),
        (
            "mhc.hc_attn_scale",
            next_chain.attention_entry.scale.data_ptr(),
            next_chain.attention_entry.scale.numel()
            * next_chain.attention_entry.scale.element_size(),
        ),
        (
            "mhc.hc_attn_base",
            next_chain.attention_entry.base.data_ptr(),
            next_chain.attention_entry.base.numel()
            * next_chain.attention_entry.base.element_size(),
        ),
        (
            "attn_norm.weight",
            next_chain.attention_entry.weight.data_ptr(),
            next_chain.attention_entry.weight.numel()
            * next_chain.attention_entry.weight.element_size(),
        ),
    ]
    next_segs += ds_l2_prefetch.linear_segments("wqa_wkv", next_chain.qa_kv)
    next_segs += ds_l2_prefetch.linear_segments("wq_b", next_chain.q_b)
    plan_next = ds_l2_prefetch.make_plan(next_segs, ds_l2_prefetch.BUDGET_NEXT, device)
    return plan_wo, plan_ffn, plan_next


def run_standalone_16mb(launchers: dict, flush: torch.Tensor, reps: int = 35) -> dict:
    n_bytes = 16 * 1024 * 1024
    n_u32 = n_bytes // 4
    weight = torch.randint(0, 1000, (n_u32,), dtype=torch.uint32, device=DEVICE)
    sink = torch.zeros((256,), dtype=torch.uint32, device=DEVICE)
    reader = make_reader(n_u32)
    reader(weight, sink)
    torch.cuda.synchronize()

    seg = glm_l2_prefetch.tensor_segment("w16m", weight)
    assert seg is not None
    plan = glm_l2_prefetch.L2PrefetchPlan([seg], weight.device)
    side = torch.cuda.Stream()

    for _ in range(5):
        reader(weight, sink)
    torch.cuda.synchronize()
    hot_samples = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        reader(weight, sink)
        e.record()
        torch.cuda.synchronize()
        hot_samples.append(s.elapsed_time(e) * 1000.0)
    hot_med = statistics.median(hot_samples)

    out = {"hot_read_median_us": round(hot_med, 2), "variants": {}}
    cold_med = None
    for name, launch in launchers.items():
        if launch is not None:
            launch(plan.segs, side.cuda_stream)
            torch.cuda.synchronize()
        pf_times = []
        read_times = []
        tot_times = []
        for _ in range(reps):
            flush.zero_()
            torch.cuda.synchronize()
            s0 = torch.cuda.Event(enable_timing=True)
            s1 = torch.cuda.Event(enable_timing=True)
            s2 = torch.cuda.Event(enable_timing=True)
            s0.record()
            if launch is not None:
                side.wait_stream(torch.cuda.current_stream())
                launch(plan.segs, side.cuda_stream)
                torch.cuda.current_stream().wait_stream(side)
            s1.record()
            reader(weight, sink)
            s2.record()
            torch.cuda.synchronize()
            pf_times.append(s0.elapsed_time(s1) * 1000.0)
            read_times.append(s1.elapsed_time(s2) * 1000.0)
            tot_times.append(s0.elapsed_time(s2) * 1000.0)
        read_stats = summarize(read_times)
        if name == "off":
            cold_med = read_stats["median_us"]
        hit_rate = (
            max(
                0.0,
                min(
                    1.0,
                    (cold_med - read_stats["median_us"]) / (cold_med - hot_med),
                ),
            )
            if cold_med and cold_med > hot_med
            else 0.0
        )
        out["variants"][name] = {
            "prefetch": summarize(pf_times),
            "read": read_stats,
            "total": summarize(tot_times),
            "effective_l2_hit_pct": round(hit_rate * 100.0, 1),
        }
    return out


def run_heterogeneous_decode_chain(
    launchers: dict,
    flush: torch.Tensor,
    row_list: tuple[int, ...] = (1, 2, 4, 6, 8, 16, 24, 32),
    replays: int = 35,
) -> dict:
    device = torch.device(DEVICE)
    pure_swa = make_pure_wait(38000)
    pure_idx = make_pure_wait(68000)
    pure_ar_attn = make_pure_wait(80000)
    pure_ar_moe = make_pure_wait(90000)
    pure_sink = torch.zeros((64,), dtype=torch.uint32, device=device)
    pure_swa(pure_sink)
    pure_idx(pure_sink)
    pure_ar_attn(pure_sink)
    pure_ar_moe(pure_sink)

    mem_swa = make_mem_window_kernel(6, 95)
    mem_idx = make_mem_window_kernel(6, 170)
    mem_ar_attn = make_mem_window_kernel(8, 210)
    mem_ar_moe = make_mem_window_kernel(8, 230)
    win_data = torch.randint(0, 1000, (262144,), dtype=torch.int32, device=device)
    win_out = torch.empty(8 * 128, dtype=torch.uint32, device=device)
    mem_swa(win_data, win_out[: 6 * 128])
    mem_idx(win_data, win_out[: 6 * 128])
    mem_ar_attn(win_data, win_out)
    mem_ar_moe(win_data, win_out)
    torch.cuda.synchronize()

    arm_names = (
        "off",
        "orig_tp4_g2_4k",
        "orig_tp3_g4_4k",
        "cand_tl_64k_p200ns",
        "cand_cute_64k_p200ns",
    )

    out = {"pure_wait": {}, "mem_active": {}}
    num_layers = 4
    with set_current_vllm_config(VllmConfig()):
        for rows in row_list:
            chains = [LayerChain(rows, seed=100 + i) for i in range(num_layers)]
            plans = [
                build_layer_plans(chains[i], chains[(i + 1) % num_layers], device)
                for i in range(num_layers)
            ]
            side = torch.cuda.Stream(device=device)

            for mode in ("pure_wait", "mem_active"):
                def run_step(launch_fn):
                    main = torch.cuda.current_stream()
                    for i, c in enumerate(chains):
                        p_wo, p_ffn, p_next = plans[i]
                        r, a, f = c.rows, c.attention_entry, c.ffn_entry
                        a(c.x, c.streams, c.post_in, c.comb_in, c.pre_in)
                        linear._block32_gemm(
                            c.qa_kv, a.mx.values, a.mx.scales, c.qa, r
                        )
                        c.q_norm(c.qa[:, :1280], out=c.qn)
                        c._project("q_b", c.q_b, c.qn, c.qb)

                        if launch_fn is not None:
                            side.wait_stream(main)
                            launch_fn(p_wo.segs, side.cuda_stream)

                        if mode == "pure_wait":
                            (pure_swa if i < 3 else pure_idx)(pure_sink)
                        else:
                            (mem_swa if i < 3 else mem_idx)(
                                win_data, win_out[: 6 * 128]
                            )

                        xq, sf = c.casts["wo_b"]
                        c.wo.a(
                            c.qb,
                            xq[:r],
                            sf[:r],
                            c.partials[0] if c.partials else None,
                        )
                        linear._block32_gemm(c.wo.b, xq, sf, c.wo_out, r)

                        if launch_fn is not None:
                            side.wait_stream(main)
                            launch_fn(p_ffn.segs, side.cuda_stream)

                        if mode == "pure_wait":
                            pure_ar_attn(pure_sink)
                        else:
                            mem_ar_attn(win_data, win_out)

                        f(c.wo_out, a.streams, a.post, a.comb, a.pre)
                        c.aux.wait_stream(main)
                        with torch.cuda.stream(c.aux):
                            linear._block32_gemm(
                                c.gate_up, f.mx.values, f.mx.scales, c.h, r
                            )
                            c.swiglu(c.h, c.act)
                            c._project("down", c.down, c.act, c.shared_out)

                        def shared(c=c, main=main):
                            main.wait_stream(c.aux)
                            return c.shared_out

                        c.plan.run(
                            f.mx,
                            c.ids,
                            c.route_weights,
                            c.experts,
                            c.out,
                            c.workspace,
                            shared=shared,
                        )

                        if launch_fn is not None:
                            side.wait_stream(main)
                            launch_fn(p_next.segs, side.cuda_stream)

                        if mode == "pure_wait":
                            pure_ar_moe(pure_sink)
                        else:
                            mem_ar_moe(win_data, win_out)

                    if launch_fn is not None:
                        main.wait_stream(side)

                graphs = {}
                ref_digests = None
                for name in arm_names:
                    launch_fn = launchers[name]
                    warmup_s = torch.cuda.Stream(device=device)
                    warmup_s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup_s):
                        run_step(launch_fn)
                    torch.cuda.current_stream().wait_stream(warmup_s)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        run_step(launch_fn)
                    g.replay()
                    torch.cuda.synchronize()
                    dig = [c.digests() for c in chains]
                    if ref_digests is None:
                        ref_digests = dig
                    else:
                        assert dig == ref_digests, (
                            f"Digest mismatch at rows={rows} mode={mode} arm={name}!"
                        )
                    graphs[name] = g

                samples = {name: [] for name in arm_names}
                for _ in range(replays):
                    for name in arm_names:
                        flush.zero_()
                        torch.cuda.synchronize()
                        start = torch.cuda.Event(enable_timing=True)
                        stop = torch.cuda.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay()
                        stop.record()
                        torch.cuda.synchronize()
                        samples[name].append(
                            start.elapsed_time(stop) * 1000.0 / num_layers
                        )

                mode_res = {}
                for name in arm_names:
                    mode_res[name] = summarize(samples[name])
                    mode_res[name]["bit_exact"] = True
                out[mode][f"rows_{rows}"] = mode_res
                del graphs
            del chains, plans
            torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1)
    ap.add_argument("--replays", type=int, default=35)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    flush = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    launchers = build_launchers()

    print(
        f"[Boot {args.boot}] 1/2 Running standalone 16 MiB L2 prefetch + read probe...",
        flush=True,
    )
    standalone = run_standalone_16mb(launchers, flush, reps=args.replays)
    for k, v in standalone["variants"].items():
        print(
            f"  {k:22s}: pf={v['prefetch']['median_us']:6.1f} us  "
            f"read={v['read']['median_us']:6.1f} us  "
            f"tot={v['total']['median_us']:6.1f} us  "
            f"L2_hit={v['effective_l2_hit_pct']:5.1f}%",
            flush=True,
        )

    print(
        f"[Boot {args.boot}] 2/2 Running 4-layer heterogeneous decode chain (M=1..32)...",
        flush=True,
    )
    hetero = run_heterogeneous_decode_chain(launchers, flush, replays=args.replays)
    for mode in ("pure_wait", "mem_active"):
        print(f"  === mode: {mode} ===", flush=True)
        for rkey, rres in hetero[mode].items():
            g2 = rres["orig_tp4_g2_4k"]["median_us"]
            g4 = rres["orig_tp3_g4_4k"]["median_us"]
            tl = rres["cand_tl_64k_p200ns"]["median_us"]
            cu = rres["cand_cute_64k_p200ns"]["median_us"]
            off = rres["off"]["median_us"]
            print(
                f"  {rkey:8s}: off={off:7.2f}  g2={g2:7.2f}  g4={g4:7.2f}  "
                f"cand_tl={tl:7.2f} ({tl - g2:+.2f} / {(tl - g2) / g2 * 100:+.2f}% vs g2, "
                f"{tl - g4:+.2f} / {(tl - g4) / g4 * 100:+.2f}% vs g4)  "
                f"cand_cu={cu:7.2f}",
                flush=True,
            )

    payload = {
        "boot": args.boot,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "replays": args.replays,
        "standalone_16mb": standalone,
        "heterogeneous_4layer_chain": hetero,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Boot {args.boot}] Saved receipt to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
