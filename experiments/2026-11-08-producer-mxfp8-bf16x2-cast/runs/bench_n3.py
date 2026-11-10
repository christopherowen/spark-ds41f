import json
import sys
import torch

from vllm.models.deepseek_v4_1.tilelang import (
    activation,
    columns,
    front,
    linear,
    mhc,
    norm,
)

label = sys.argv[1]  # e.g. "boot1_base_130", "boot1_cand_130", "boot1_base_135", "boot1_cand_135"
out_json = sys.argv[2]

device = torch.device("cuda:0")
torch.manual_seed(20261108)


def bench_graph(fn, warmup=50, rep=500):
    # Warmup and compile outside capture
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) * 1000.0) / (rep * 10)  # microseconds per kernel launch


results = {}

# 1. join_columns_and_cast_mxfp8 (widths = (5120, 5120, 5120))
for m in (4, 16, 64, 256, 1024):
    parts = [torch.randn((m, 5120), device=device, dtype=torch.bfloat16) for _ in range(3)]
    out_bf16 = torch.empty((m, 15360), device=device, dtype=torch.bfloat16)
    cast = linear.empty_mxfp8(out_bf16)
    key = columns.join_columns_and_cast_mxfp8_key((5120, 5120, 5120))
    fn_join = columns._bound_join_cast(*key)
    vals = cast.values[:m]
    sfs = cast.scales[:m].view(torch.uint32)
    us = bench_graph(lambda: fn_join(out_bf16, parts[0], parts[1], parts[2], vals, sfs, m))
    results[f"join_columns_m{m}"] = round(us, 3)

# 2. cast_mxfp8 / per_token_cast (hidden = 5120)
for m in (1, 4, 16, 64, 256, 1024):
    x = torch.randn((m, 5120), device=device, dtype=torch.bfloat16)
    us = bench_graph(lambda: linear.cast_mxfp8(x))
    results[f"cast_mxfp8_m{m}"] = round(us, 3)

# 3. activation_swiglu_clamp_cast (n = 1536, input = 3072)
act_mod = activation.SiluAndMulClampFP32(7.0, width=1536, mxfp8_output=True)
for m in (1, 4, 16, 64, 256, 1024):
    x = torch.randn((m, 3072), device=device, dtype=torch.bfloat16)
    us = bench_graph(lambda: act_mod.forward_mxfp8(x))
    results[f"act_swiglu_m{m}"] = round(us, 3)

# 4. mhc._finalize (hidden = 5120, k = 20480)
fn_fin = mhc._finalize(5120, 20480, 1e-6, 1e-4, 20)
for m in (1, 4, 16, 64, 256):
    P = torch.randn((40, m, 1, 24), device=device, dtype=torch.float32) * 0.01
    S = torch.rand((40, m, 1), device=device, dtype=torch.float32) * 0.1 + 0.01
    C = torch.randn((m, 5120), device=device, dtype=torch.bfloat16)
    Q = torch.rand((40, m), device=device, dtype=torch.float32) * 0.1 + 0.01
    scale = torch.rand(3, device=device, dtype=torch.float32) * 0.1 + 0.01
    base = torch.randn(24, device=device, dtype=torch.float32) * 0.1
    weight = torch.randn(5120, device=device, dtype=torch.bfloat16)
    post = torch.empty((m, 4), device=device, dtype=torch.float32)
    comb = torch.empty((m, 4, 4), device=device, dtype=torch.float32)
    pre_out = torch.empty((m, 4), device=device, dtype=torch.float32)
    y = torch.empty((m, 5120), device=device, dtype=torch.bfloat16)
    yq = torch.empty((m, 5120), device=device, dtype=torch.float8_e4m3fn)
    ysf = torch.empty((m, 160), device=device, dtype=torch.uint8)
    us = bench_graph(
        lambda: fn_fin(P, S, C, Q, scale, base, weight, post, comb, pre_out, y, yq, ysf)
    )
    results[f"mhc_finalize_m{m}"] = round(us, 3)

# 5. norm_forward_and_per_token_cast (hidden = 5120)
for m in (1, 4, 16, 64, 256, 1024):
    x = torch.randn((m, 5120), device=device, dtype=torch.bfloat16)
    res = torch.randn((m, 5120), device=device, dtype=torch.bfloat16)
    w = torch.randn(5120, device=device, dtype=torch.bfloat16)
    us = bench_graph(
        lambda: norm.norm_forward_and_per_token_cast(
            x=x,
            weight=w,
            eps=1e-6,
            fmt="e4m3",
            num_per_channels=32,
            residual=res,
            out_scale=1.0,
            use_tma_aligned_col_major_sf=False,
            round_sf=True,
            use_packed_ue8m0=True,
            with_out_bf16=True,
        )
    )
    results[f"norm_cast_m{m}"] = round(us, 3)

# 6. attention_front (Q_LATENT = 1280, KV_LATENT = 512)
cos_sin = torch.randn((256, 64), device=device, dtype=torch.float32)
q_w = torch.randn(1280, device=device, dtype=torch.bfloat16)
kv_w = torch.randn(512, device=device, dtype=torch.bfloat16)
cache = torch.zeros((16, 64, 528), device=device, dtype=torch.uint8)
for m in (1, 4, 16, 64, 256):
    qa_kv = torch.randn((m, 1792), device=device, dtype=torch.bfloat16)
    pos = torch.arange(m, device=device, dtype=torch.int64) % 256
    slots = torch.arange(m, device=device, dtype=torch.int64)
    qn = torch.empty((m, 1280), device=device, dtype=torch.bfloat16)
    q_values = torch.empty((m, 1280), device=device, dtype=torch.float8_e4m3fn)
    q_scales = torch.empty((m, 40), device=device, dtype=torch.uint8)
    kv = torch.empty((m, 512), device=device, dtype=torch.bfloat16)
    us = bench_graph(
        lambda: front.attention_front(
            qa_kv, q_w, kv_w, 1e-6, pos, cos_sin, cache, slots, 64, qn, q_values, q_scales, kv
        )
    )
    results[f"attention_front_m{m}"] = round(us, 3)

print(f"=== {label} ===")
for k, v in results.items():
    print(f"  {k:22s}: {v:7.3f} us")

try:
    with open(out_json) as f:
        all_res = json.load(f)
except FileNotFoundError:
    all_res = {}
all_res[label] = results
with open(out_json, "w") as f:
    json.dump(all_res, f, indent=2)
