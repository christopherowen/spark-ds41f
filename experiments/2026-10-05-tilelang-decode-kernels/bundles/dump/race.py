"""Small decode tiles: race or arithmetic? Repeat identical inputs, compare runs, prefill and FP32."""
import importlib.util

import torch

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(1)
N, K = 6400, 5120


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sfw(rows):
    return g.pack_scale_words(torch.randint(124, 131, (rows, K // 32), device=DEV, generator=GEN, dtype=torch.uint8))


def dq(x, w):
    e = w.contiguous().view(torch.uint8)[:, : x.shape[1] // 32].float() - 127
    return x.float() * torch.exp2(e).repeat_interleave(32, dim=1)


CASES = {
    "m16-n64-st2-t128": dict(block_M=16, block_N=64, block_K=128, num_stages=2, threads=128),
    "m16-n64-st2-t64": dict(block_M=16, block_N=64, block_K=128, num_stages=2, threads=64),
    "m16-n128-st2-t128": dict(block_M=16, block_N=128, block_K=128, num_stages=2, threads=128),
    "m16-n32-st2-t128": dict(block_M=16, block_N=32, block_K=128, num_stages=2, threads=128),
    "m64-n64-st4-t128": dict(block_M=64, block_N=64, block_K=128, num_stages=4, threads=128),
}
prefill = g.mxfp8_gemm(N, K, **g.default_config(g.DECODE_ROWS + 1, K))
print("prefill config", g.default_config(g.DECODE_ROWS + 1, K))
w, wsf = fp8((N, K)), sfw(N)
for name, cfg in CASES.items():
    kernel = g.mxfp8_gemm(N, K, **cfg, padded_rows=True)
    for rows in (6, 16):
        a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS)
        a_big = torch.cat([a[:rows], fp8((80 - rows, K))])
        asf_big = torch.cat([asf[:rows], sfw(80 - rows)])
        big = torch.empty(80, N, dtype=torch.bfloat16, device=DEV)
        prefill(a_big, w, asf_big, wsf, big)
        ref = dq(a[:rows], asf[:rows]) @ dq(w, wsf).T
        outs = []
        for _ in range(200):
            out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
            kernel(a, w, asf, wsf, out)
            outs.append(out)
        torch.cuda.synchronize()
        distinct = len({o.view(torch.int16).cpu().numpy().tobytes() for o in outs})
        bad = [o for o in outs if not torch.equal(o.view(torch.int16), big[:rows].view(torch.int16))]
        line = f"{name} rows={rows}: distinct outputs {distinct}/200, unequal to prefill {len(bad)}/200"
        if bad:
            o = bad[0]
            diff = (o.view(torch.int16) != big[:rows].view(torch.int16)).nonzero()
            r_set = sorted(set(diff[:, 0].tolist()))
            c = diff[:, 1]
            e_dec = ((o.float() - ref).norm() / ref.norm()).item()
            e_pre = ((big[:rows].float() - ref).norm() / ref.norm()).item()
            line += (f"; first mismatch: {len(diff)} elems, rows {r_set[:16]}, cols {c.min().item()}..{c.max().item()}"
                     f" (col mod 64 set {sorted(set((c % 64).tolist()))[:20]}), rel err decode {e_dec:.2e} prefill {e_pre:.2e}")
        print(line, flush=True)
