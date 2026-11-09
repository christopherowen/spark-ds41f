import sys
root = sys.argv[1]
g = f"{root}/vllm/models/deepseek_v4_1/tilelang/gemm.py"
s = open(g).read()
old_cfg = """    (12288, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=128, num_stages=2),
    },
    (8192, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=64, block_K=128, num_stages=3),
    },
"""
new_cfg = """    # 64-row Q-B tiles read whole 128-row tile-contiguous slabs (block_N =
    # TILE_WEIGHT_N); N // 128 >= 64 tiles still cover the 48 SMs.
    (12288, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
    },
    (8192, 1280): {
        16: dict(block_N=128, block_K=128, num_stages=3),
        32: dict(block_N=64, block_K=128, num_stages=3),
        64: dict(block_N=128, block_K=128, num_stages=2),
    },
"""
assert s.count(old_cfg) == 1
s = s.replace(old_cfg, new_cfg)
open(g, "w").write(s)
t = f"{root}/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py"
s = open(t).read()
old_t = """    for block_M in DECODE_TILE_ROWS:
        cfg = fp8_decode_config(n, k, block_M)
        k_ref = mxfp8_gemm_decode(n, k, **cfg, padded_rows=True)
"""
new_t = """    for block_M in DECODE_TILE_ROWS:
        cfg = fp8_decode_config(n, k, block_M)
        if block_M == DECODE_ROWS and n >= 8192:
            assert cfg["block_N"] == 128 and cfg["block_K"] == 128, (n, k, block_M, cfg)
        k_ref = mxfp8_gemm_decode(n, k, **cfg, padded_rows=True)
"""
assert s.count(old_t) == 1
s = s.replace(old_t, new_t)
open(t, "w").write(s)
print("edited")
