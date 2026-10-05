"""Print the CUDA source of chosen decode GEMM tiles (diagnosis of the small-tile race)."""
import importlib.util

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
CASES = {
    "bad-m16-n64-st2-t128": dict(block_M=16, block_N=64, block_K=128, num_stages=2, threads=128),
    "good-m16-n64-st2-t64": dict(block_M=16, block_N=64, block_K=128, num_stages=2, threads=64),
    "bad-m16-n32-st2-t128": dict(block_M=16, block_N=32, block_K=128, num_stages=2, threads=128),
    "large-m64-n64-st4-t128": dict(block_M=64, block_N=64, block_K=128, num_stages=4, threads=128),
}
for name, cfg in CASES.items():
    kernel = g.mxfp8_gemm(6400, 5120, **cfg, padded_rows=True)
    print(f"===== {name}")
    print(kernel.get_kernel_source())
print("===== end")
