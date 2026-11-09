#!/usr/bin/env python3
"""warm.py SCRIPT [ARGS...]: after one untimed GEMM (cuBLAS init), run at least 200 ms of back-to-back bf16 GEMMs on the GPU (clock
warm-up), then run SCRIPT unmodified as __main__ with ARGS (runpy), in the same process."""
import runpy, sys, time
import torch
x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
torch.mm(x, x, out=y); torch.cuda.synchronize()  # cuBLAS initialisation, outside the timed warm-up
t0 = time.perf_counter(); n = 0
while time.perf_counter() - t0 < 0.2:
    torch.mm(x, x, out=y); n += 1
    if n % 8 == 0: torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"[warm.py] {n} GEMMs in {time.perf_counter() - t0:.3f} s", file=sys.stderr, flush=True)
del x, y; torch.cuda.empty_cache()
script = sys.argv[1]; sys.argv = sys.argv[1:]
sys.path.insert(0, __import__("os").path.dirname(script))
runpy.run_path(script, run_name="__main__")
