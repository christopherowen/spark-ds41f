"""Matmul speed with weights in the display carve-out vs ordinary memory."""
import time, torch
from vllm.v1.worker import display_carveout as dc

dev = torch.device("cuda:0")
pool = dc.allocate(1536 << 20, dev)  # carve-out backing, int8 view
offset = 0
def carve(shape, dtype):
    global offset
    n = torch.Size(shape).numel() * torch.tensor([], dtype=dtype).element_size()
    t = pool[offset:offset + n].view(dtype).view(shape)
    offset += (n + 4095) // 4096 * 4096
    return t

def bench(fn, reps=20):
    fn(); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps * 1e3

cases = [  # (label, M, K, N)
    ("ViT qkv  M=8649 K=1024 N=3072", 8649, 1024, 3072),
    ("ViT mlp  M=8649 K=1024 N=5632", 8649, 1024, 5632),
    ("ViT down M=8649 K=2816 N=1024", 8649, 2816, 1024),
    ("decode   M=6    K=5120 N=16384", 6, 5120, 16384),
    ("decode   M=48   K=5120 N=16384", 48, 5120, 16384),
    ("head     M=6    K=5120 N=43136", 6, 5120, 43136),
]
for label, m, k, n in cases:
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_ord = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    w_car = carve((n, k), torch.bfloat16); w_car.copy_(w_ord)
    t_ord = bench(lambda: x @ w_ord.t()); t_car = bench(lambda: x @ w_car.t())
    gbs = n * k * 2 / 1e6
    print(f"{label}: ordinary {t_ord:.3f} ms ({gbs / t_ord:.0f} GB/s weights), carve-out {t_car:.3f} ms "
          f"({gbs / t_car:.0f} GB/s), carve-out/ordinary {t_car / t_ord:.2f}x", flush=True)
    del w_car; offset = 0
