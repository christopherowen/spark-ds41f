# usage: fingerprint_gpu_check.py   (GPU, cluster stopped; mounts attn-exact checksum_debug.py at /cd.py)
# The Triton row fingerprint against the torch reference: BF16 and FP32, widths 1-11264, 1-64
# non-contiguous rows, one flipped value per case, and a CUDA graph capture and replay.
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("cd", "/cd.py")
cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(cd)
log = cd._Log.__new__(cd._Log)
torch.manual_seed(0)
bad = 0; checked = 0
for dtype in (torch.bfloat16, torch.float32):
    for width in (1, 32, 128, 384, 512, 1280, 5120, 11264):
        for rows in (1, 7, 19, 64):
            x = torch.randn(rows, width + 3, device="cuda").to(dtype)[:, :width]   # non-contiguous
            a = torch.empty(rows, device="cuda"); b = torch.empty(rows, device="cuda")
            log.row_sums(x, a)
            cd._fingerprint_torch(x.reshape(rows, -1), b)
            checked += 1
            if not torch.equal(a, b):
                bad += 1; print("MISMATCH", dtype, width, rows, a[:3].tolist(), b[:3].tolist())
            y = x.clone(); y[rows // 2, width // 2] = -y[rows // 2, width // 2] if y[rows // 2, width // 2] != 0 else 1.0
            c = torch.empty(rows, device="cuda"); log.row_sums(y, c)
            if torch.equal(a, c): bad += 1; print("MISSED CHANGE", dtype, width, rows)
x = torch.randn(48, 11264, device="cuda").bfloat16(); o = torch.empty(48, device="cuda")
g = torch.cuda.CUDAGraph()
log.row_sums(x, o)
with torch.cuda.graph(g):
    log.row_sums(x, o)
ref = o.clone(); o.zero_(); g.replay(); torch.cuda.synchronize()
print(f"triton vs torch reference: {checked - bad}/{checked} cases agree and detect a flipped value; "
      f"graph replay reproduces: {torch.equal(o, ref)}; max {float(ref.max()):.0f} < 2**24")
