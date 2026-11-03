"""Host cost of the none arms' layout CRC per budgeted step (not committed)."""
import time
from types import SimpleNamespace

from vllm.v1.worker.gpu import model_runner as mr

mr.logger.warning = lambda *a: None
for n in (1, 8, 16):
    ids = [f"req-{i}" for i in range(n)]
    r = SimpleNamespace(adaptive_verification=SimpleNamespace(
        _batch_budget=({k: 5 for k in ids}, {}, 0)))
    for _ in range(2000):
        mr.GPUModelRunner._layout_census(r, ids, 100)
    t = time.perf_counter()
    for _ in range(20000):
        mr.GPUModelRunner._layout_census(r, ids, 100)
    print(f"{n} requests: {(time.perf_counter() - t) / 20000 * 1e6:.2f} us a step")
