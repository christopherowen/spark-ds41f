"""host cost of one view of a mask: getitem slice, narrow, and a dict hit (ours)."""
import time, torch
m = torch.zeros(8192, dtype=torch.bool, device="cuda")
d = {(id(m), 8192, 3, 0, 2731, False): (m, m[0:2731])}
k = (id(m), 8192, 3, 0, 2731, False)
def t(f, n=200000):
    for _ in range(1000): f()
    t0 = time.perf_counter()
    for _ in range(n): f()
    return round((time.perf_counter() - t0) * 1e9 / n, 1)
for mode in ("no_grad", "inference_mode"):
    with (torch.no_grad() if mode == "no_grad" else torch.inference_mode()):
        print(mode, "getitem", t(lambda: m[0:2731]), "narrow", t(lambda: m.narrow(0, 0, 2731)),
              "two_getitem", t(lambda: m[:8192][0:2731]), "dict_hit", t(lambda: d.get((id(m), 8192, 3, 0, 2731, False))))
