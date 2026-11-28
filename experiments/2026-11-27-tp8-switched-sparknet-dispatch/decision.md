# Decision

Status: **promote** (scoped to `tp8-switched`)

On an 8-node switched DGX Spark cluster (`spark-ds41f:1126`, MikroTik CRS804 200G RoCE), setting `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=1048576` and `SPARKNET_ROCE_PROXY_CPU=fast` on `tp8-switched` delivered:

- **C32 decode:** `+17.3%` aggregate (`842.79 -> 989.00 tok/s`) and `+22.1%` per-stream (`32.97 -> 40.26 tok/s`), with `+31.8%` aggregate / `+38.4%` per-stream on `coding` (`39.82 -> 55.11`) and `+41.1%` / `+45.7%` on `ceiling_count` (`48.45 -> 70.60`).
- **Cold prefill:** `+13.3%` at 2K (`6,000.9 -> 6,799.9 tok/s`) and `+7.1%` at 8K (`9,807.7 -> 10,500.0 tok/s`); 32K essentially flat (`-0.6%`, `11,320.6 -> 11,247.3 tok/s`).
- **C1 / C8 / C16 decode:** `+0.5%` to `+1.5%` across all three points, zero regressions.
- **Clocks & health:** all 8 ranks verified `2,171–2,190 MHz` throughout both arms (`runs/clocks-base.txt`, `runs/clocks-sn1m.txt`), zero request failures, `>= 12.2 GiB` `MemAvailable` per rank.

Receipts: `runs/bench-base.md`, `runs/bench-sn1m.md`, `runs/clocks-base.txt`, `runs/clocks-sn1m.txt`.
