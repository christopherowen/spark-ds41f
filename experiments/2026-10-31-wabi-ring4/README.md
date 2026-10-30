# 2026-10-31 wabi-ring4: an external-site receipt for 1030

An independent four-node ring running spark-ds41f-1030, measured with `bin/spark bench`.
Two results worth recording: the Wi-Fi control-plane penalty from
[#712](https://github.com/christopherowen/spark-ds41f/issues/712) reproduces exactly, and
1030 is faster than 1025 on the same hardware once the control plane is on the ring.

## Site

| | |
| --- | --- |
| Nodes | 4 × DGX Spark (GB10), 128 GB unified memory each, ring cabling 1-2-3-4-1 |
| Kernel | `7.0.0-1019-nvidia-64k` (64 KiB pages), driver 580.178.04 |
| Memory saver | installed on three of four nodes |
| Fan control | `dgx-spark-fan-control` 0.1.4, curve shifted +7 °C |
| Shape | `tp4-ring`, 1M context, KV 10.5 GiB, no shape change |
| Image | built on site from `9461a70` |
| Weights | DeepSeek-V4.1-Flash-1017, unmodified |
| Clients | none during the runs; the API watchdog stopped for each |

## The control plane

Every run below is `bench --decode-cases prose-nothink,code-nothink --compare none`,
thinking off, seed 0, temperature 0, single stream, after a 60 s warm-up and 300 s window.

| 1030, prose, single stream | Wi-Fi control plane | ring control plane |
| --- | --- | --- |
| output tok/s | 71.4 ±6.4% | **83.4 ±1.0%** |
| decode window | 75.3 | **87.3** |
| engine step | 33.26 ms | **28.64 ms** |
| tokens/step | 2.500 | 2.500 |

Tokens per step are unchanged, so the loss is step time, not speculation — the same
signature #712 reports. Head-to-worker RTT on the ring is 0.20–0.58 ms to a neighbour and
1.24 ms mean to the opposite node; the Wi-Fi management network measured 3.8–4.4 ms with
spikes. Neighbour-only cabling leaves no host route to the opposite node, so each node
needs two static routes through a neighbour (`ip_forward=1`), pointed at the CX-7
interfaces; a oneshot unit re-applies them after boot.

## 1025 against 1030, both on the ring control plane

| | 1025 | 1030 |
| --- | --- | --- |
| prose, single stream | 80.5 ±0.5% | **83.4 ±1.0%** |
| prose, engine step | 29.71 ms | **28.64 ms** |
| code, single stream | 118.5 ±2.0% | **122.1 ±1.8%** |
| tokens/step, prose / code | 2.500 / 4.339 | 2.500 / 4.339 |
| quality gate | 5/5 | 5/5 |
| KV pool | 6,512,700 tokens (6.21× 1M) | **11,554,122 tokens (11.02× 1M)** |

## Prefill, 1030 on the ring control plane

3 samples per size, the benchmark's own filler text:

| prompt | tok/s |
| --- | --- |
| 32K | 7,990 |
| 64K | 7,980 |
| 128K | 7,854 |
| 262K | 7,519 |
| 524K | 6,861 |
| 1M | 5,828 |

The same sizes on the Wi-Fi control plane read 7,508 / 7,601 / 7,458 / 7,216 / 6,664 /
5,675, i.e. 2.7–6.4% lower, so the control plane moves prefill as well as decode.

## Thermal

1M prefill: GPU maximum 72–77 °C per node, system maximum 82.7–84.3 °C, power 78–80 W,
thermal slowdown 0 ms, fans at state 12/12 during the run and back to 2–4 afterwards. The
clock floor is 2,500 MHz with no ceiling; idle fans sit at 3,600–4,500 rpm.

## Commands

```sh
./bin/spark --cluster-config <ring-config> bench \
  --decode-cases prose-nothink,code-nothink --compare none

./bin/spark --cluster-config <ring-config> bench --suites prefill \
  --prefill-sizes 32768,65536,131072,262144,524288,1000000 --compare none
```

## Conclusion

1030 is 3.6% faster than 1025 in single-stream decode and 2.7–10% faster in prefill on
this site, with the KV pool holding 77% more tokens in the same reserved bytes. The
control plane must be on the ring: putting it on Wi-Fi costs 16.8% of decode on the same
build. Both findings agree with #712 and with the release comparison.
