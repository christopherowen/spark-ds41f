# Boot time

Base: the promoted configuration (`2026-09-28-karmic-kraken-r5h`).

## Question

A cold start takes about 150 s from `bin/spark3 cluster start` to "cluster
ready" (124 s from container start to API). Weight reading is already at
NVMe rate through the B12X O_DIRECT loader (101 GB in 15 s), so where does
the rest go, and which of these cut it?

1. Persisting the driver's PTX JIT cache (`CUDA_CACHE_PATH`; each boot
   rebuilt ~195 MB in the container's home) and TileLang's cache.
2. Explaining the phases no log line accounts for (API-server setup, the
   gap before graph capture, post-load processing) with py-spy samples.
3. NCCL communicator init, timed with `NCCL_DEBUG_SUBSYS=INIT`, with and
   without `NCCL_GIN_ENABLE=0`.
4. The launcher: one SSH connection per call (about 0.3 s each) and a 10 s
   readiness poll. `bin/spark3` now multiplexes SSH and polls every second.

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `base` | none |
| `persist` | `CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE`, `TILELANG_CACHE_DIR` under `/cache/kkref/jit` |
| `nccl-info` | `persist` + NCCL INIT logging |
| `nogin` | `nccl-info` + `NCCL_GIN_ENABLE=0` |

## Method

`boot.sh ARM LABEL [pyspy]` stops everything, starts the arm with every
launcher line timestamped, saves each node's container log under
`results/private/boot/LABEL`, and prints `timeline.py`'s phase table
(seconds after the launcher started). `pyspy_sampler.py` takes repeated
`py-spy dump` samples of the dgx1 container's Python processes during a boot.
`sequence.sh` runs the boots and restores the promoted service.

## Results

Pending.
