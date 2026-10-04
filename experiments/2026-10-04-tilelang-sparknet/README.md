# TileLang family on sparknet collectives

Base deployment commit: the [TileKernels mHC](../2026-10-04-tilelang-mhc/README.md)
experiment (vLLM patches 0032–0037).

The TileLang family's tensor-parallel collectives still came from B12X: the
one-shot RoCE all-reduce and all-gather (`b12x.comm.roce` with spark3's
patches 0001–0011), the largest B12X share left in a decode step.
[sparknet](https://github.com/christopherowen/dgx-spark-networking) vendors
exactly those collectives, decoupled from B12X's preparation framework, with
the measured fabric profiles. This experiment serves them from sparknet.

- **Image.** The lock lists sparknet at `f73a3ce` (0.2.0, tree `bd4191b3`).
  `bin/spark3 build image` then ends with `runtime-tilelang-sparknet`: the
  TileLang runtime plus the sparknet wheel, installed without dependencies.
  It builds the proxy into `SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce`, so
  no compiler runs at startup, and labels `local.spark3.sparknet.*`. The GPU
  smoke imports the vLLM adapter and loads the proxy.
- **Names.** sparknet reads only its own `SPARKNET_ROCE_*` settings. The
  `B12X_ROCE_*` and `VLLM_*ROCE*` aliases were retired in 0.2.0. One-shot has
  no enable switch: it is always on.
- **[0038](vllm/0038-tilelang-sparknet-collectives.patch).** The TileLang
  kernel family always constructs sparknet's adapter
  (`SparknetOneShotAllReduce`) for its tensor-parallel group:
  - it is prepared at construction;
  - it refuses `VLLM_ENABLE_ROCE_ALLREDUCE`, which does not apply to it;
  - it requires custom all-reduce;
  - the worker freezes sparknet's kernel resolution after warm-up, so a
    compile inside a step raises instead of stalling it.

  The B12X family is unchanged.
- **Transport.** spark3 adds sparknet's transports, `oneshot-direct` and
  `oneshot-ring4`. They route like RoCEnante's but render
  `SPARKNET_ROCE_PEER_HCAS`, `SPARKNET_ROCE_GID_INDEX` and
  `SPARKNET_ROCE_TOPOLOGY` per node, and they refuse any `B12X_ROCE_*` or
  B12X `VLLM_*` RoCE setting. The doctor pairs them with an image that
  carries `tilelang-sparknet-collectives` and the lock's sparknet source,
  and refuses that image on any other one-shot transport. The candidate sets
  `SPARKNET_ROCE_ALLREDUCE_CAPACITY_BYTES`, `ALLGATHER_MAX_BYTES` (2 MiB
  each), `ALLREDUCE_DISPATCH_MAX_BYTES` (1 MiB), `SPIN_LIMIT` and
  `GID_INDEX`: the same values as the B12X configuration, under sparknet's
  names.

[candidate.json](candidate.json) is the mHC candidate on the
`-tilelang-sparknet-v2` image and the `oneshot-ring4` transport. (`-v1`
carried sparknet 0.1.0 with its aliases; it was not served.)

## Results

**Collective probe.** `bin/spark3 topology probe` ran on all four nodes at
once with the candidate (numerics and benchmark), in vLLM's own TP
communicator, which logged `SPARKNET_ONESHOT` as its collective policy:

- every rank passed 20 checks and 8 exact BF16/FP32 numerical checks on
  transport `oneshot-ring4`;
- the runtime applied the requested settings: topology `ring4`, two stripes,
  2 MiB capacity and all-gather limit, 1 MiB dispatch limit, spin limit
  5,000,000;
- each rank posted 17,601 operations;
- a 5,120-element BF16 all-reduce took about 15 µs.

**Serving**, against the mHC candidate (the same image line on B12X's
collectives):

| | Control (B12X collectives) | sparknet | Change |
| --- | ---: | ---: | --- |
| Step time, prose c1 | 33.20 ms | 33.12 ms | −0.2% [−0.6, +0.1], same |
| Step time, code c1 | 36.65 ms | 36.46 ms | −0.5% [−1.1, +0.1], same |
| Decode, prose c8 | 215.8 tok/s | 218.3 tok/s | +1.1% [−2.6, +4.9], same |
| Decode, code c8 | 252.6 tok/s | 251.6 tok/s | −0.4% [−3.2, +2.5], same |
| Prefill, 32,768 tokens | 4,881 tok/s | 4,919 tok/s | +0.8% [−7.8, +9.3], same |
| Prefill, 262,144 tokens | 4,631 tok/s | 4,678 tok/s | +1.0% [−1.5, +3.6], same |

Quality passed 5 of 5. The serving log names `SPARKNET_ONESHOT`. In the
single-stream decode profile, sparknet's one-shot kernels replace B12X's one
for one, at the same launch counts:

- the all-reduce goes from 5.28 to 4.99 ms per scheduler step, much of it
  waiting on peers;
- the all-gather goes from 0.35 to 0.31 ms.

B12X's share of decode GPU time falls to 6.1%: its dense GEMM, attention
rotary, BF16 GEMV and KV-cache kernels.

## Procedure

`-tilelang-sparknet-v2` passed the tests bundle (97 of 97). One window ran
the collective probe on all four nodes, then booted the mHC candidate (the
control) and this arm in turn: decode at one and eight streams, prefill, and
the single-stream decode profile for each.
