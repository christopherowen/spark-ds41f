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

## Procedure

sparknet had not run on the Sparks as a package, so the fabric is qualified
before serving, inside one window with serving stopped:

1. Build `-tilelang-sparknet-v1`; run the [tests](bundles/tests/candidate.json)
   bundle.
2. Run `bin/spark3 topology probe` on all four nodes with the candidate.
   This exercises vLLM's TP communicator, now sparknet's, with CUDA graphs.
3. Boot the mHC candidate (the control) and this arm in turn: decode at one
   and eight streams, prefill, and the single-stream decode profile for each.
