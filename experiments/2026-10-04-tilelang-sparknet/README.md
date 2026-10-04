# TileLang family on sparknet collectives

Base deployment commit: the [TileKernels mHC](../2026-10-04-tilelang-mhc/README.md)
experiment (vLLM patches 0032–0037).

The TileLang family's tensor-parallel collectives still came from B12X: the
one-shot RoCE all-reduce and all-gather (`b12x.comm.roce` with spark3's
patches 0001–0011), the largest B12X share left in a decode step.
[sparknet](https://github.com/christopherowen/dgx-spark-networking) vendors
exactly those collectives, decoupled from B12X's preparation framework, with
the measured fabric profiles. This experiment serves them from sparknet.

- **Image.** The lock lists sparknet at `685ab27` (0.1.0, tree `6b215155`).
  `bin/spark3 build image` then ends with `runtime-tilelang-sparknet`: the
  TileLang runtime plus the sparknet wheel, installed without dependencies.
  It builds the proxy into `SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce`, so
  no compiler runs at startup, and labels `local.spark3.sparknet.*`. The GPU
  smoke imports the vLLM adapter and loads the proxy.
- **[0038](vllm/0038-tilelang-sparknet-collectives.patch).** With the RoCE
  policy selected, the TileLang kernel family constructs sparknet's adapter
  (`SparknetOneShotAllReduce`) instead of `B12xRoceAllReduce`: the same
  constructor and methods, prepared at construction, the same environment.
  The worker freezes sparknet's kernel resolution after warm-up, so a compile
  inside a step raises instead of stalling it. The B12X family is unchanged.

[candidate.json](candidate.json) is the mHC candidate on the
`-tilelang-sparknet-v1` image.

## Procedure

sparknet had not run on the Sparks as a package, so the fabric is qualified
before serving, inside one window with serving stopped:

1. Build `-tilelang-sparknet-v1`; run the [tests](bundles/tests/candidate.json)
   bundle.
2. Run `bin/spark3 topology probe` on all four nodes with the candidate.
   This exercises vLLM's TP communicator, now sparknet's, with CUDA graphs.
3. Boot the mHC candidate (the control) and this arm in turn: decode at one
   and eight streams, prefill, and the single-stream decode profile for each.
