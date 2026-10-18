# r6x candidate: the routed SwiGLU, 4-bit expert scale codes, a draft budget with nothing timed

r6x is r6v ([experiments/2026-10-17-r6v](../2026-10-17-r6v/README.md)) with the three
changes of 2026-10-18 the owner asked to promote together: "when we are done, I would like
to promote the swiGLU and expert scale. we aren't done until we have solved the drafting
question."

## Change (vLLM series: r6w's, then 0182-0184; tree 4d6851c8)

- **0181, the routed SwiGLU** ([2026-10-18-r6w-swiglu](../2026-10-18-r6w-swiglu/README.md)):
  TileKernels' SwiGLU and MXFP8 cast over the live tiles of the dispatch layout. Bits
  identical; engine grid -2.2% step time, single stream +2.6%.
- **0182-0183, the routed experts' scales as row bases and 4-bit codes**
  ([2026-10-18-scale-codes](../2026-10-18-scale-codes/README.md)): each output row's UE8M0
  scales as its smallest exponent and a 4-bit delta per scale, expanded in the grouped
  GEMM's stage. Exact (rows span at most 8 exponents); bits identical; c16 +3.75%, prefill
  32K +0.56%; about 2.1 GiB a rank freed at TP4 (2.7 GiB at TP3).
- **0184, the DSpark draft budget by a fixed linear step cost**
  ([2026-10-18-dspark-linear-budget](../2026-10-18-dspark-linear-budget/README.md)): the
  cost model's argmax with its boot-timed curves replaced by one recipe constant,
  `adaptive_verification_row_cost` 0.02. The boot profile, the pinned curves and the cost
  scale are deleted, so the draft schedule is a function of the inputs and the recipe
  alone, the same on every build and boot.

The recipe (`r6x-tp4-1017.json`) is r6v's with the r6x image and the speculative config's
`adaptive_verification_cost_scale` 2.0 replaced by `adaptive_verification_row_cost` 0.02.
The KV pool is unchanged, so the comparison is the engine's alone; the freed scale memory
is headroom for a later KV change.

## Runs

1. `bundles/config-discovery` (r6v image, r6x's configuration files mounted):
   `tests/test_config.py`, which the series now changes, run as a gate file before it is
   classified. The series moves the row cost's two configuration checks beside the adaptive
   verification tests, which the release gates; `config/release-tests.json` gates the
   warm-up gate test and, pending this run, `tests/test_config.py`.
2. `build.json`: the image on dgx2 from `source.json` (patch head 7316c7f9), copied to the
   other nodes.
3. `gate-tp4-1017.json`: determinism (temperature-0 and greedy outputs equal to r6v's,
   3eab5112 / db25fde3, c9cf9dbe) and distinct streams.
4. `release-tests.json`: the release tests (`scripts/release_tests.py gate`, 60 files).
5. `release-tp4-1017.json`: the release comparison with its companions, single-stream and
   sampled: r6x, r6v, r6x again.

Gate: outputs identical to r6v's; no metric worse than r6v.

## Promotion (the owner's decision)

- `config/cluster.json`: the r6x image and lock; `adaptive_verification_row_cost` 0.02 in
  place of the cost scale in the speculative config. The recipe has one image for both
  shapes: r6v serves the triangle on r6u's TP3 path, while 0181-0184 change TP3's path too
  (its kernel bits are tested; its serving is not).
- `bin/spark` stops deriving `SPARK3_DSPARK_COST_DIR`, with its tests and docs, once no
  production shape runs an image that reads it (TP3 still serves r6u's).
- TP3: `tp3-triangle` needs the triangle cabling to qualify r6x there.
