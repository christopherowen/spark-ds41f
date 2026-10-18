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
  `adaptive_verification_row_cost` (vLLM's default 0.02; the recipe sets 0.035). The boot
  profile, the pinned curves and the cost scale are deleted, so the draft schedule is a
  function of the inputs and the recipe alone, the same on every build and boot.

The recipe (`r6x-tp4-1017.json`) is r6v's with the r6x image and the speculative config's
`adaptive_verification_cost_scale` 2.0 replaced by `adaptive_verification_row_cost` 0.035:
both constants were level with r6v's timed budget on throughput, and 0.035 keeps r6v's
operating point at sixteen streams (tokens a step and first-token latency level) where 0.02
verified 8-12% more for the same output. The KV pool is unchanged, so the comparison is the engine's alone; the freed scale memory
is headroom for a later KV change.

## Runs

1. `bundles/config-discovery` (r6v image, r6x's configuration files mounted):
   `tests/test_config.py`, which the series now changes, run as a gate file before it is
   classified: 279 passed, 117 skipped (uncached Hub configs), so it is gated. The series
   moves the row cost's two configuration checks beside the adaptive verification tests,
   which the release gates; `config/release-tests.json` also gates the warm-up gate test.
2. `build.json`: the image on dgx2 from `source.json` (patch head 7316c7f9), copied to the
   other nodes.
3. `gate-tp4-1017.json`: determinism (temperature-0 and greedy outputs equal to r6v's,
   3eab5112 / db25fde3, c9cf9dbe) and distinct streams.
4. `release-tests.json`: the release tests (`scripts/release_tests.py gate`, 60 files).
5. `release-tp4-1017.json`: the release comparison with its companions, single-stream and
   sampled: r6x, r6v, r6x again.

Gate: outputs identical to r6v's; no metric worse than r6v.

## Promotion (the owner's decision)

- `config/cluster.json`: the r6x image and lock; `adaptive_verification_row_cost` 0.035 in
  place of the cost scale in the speculative config. The recipe has one image for both
  shapes: r6v serves the triangle on r6u's TP3 path, while 0181-0184 change TP3's path too
  (its kernel bits are tested; its serving is not).
- The README's Benchmarking section loses its note that boots differ by about 3% from the
  startup cost profile, and docs/lab.md its pinned-curves rule.
- `bin/spark` stops deriving `SPARK3_DSPARK_COST_DIR`, with its tests and docs, once no
  production shape runs an image that reads it (TP3 still serves r6u's).
- TP3: `tp3-triangle` needs the triangle cabling to qualify r6x there.

## Results (2026-10-18, the ring, 1017)

**Build and tests.** Image `vllm-ds41f-kkref:19f2c20ed4d6-r6x`, one ID on every node
(sha256:6339bda18247). Release tests: 60 gate files, 1,680 passed, 158 skipped, 0 failed.
The `tests/test_config.py` discovery: 279 passed, 117 skipped (uncached Hub configs).

**Determinism gate.** Temperature-0 and greedy outputs identical to r6v's (3eab5112 /
db25fde3, c9cf9dbe / c6bb4f4e), zero token and log-probability differences in every
scenario. The release suite's model step: greedy responses identical 256/256, teacher-forced
distributions identical over 65,536 positions (KL 0). Boot 76 s (r6v 87-90 s: no cost
profile).

**Release comparison** (r6x, r6v, r6x again; verdicts against the run's own repeat; tables in [results/](results/)):

| Suite | Metric | r6v | r6x | Change | ±95% | Repeat | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| release | engine grid, geometric mean of 20 shapes (step time) | | | -2.53% | 0.15% | +0.66% | better |
| release | c16 tok/s | 241.7 | 250.3 | +3.55% | 0.85% | -0.38% | better |
| release | c8 tok/s | 184.4 | 191.5 | +3.84% | 1.11% | -0.05% | better |
| release | Poisson tok/s (the offered load) | 180.4 | 180.4 | +0.02% | 17.1% | +0.01% | same |
| release | prefill 32K-1M tok/s | | | -0.01 to +0.03% | 0.19-1.09% | | same |
| single-stream | c1 tok/s | 70.95 | 74.14 | +4.49% | 4.49% | +0.06% | same |
| sampled | c16 tok/s | 237.2 | 244.2 | +2.96% | 0.94% | -0.08% | better |
| sampled | c8 tok/s | 180.1 | 188.1 | +4.49% | 1.46% | -0.26% | better |
| sampled | c1 tok/s | 68.5 | 71.4 | +4.23% | 5.70% | +0.11% | same |

No verdict is worse. The grid is faster at all 20 shapes (0.75-4.2%). The engine runs more
steps a second at the same tokens a step: c16 steps/s +2.6% with tokens a step level (30.8
against 30.5), single stream steps/s +4.1% (interval 1.7%) with tokens a step level. The
two one-stream throughputs read same only because the unpaired interval over windows of
different prompts is 4.5-5.7%; each repeat matched itself within 0.1%.

Latency (percentiles, no interval): Poisson first token p50 -5.7% and p99 -11.8%, time per
token p50 -7.7%; c8 time per token p50 -4.3%; single stream first token p50 -5.0%. The one
"worse" row, sampled one-stream first token p50 (95 to 100 ms, both r6x arms), is a median
of different request sets: r6x's faster window held 84 requests against r6v's 80, and the
same distribution's mean (100.0 to 98.4 ms), p90 (118.5 to 115.0) and p99 (128.5 to 123.2)
all improved.

**Repeatability.** The same build a session apart: tokens a step within 0.13% (c16 30.78 /
30.74), throughput within 0.4% everywhere, the grid +0.66% (a boot-level shift every shape
shared; the 10-14 clock-lock experiment found such shifts belong to the boot, not the
clocks). Drafting no longer differs between builds.

**Verdict:** correct (outputs identical to r6v's, release tests pass) and faster: the
routed SwiGLU and the scale codes shorten every decode step by about 2.5% and raise batched
throughput 3-4.5%, with the draft budget a rule of the inputs. For the owner's promotion
decision.
