# rank-order-strides

One variable: the reduction source patch in vllm/series. Base and candidate identities
are in base.json and candidate.json. Apply after the exact candidate-38 stack;
no promoted configuration, lock, baseline or serving runtime is changed.

Source patch includes its upstream base, rationale, quality implications, tests
and local upstream status. The overlay is an exact Git export of the patch tree,
not an authority independent of the patch. Every test first checks original source
hashes in the frozen image, then verifies all patched file hashes before import.

Run on authorized idle sp8 from a clean published checkout:

```sh
python3 experiments/2026-11-04-rank-order-strides/run.py \
  --experiments experiments/2026-11-04-rank-order-strides --run reduction-v1
```

The runner holds an exclusive sp8 reservation and GPU lock, pre-cools below 55 C,
monitors memory, and removes only its own test container/reservation. Each variant
boots twice for independent correctness and twice for paired cold-L2 CUDA-graph
microbenchmarks (21 samples per method). CUDA profiler inventories compare the
original operation's GPU kernels with the candidate. Reference operations are
independent microkernels, not a remeasurement of a released model build.

All runs, failures, exact commands, memory observations and source identities
will be retained. CUDA graphs, independent references, input immutability and
unsupported-input behavior are explicit gates. Full model/real-fabric qualification
is outside this isolated test, and no serving speedup is claimed.

## Change and validation

The packed BF16 kernel remains for its aligned path. Real CUDA inputs with
tails or strides use a TileLang FP32 rank-order reduction; CPU retains the original
reference. A size-one metadata view normalizes model tensor ranks to eight
without moving data. Aliased output uses independent scratch; more than eight
parts retain FP32 intermediates. Unsupported CUDA contracts fail explicitly,
without switching to ATen. The normal floating model keys and copy-out keys are
registered in the same argument form used at runtime.

Both final boots pass **240 tests**, with no skips or failures. The two
benchmark boots each pass **24 independent-reference cases** and retain 21
cold-L2 graph samples per method per case. CUDA profiler traces contain the
expected TileLang candidate kernel and no ATen kernel for the captured replacement
operation. The original references show the ATen work being replaced; user scope
annotations in the raw event lists are not GPU kernels. Traces and logs stay on
sp8, with hashes and exact commands in the portable receipts.

Across 24 paired timing cases, ratios of boot means range from -77.86% to +4.82% candidate time. Every result is retained; this is not a blanket speedup verdict.

The eight-part BF16 tail case (7 x 513) changes from 34.98–36.58 us to
7.90–7.94 us. The largest percentage increase is an unchanged packed one-row
BF16 path at world eight: 3.94–4.03 us versus 4.16–4.19 us (4.82%, about 0.19 us
on means). Do not treat these descriptive ratios as serving throughput evidence;
timing excludes host submission costs and actual cross-rank traffic.

`results.json` retains both boot medians and 95% Student t intervals (df=1).
Two boots give limited generalization; ratios are descriptive without a
significance claim. Source overlays are isolated kernel/model-unit tests, not a
new OCI build or full-model serving qualification.

## Evidence and local checks

The final native run is `/home/chris/spark-lab/aten-runtime-20261104/replacements-v1`, published deployment source
`1462b9207b6a96deb52aaf4db83370d477acc375`. This checkout aggregates the three experiments for
back-to-back scheduling; each fresh test container applies only its own source
patch. Raw source and patched file hashes are checked before each import.
The minimum sampled host MemAvailable for this experiment is 96.57 GiB.
No guard stop or workload error. Both batch reservations were released and their
own test containers removed; no serving operation was performed.

`runs/local.json` records a fresh reproduction of all 572 candidate-38 patches,
independent application of this patch, matching source/export trees, Python
parsing, doctor and diff checks. New-code lint passes; any inherited diagnostics
are listed verbatim and confirmed in the pinned base. The full build check
reports the promoted build directory is not prepared locally. All three patches
also apply together with ordinary git am, ordered gather, reduction, sampler;
that composition check is a source check, not a combined GPU qualification.

Reproduce normalization without GPU access:

```sh
python3 experiments/2026-11-04-rank-order-strides/analyze.py \
  experiments/2026-11-04-rank-order-strides/runs/native-v1.json
```

[Review PR](https://github.com/christopherowen/spark-ds41f/pull/886).
Retain as a candidate for review. No promoted recipe, lock or immutable release
receipt is changed. Build-image and real-model serving qualification remain.
