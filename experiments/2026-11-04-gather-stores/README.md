# gather-stores

One variable: the gather source patch in vllm/series. Base and candidate identities
are in base.json and candidate.json. Apply after the exact candidate-38 stack;
no promoted configuration, lock, baseline or serving runtime is changed.

Source patch includes its upstream base, rationale, quality implications, tests
and local upstream status. The overlay is an exact Git export of the patch tree,
not an authority independent of the patch. Every test first checks original source
hashes in the frozen image, then verifies all patched file hashes before import.

Run on authorized idle sp8 from a clean published checkout:

```sh
python3 experiments/2026-11-04-gather-stores/run.py \
  --experiments experiments/2026-11-04-gather-stores --run gather-v1
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

The compressor and sharded vision gather rank-major rows, then one TileLang
kernel performs concatenation, unpadding and final one/two-buffer stores.
Both scalar and aligned-vector paths are declared before freeze. Collective
unaligned rows additionally use an opaque byte-copy kernel instead of returning
to the communicator's PyTorch movedim/reshape copy. Payload bits, including
complex tail payloads, are preserved. Inputs and outputs must be disjoint;
output rows that overlap are rejected before launch.

Both final boots pass **74 tests**, with no skips or failures. The two
benchmark boots each pass **28 independent-reference cases** and retain 21
cold-L2 graph samples per method per case. CUDA profiler traces contain the
expected TileLang candidate kernel and no ATen kernel for the captured replacement
operation. The original references show the ATen work being replaced; user scope
annotations in the raw event lists are not GPU kernels. Traces and logs stay on
sp8, with hashes and exact commands in the portable receipts.

Across 28 paired timing cases, ratios of boot means range from -50.25% to +0.42% candidate time. Every result is retained; this is not a blanket speedup verdict.

The gather timing reference is PyNCCL's existing TileLang concatenation plus
persistent-buffer copies, excluding communication. Sparknet can perform layout
work inside its networking kernel, so these numbers do not qualify its complete
path. The largest mean increase in the final matrix is 0.42%; the largest
reduction is 50.25%. The new dim-zero request can change transport dispatch;
actual NIC/proxy traffic and distributed serving remain required before promotion.
The TP3/TP4 vision checks emulate ranks with threads on one GPU.

`results.json` retains both boot medians and 95% Student t intervals (df=1).
Two boots give limited generalization; ratios are descriptive without a
significance claim. Source overlays are isolated kernel/model-unit tests, not a
new OCI build or full-model serving qualification.

The initial source variant passed 73 tests twice and 24 timing cases twice.
Those receipts remain in runs/native-v1.json with their original source identities.
The final overlap guard adds one test; its fresh run is runs/native-v2.json, with
74 tests and 28 timing cases per boot, plus a separate tail-layout inventory.
No earlier receipt was relabeled as the final source.

## Evidence and local checks

The final native run is `/home/chris/spark-lab/aten-runtime-20261104/replacements-v2`, published deployment source
`9929a8ad464a6fec764e6e29b2d128de7196f601`. This checkout aggregates the three experiments for
back-to-back scheduling; each fresh test container applies only its own source
patch. Raw source and patched file hashes are checked before each import.
The minimum sampled host MemAvailable for this experiment is 110.39 GiB.
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
python3 experiments/2026-11-04-gather-stores/analyze.py \
  experiments/2026-11-04-gather-stores/runs/native-v2.json
```

[Review PR](https://github.com/christopherowen/spark-ds41f/pull/887).
Retain as a candidate for review. No promoted recipe, lock or immutable release
receipt is changed. Build-image and real-model serving qualification remain.
