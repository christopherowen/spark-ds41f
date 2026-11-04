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

Results pending.
