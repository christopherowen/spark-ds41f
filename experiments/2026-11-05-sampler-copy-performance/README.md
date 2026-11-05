# Sampler copy performance

This supersedes the earlier sampler-copy candidate's unresolved/regressing GPU
cases. Historical measurements remain unchanged in the November 4 experiment.
Base identity is the exact candidate-38 source in `base.json`; the ordered local
series adds the original replacement followed by the accepted correction.

One intended variable is the sampler copy implementation. Aligned casts and
small strided copies use four adjacent values per thread and a column-first
TileLang grid. Wide, 64-byte-aligned FP32 matrices of at least 8 MiB use native
pitched CUDA DMA on the current device/stream. Dense FP32 retains its native DMA.
Other strides use the general TileLang copy. No ATen compute fallback remains.
The independent wide-tile and scalar-copy regressions are retained in the combined
experiment and are not selected. Live graph payloads and side streams are tested.

Both final boots pass 120 tests each. All 18 GPU cases pass independently at a
five-standard-error upper ratio of at most one, or exact program/binding equality
for unchanged native DMA. The full-vocabulary 96-row padded FP32 copy is about
2% faster. `results.json` contains every gate and launch count. All three patched
files are byte-identical to the composed candidate, proved by
`source-equivalence.json`. No serving throughput or host-overhead claim follows.

Raw evidence, inputs, generated programs, profiler traces, commands and memory
observations are retained in `experiments/2026-11-05-tilelang-runtime-closure`
on branch `feat/tilelang-runtime-closure`, including rejected variants.
Final accepted run: `final-sampler-v6`, published deployment `4af4c960`.
Run from that clean published checkout on authorized idle sp8:

```sh
python3 experiments/2026-11-05-tilelang-runtime-closure/run.py \
  --experiments experiments/2026-11-05-tilelang-runtime-closure \
  --run UNIQUE --operations tests pair --family sampler
```

The generic runner takes an exclusive sp8 reservation, pre-cools, guards memory
and runs two boots. It verifies frozen image and candidate hashes before import.
No serving cluster was changed. A new built image, full serving and physical
transport qualification remain required before promotion.
