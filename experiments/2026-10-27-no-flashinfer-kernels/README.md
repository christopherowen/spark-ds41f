# Serving without FlashInfer's precompiled kernels

The vLLM base image carries FlashInfer 0.6.18.post1 as three packages: flashinfer-cubin
(6.3 GB), flashinfer-jit-cache (1.7 GB, 907 modules) and flashinfer-python (113 MB).
Nothing but vLLM requires flashinfer-python, and nothing requires the other two. On
SM12x, 1025 serves through TileLang: attention is `DeepseekV41TileLangAttention`, and
the V2 runner's top-k/top-p filtering is TileLang (vLLM 0436). The "Using FlashInfer for
top-p & top-k sampling" startup line comes from the V1 sampler, which is constructed but
not on the V2 runner's path. A live 1025 ring maps no FlashInfer library on any node
(2026-10-27, read with this experiment's probe), and has compiled no FlashInfer module
since it started; the four in `/cache` are FlashInfer 0.7.0rc1's, from older images.

The question: does 1025 serve the same tokens with the two kernel packages removed?

## The variant

`"build": {"flashinfer_kernels": false}` in an arm ends the image on
`runtime-no-flashinfer-kernels` (docker/Dockerfile): the TileLang runtime with sparknet,
with flashinfer-cubin and flashinfer-jit-cache uninstalled and checked absent;
flashinfer-python stays, since vLLM imports it. The image is labelled
`local.spark-ds41f.flashinfer-kernels=removed`. It is no smaller: the base image's
layers still hold the files, and shrinking it needs a final stage that copies the
runtime instead. Should FlashInfer be called after all, it compiles its module at run
time, so the probe below sees it.

- `1025-tp4.json`: the recipe at tp4-ring, unchanged (`spark-ds41f:19f2c20ed4d6-next-6`).
- `no-flashinfer-kernels-tp4.json`: the same lock and vLLM tree on
  `spark-ds41f:19f2c20ed4d6-next-6-no-flashinfer-kernels`.

## The run

`run.json`: sync, then build the variant on dgx4 (serving stopped) and copy it over the
CX7 links, then measure variant, 1025, variant again, each with the lean bench and:

- `flashinfer_probe.py`: seeded top-p 0.9, top-k 40, and top-k 20 with top-p 0.8, one
  at a time and eight at once (digests), then per node the FlashInfer libraries any
  process has mapped and the FlashInfer modules compiled since the container started;
- the wide determinism gate, the seeded sampled digests, the controls digests and probe,
  the image digests, and the startup lines.

The variant passes if it boots, its digests equal 1025's (each probe's one-at-a-time
digest, the determinism and controls digests), `flashinfer_used` is false on every arm,
and the bench is within noise of 1025's.

## Results

Not run yet.
