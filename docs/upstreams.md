# Upstreams and contribution flow

`upstreams.lock.json` is the promoted machine-readable authority. An experimental
cluster profile may select a separate repository-relative lock through
`upstreams_config`; build preparation, doctor and image builds use that lock.
This lets candidates add patches without changing the promoted image identity.
The production chain (revisions and series in `upstreams.lock.json`; the image in
`config/cluster.json`) is:

```text
local-inference-lab/vllm integration/karmic-kraken-beta
        (canonical base vllm-project/vllm, parked)
        + the release's vLLM series (Engram projection TP padding and
          asynchronous rows, multimodal block hashes, DSpark verification
          rules, sequence-parallel prefill, explicit collective policy,
          packed BF16 output head, display carve-out weights, the TileLang
          kernel family for every serving kernel, sparknet collectives,
          the SPARK_DS41F_* variables)
                    \
local-inference-lab/b12x integration/karmic-kraken-beta (archived; its last commit)
        + its series; serving uses only its checkpoint loader
          (--load-format b12x) and the modules vLLM still imports
          (spark-ds41f-1025; locks from the release after next leave
          B12X out, see "Locks without B12X")
                    |
tile-ai/tilelang main (0.1.15)
        + patches (SM120/SM121 block-scaled MMA: MXFP8, FP8 x FP4, MXFP4)
deepseek-ai/TileKernels main (2.0.0) + patches/tile_kernels
christopherowen/dgx-spark-networking (sparknet)
                     ---- spark-ds41f:<vLLM base>-<release>
                    /
NVIDIA/nccl v2.30.7-1
        + its series (IB send-path fence, bidirectional switchless
          rings, balanced channels), replacing the base
          image's nvidia-nccl-cu13 libnccl.so.2

deepseek-ai/DeepSeek-V4.1-Flash-1017 (the recipe's model; unchanged, and TP3
        head padding is in the serving source, so no config overlay is mounted)
```

The official vLLM nightly at `af1c0149` supplies the CUDA/runtime base, including
FlashInfer 0.6.18.post1; only vLLM's two stable CUDA extensions are rebuilt for
SM121 against CUTLASS v4.7.1. Canonical vLLM main is parked while the Local
Inference Lab integration branch is the serving source.

## Source map

| Lock key | Canonical upstream | Contribution fork | Purpose |
|---|---|---|---|
| `vllm` | `local-inference-lab/vllm` (`integration/karmic-kraken-beta`) | `christopherowen/vllm` | serving source; canonical `vllm-project/vllm` main is parked |
| `vllm_base_image` | `docker.io/vllm/vllm-openai` | not applicable | official CUDA/Torch/native-extension foundation |
| `b12x` | `local-inference-lab/b12x` (`integration/karmic-kraken-beta`, archived) | none (archived) | spark-ds41f-1025 and earlier: the checkpoint loader (`--load-format b12x`) and modules vLLM still imports; no B12X kernel runs in serving. Not listed from the release after next |
| `nccl` | `NVIDIA/nccl` (tag `v2.30.7-1`) | not created yet | the base image's NCCL version, rebuilt for SM121 with `patches/nccl` |
| `tilelang` | `tile-ai/tilelang` (`main`) | `christopherowen/tilelang` | the promoted kernel compiler, with `patches/tilelang` for SM120/SM121 |
| `tile_kernels` | `deepseek-ai/TileKernels` (`main`) | not created yet | DeepSeek's TileLang kernels: MoE gate, casts, mHC |
| `sparknet` | `christopherowen/dgx-spark-networking` (`main`) | not applicable | one-shot RoCE collectives for the TileLang family |
| `flashinfer` | `flashinfer-ai/flashinfer` | not created yet | not a promoted build input (the runtime base supplies 0.6.18.post1) |
| `cutlass` | `NVIDIA/cutlass` | not created yet | SM121 stable-extension headers at `v4.7.1` |
| `cutlass_dsl` | NVIDIA packages on PyPI | not applicable | SHA-256-locked ARM64 CuTe DSL wheel set |
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash` | not applicable | unchanged native model weights and tokenizer |

The URL and full immutable revision in `upstreams.lock.json`, not a directory
name or Docker tag, define identity. The `tracking_ref` says which canonical
branch to inspect for future pulls; it does not float the build.

## Refreshing an upstream

Prepare an isolated tree:

```sh
bin/spark upstream prepare vllm
bin/spark upstream prepare b12x
```

The helper clones the canonical repository as `upstream`, adds Christopher's fork
as `origin` when one exists, checks out the full pinned SHA, initializes pinned
submodules where required, applies recorded upstream commits, and then applies
the local series. Worktrees live under ignored `.work/upstreams/`.

To evaluate a newer upstream, create a branch in that prepared tree, rebase the
small patch series, build a new immutable image tag, and record it as an experiment.
Do not advance `upstreams.lock.json` merely because a newer commit exists.

Shared runtime packages move forward, never back. CuTe DSL is used by vLLM,
quack-kernels, FlashInfer, the cuDNN frontend and B12X (while a lock lists it); the
locked wheel set in
`requirements/cutlass-dsl-aarch64.txt` must satisfy every one of them and must
not be older than the version the vLLM base image installs. When a component
pins an older version, patch the component's pin forward and qualify it rather
than downgrading the others. The image build checks each consumer's declared
requirement and fails on a mismatch.

## Locks without B12X

From the release after next, the lock has no `b12x` entry: its vLLM series ports what
serving still took from B12X (the direct loader, the packed BF16 head's packer, Engram's
token map, geometry and caps, and a TileLang gather for resident Engram tables;
[experiments/2026-10-26-retire-b12x](../experiments/2026-10-26-retire-b12x/README.md)).
`bin/spark build` handles both kinds of lock:

- A lock that lists `b12x` (spark-ds41f-1025 and earlier) builds exactly as before: the
  same build directory, the `b12x-source` context, Dockerfile stage `runtime` (B12X on
  `runtime-core`) under the TileLang runtime, and the `local.spark-ds41f.b12x.*` labels.
- A lock without it prepares no B12X project or context, builds the TileLang runtime on
  `runtime-core` (`TILELANG_BASE_STAGE`), records no B12X labels, and its image smoke
  checks that B12X is not installed. Such a lock must list the TileLang sources.

To make such a lock, drop `sources.b12x` from the candidate's lock (the source manifest's
`b12x` record is then not read) and `local.spark-ds41f.b12x.tree` from its profile's
`container.expected_labels`. The recipe loads with `--load-format direct` and an empty
`VLLM_PLUGINS`.

## Contributing back

One conceptual local patch should become one upstream branch and pull request.
Preserve tests and benchmark evidence alongside it. After acceptance:

1. advance the pinned upstream revision to include the merged commit;
2. remove the redundant local patch from `series`;
3. rebuild once and distribute the same digest;
4. reproduce the accepted baseline before promotion.

B12X is archived (the lock pins its last commit), so it gets no contribution fork
and its `contribution_remote` stays `null`. FlashInfer and CUTLASS have no fork yet:
add one only when creating the first contribution, and do not invent a fork URL or use
this deployment repository as their `origin`.

The GLM switchless repository is recorded as a research reference, not an
upstream. Ideas may be reimplemented and measured, but its topology- and
model-specific patches do not enter our patch stack implicitly.

The complete research-reference watchlist, review questions, and adoption gates
are maintained in [inspiration.md](inspiration.md). Entries under `references` in
`upstreams.lock.json` are deliberately excluded from build preparation.
