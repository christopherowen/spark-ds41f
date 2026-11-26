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
        + the release's vLLM series (TileLang kernels end to end, direct
          O_DIRECT checkpoint loader, contiguous CUDA pool, zero-copy
          io_uring expert slot cache on TP1/TP2, PP=2 pipeline parallelism
          and intra-stage SP, Engram projection TP padding and native
          async rows, multimodal block hashes and TileLang vision tower,
          DSpark + CopyDrafter adaptive verification, sequence-parallel
          prefill, packed BF16 output head, sparknet collectives and P2P,
          GLP steering, the SPARK_DS41F_* variables)
                    \
docker.io/vllm/vllm-openai (nightly base)
        + hash-locked CUDA 13.5.37 toolkit & math libraries (over 13.0.2)
        + CUTLASS v4.7.1 headers (SM121 stable CUDA extension rebuild)
                    |
tile-ai/tilelang main (0.1.15)
        + patches (SM120/SM121 block-scaled MMA: MXFP8, FP8 x FP4, MXFP4,
          device-compiler cache key)
deepseek-ai/TileKernels main (2.0.0) + patches/tile_kernels
christopherowen/dgx-spark-networking (sparknet v0.6.0, zero local patches:
        one-shot RoCE collectives, one-shot PP P2P, proxy ABI 10)
                     ---- spark-ds41f:<release>
                    /
NVIDIA/nccl v2.30.7-1
        + its series (IB send-path fence, bidirectional switchless
          rings, balanced channels), replacing the base
          image's nvidia-nccl-cu13 libnccl.so.2

deepseek-ai/DeepSeek-V4.1-Flash-1017 (the recipe's model; unchanged native
        FP8/FP4/BF16 weights; uneven TP shard padding is in serving source)
msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5
        (optional pinned GLP steering vector for deepseek-v4.1-flash-weightless)
```

The official vLLM nightly at `af1c0149` supplies the runtime Python/Torch base,
upgraded with the hash-locked CUDA `13.5.37` toolkit layer; only vLLM's two
stable CUDA extensions are rebuilt for SM121 against CUTLASS `v4.7.1`. Canonical
vLLM main is parked while the Local Inference Lab integration branch is the
serving source. B12X (`local-inference-lab/b12x`) was completely retired in
`spark-ds41f-1120` (`vLLM 0802`) and is no longer listed in promoted locks.

## Source map

| Lock key | Canonical upstream | Contribution fork | Purpose |
|---|---|---|---|
| `vllm` | `local-inference-lab/vllm` (`integration/karmic-kraken-beta`) | `christopherowen/vllm` | serving source; canonical `vllm-project/vllm` main is parked |
| `vllm_base_image` | `docker.io/vllm/vllm-openai` | not applicable | official PyTorch/runtime foundation |
| `cuda_toolkit` | NVIDIA's Ubuntu 24.04 SBSA CUDA apt repository | not applicable | SHA-256-locked CUDA `13.5.37` toolchain and math-library `.deb` layer over the base image's CUDA `13.0.2` (active in promoted locks since `spark-ds41f-1120`; [docker/README.md](../docker/README.md#cuda-toolkit)) |
| `tilelang` | `tile-ai/tilelang` (`main`) | `christopherowen/tilelang` | the promoted kernel compiler (`0.1.15`), with `patches/tilelang` for SM120/SM121 |
| `tile_kernels` | `deepseek-ai/TileKernels` (`main`) | not created yet | DeepSeek's TileLang kernels (`2.0.0`): MoE gate, casts, mHC |
| `sparknet` | `christopherowen/dgx-spark-networking` (`v0.6.0` / `main`) | not applicable | one-shot RoCE collectives, one-shot PP P2P, and proxy ABI 10 for the TileLang family (`0` local patches from `v0.6.0`) |
| `nccl` | `NVIDIA/nccl` (tag `v2.30.7-1`) | not created yet | the base image's NCCL version, rebuilt for SM121 with `patches/nccl` |
| `cutlass` | `NVIDIA/cutlass` (`v4.7.1`) | not created yet | SM121 stable-extension headers at `v4.7.1` |
| `cutlass_dsl` | NVIDIA packages on PyPI | not applicable | SHA-256-locked ARM64 CuTe DSL wheel set |
| `torch` | PyTorch's wheel index (`download.pytorch.org/whl/nightly/cu135`) | not applicable | optional, listed only by a lock that also lists the CUDA toolkit it was compiled for: SHA-256-locked wheels (`requirements/torch-2-16-cu135-aarch64.txt`) |
| `flashinfer` | `flashinfer-ai/flashinfer` | not created yet | not a promoted build input (the runtime base supplies `0.6.18.post1`); tracked for upstream review only |
| `b12x` *(retired)* | `local-inference-lab/b12x` (`integration/karmic-kraken-beta`, archived) | none (archived) | retired in `spark-ds41f-1120` (`vLLM 0802`); present only in historical locks (`spark-ds41f-1025` and earlier) |
| `model` | `deepseek-ai/DeepSeek-V4.1-Flash` (`1017` revision) | not applicable | unchanged native FP8/FP4/BF16 model weights and tokenizer |

Three additional pinned dependencies live outside the image build lock because they are
host-side tools or optional runtime weights rather than container image inputs:

- [Sonar](https://github.com/christopherowen/Sonar) (`christopherowen/Sonar`, `main`): the
  benchmark client, pinned in `config/sonar.json` by URL and full revision, checked out on
  first use under `.work/sonar/<revision>`, and shared across every arm a run compares. To
  move the pin, merge the change into Sonar, set the new revision in `config/sonar.json`,
  and record which outputs (suite digests, request payloads, result formats) stay unchanged.
- [dgx-spark-memory-saver](https://github.com/christopherowen/dgx-spark-memory-saver): the
  signed DKMS kernel module required on 64 KiB DGX Spark profiles for UVM leaf-table packing
  ([setup.md](setup.md#2-kernel-64-kib-profiles)).
- **GLP steering vector** (`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5`):
  pinned by repository, commit revision, filename, and SHA-256 in `config/cluster.json` under
  `glp` to serve `deepseek-v4.1-flash-weightless` beside the unchanged stock model.

The URL and full immutable revision in `upstreams.lock.json`, not a directory
name or Docker tag, define identity. The `tracking_ref` says which canonical
branch to inspect for future pulls; it does not float the build.

## Refreshing an upstream

Prepare an isolated tree:

```sh
bin/spark upstream prepare vllm
bin/spark upstream prepare tilelang
bin/spark upstream prepare sparknet
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

## Patched release arms

Published release manifests (`manifests/sources/`, `releases/*/source.json`, and
`upstreams.lock.json`) are immutable provenance records. Extending a release profile
with only a rebuilt `container.image` and updated `expected_labels` makes
`build distribute` and `doctor` disagree: `distribute` reads `expected_labels`
from the profile while `doctor` checks the release's `source.json`.

To ship a release plus one or more local patches so `build prepare`, `build image`,
`build distribute`, `cluster sync`, and `doctor` all agree:

1. Create an arm series in `experiments/<dir>/<project>/series` using `include`
   to compose the base series with the extra patch(es):

   ```text
   include ../../../releases/<release>/vllm/series
   0001-local-fix.patch
   ```

2. Copy the release's `upstreams.lock.json` and `source.json` into the arm
   directory, point `source_manifest` and `patch_series` at the arm copies, and set
   `upstreams_config` in the arm profile overlay.
3. Run `bin/spark --cluster-config <profile> build prepare --lock` to compute and
   align `patchset_sha256`, `patch_head`, `expected_tree`, and
   `container.expected_labels` across the arm's lock, manifest, and profile.
4. Build and distribute normally with `bin/spark --cluster-config <profile> build image --apply`
   and `build distribute --apply`.

## Locks without B12X

Promoted locks since `spark-ds41f-1030` / `spark-ds41f-1120` omit `b12x` completely: the
vLLM series carries the direct `O_DIRECT` checkpoint loader (`--load-format direct`,
`VLLM_PLUGINS` empty), packed BF16 LM-head packer, Engram token map/geometry/caps, and
TileLang gathers for resident Engram tables
([experiments/2026-10-26-retire-b12x](../experiments/2026-10-26-retire-b12x/README.md);
B12X retired fully in `vLLM 0802`). `bin/spark build` supports both current and historical
locks:

- Current locks (without `b12x`) prepare no B12X project or context, build the TileLang
  runtime directly on `runtime-core` (`TILELANG_BASE_STAGE`), record no `b12x` labels, and
  verify during GPU image smoke that B12X is not installed.
- Historical locks (`spark-ds41f-1025` and earlier, which still list `b12x`) continue to
  build the `b12x-source` context and `local.spark-ds41f.b12x.*` labels for reproducibility.

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
