# Image build

`docker/Dockerfile` builds the promoted image: Local Inference Lab's
karmic-kraken-beta vLLM and B12X (while the lock lists it; see
[Locks without B12X](#locks-without-b12x)) with the local patch series, plus TileLang,
DeepSeek's TileKernels and sparknet for the promoted TileLang kernel family
(the `runtime-tilelang-sparknet` stage), on the canonical
vLLM ARM64 nightly `af1c0149`. Only vLLM's `_C_stable_libtorch` and
`_moe_C_stable_libtorch` are rebuilt, for SM121, and NCCL 2.30.7 (the base
image's version) is rebuilt from its release tag with `patches/nccl`, replacing
the wheel's `libnccl.so.2`; the image build checks the installed library's
version and checksum. The CUDA toolkit is the base image's CUDA 13.0.2 unless the lock
selects another ([CUDA toolkit](#cuda-toolkit)). FlashInfer 0.6.18.post1 comes from the
base image. The r1 and r2 images came from the same recipe in
`experiments/2026-09-23-karmic-kraken-reference/`; r3 was the first built by
`bin/spark build`. Images are tagged `spark-ds41f:<vLLM base>-<release>` (for example
`spark-ds41f:19f2c20ed4d6-MMDD`); images before the release after spark-ds41f-1022 were
tagged `vllm-ds41f-kkref:*`, and their tags stay as built.

```sh
bin/spark build prepare        # create or repair the build directory
bin/spark build check          # verify it
bin/spark build image          # print the build command
bin/spark build image --apply  # build and smoke-test on an idle host
bin/spark build smoke          # rerun the GPU import smoke
```

## Build directory

`upstreams.lock.json` and the source manifest it names determine every build
input:

- the vLLM and NCCL revisions, and B12X's while the lock lists it;
- the patch-series fingerprints, recorded patch heads, and trees;
- the CUTLASS revision;
- the CuTe DSL wheel lock;
- the CUDA toolkit package lock, when the lock lists `cuda_toolkit`;
- the base image digest.

A hash of those inputs names the directory:

```
.work/build/vllm-<12>-b12x-<12>-<input hash>/
  inputs.json            # the inputs and a content digest of each context
  src/vllm, src/b12x, src/nccl  # pinned revision + git am of patches/*/series
  src/cutlass            # pinned CUTLASS revision (headers only)
  context/vllm-source    # clean exports: no .git, no bytecode
  context/b12x-source
  context/nccl-source
  context/cutlass-source
  context/cutlass-dsl-wheels   # hash-verified wheels
  context/cuda-toolkit-debs    # hash-verified .deb files (a lock with cuda_toolkit)
  context/vllm-deletions/deleted.txt
  context/empty          # the main context; every COPY uses a named context
  images/<image id>.json # receipt for each image built from it
```

A lock that lists the optional `tilelang` and `tile_kernels` sources adds
`src/tilelang` (with its submodules), `src/tile_kernels` and their
`*-source` contexts; see [TileLang kernel backend](#tilelang-kernel-backend).

The same lock always yields the same directory with the same contents.

**Patch heads:** `git am` runs with a fixed committer and
`--committer-date-is-author-date`, so the patched heads reproduce the source
manifest's `patch_head` commits exactly. Prepare refuses a head or tree that
differs from the manifest. The patch-set fingerprint must also match the
manifest, so a changed patch has to be recorded before it can be built.

**Reuse and repair:** re-running `prepare` reuses every verified part and
rebuilds only what is missing or wrong. It writes `inputs.json` last, so an
interrupted run is never mistaken for a complete one. `check` recomputes each
context's digest.

**Measured evidence:** `prepare --lock` refreshes build identifiers in the lock,
source manifest and arm configurations. It leaves results and verdict receipts
unchanged, including their source identifiers, image digests and notes. A source
change needs new test and benchmark receipts for that build; an earlier passing
result does not qualify it. Preserve the earlier run and its identity instead of
editing its identifiers to match the new source. The source-lock tests flag a
candidate summary that still names a different source from its current manifest.

**Contexts:** these are exports without Git metadata. File modes are
normalized to Git's 644 and 755, so the contexts don't depend on the host
umask and are identical on every host. The earlier recipe copied whole
checkouts, which put about 100 MB of `.git` history into the image and
changed the build cache key on every fresh clone.

**CI:** CI runs `bin/spark build prepare --only vllm` and `--only b12x` to
check that the series still apply and reproduce the recorded trees, and
`--only vllm,tilelang,tile_kernels` for the TileLang candidate.

**Missing patches:** a series entry whose patch file does not exist yet stops
`prepare`, `check` and `image` with the missing path. `prepare --only` still
prepares the other parts.

## TileLang kernel backend

For a lock that lists `tilelang` and `tile_kernels`
([scripts/kernel_backend.py](../scripts/kernel_backend.py)), `prepare` checks out
TileLang's submodules recursively at the commits the patched tree records and
refuses commits that differ from the source manifest. `build image` passes
their contexts, `TILELANG_*` and `TILE_KERNELS_*` arguments (revision, patch
head, tree and version), the compile jobs as `TILELANG_COMPILE_JOBS`, and
`RUNTIME_STAGE=runtime-tilelang`. That stage starts from the B12X runtime
stage (`TILELANG_BASE_STAGE=runtime`; `runtime-core` for a lock without B12X),
builds wheels of both, replaces the base image's `tilelang` 0.1.12,
installs TileKernels without dependencies, and checks the installed files,
versions and every shared `apache-tvm-ffi` and `tilelang` requirement. The
image gains `local.spark-ds41f.tilelang.*` and `local.spark-ds41f.tile_kernels.*`
labels, and the GPU smoke imports both packages.

Without those sources the command has no new argument, BuildKit skips the
TileLang stages and their contexts, and the image is the unchanged `runtime`
stage.

## CUDA toolkit

Every builder stage and `runtime-core` start from `CUDA_STAGE`, whose default `base` is
the base image as it is: CUDA 13.0.2 (`nvcc`/`ptxas` 13.0.88, `libcublasLt` 13.1.1.3).
A lock without `sources.cuda_toolkit` (every lock so far, spark-ds41f-1107 and
candidate-54 included) builds exactly as before: its inputs, build directory and `build
image` command are unchanged (`tests/test_build.py` compares the command with the one
recorded before the option existed), and BuildKit builds neither CUDA 13.5 stage nor
reads the `cuda-toolkit-debs` context.

A lock that lists the toolkit builds with CUDA 13.5:

```json
"cuda_toolkit": {
  "kind": "deb",
  "upstream": "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa",
  "version": "13.5.37",
  "lockfile": "requirements/cuda-13-5-sbsa.txt",
  "role": "CUDA 13.5 toolchain and math libraries over the base image's CUDA 13.0.2"
}
```

- `requirements/cuda-13-5-sbsa.txt` lists every package the stage installs, one
  `package=version filename size sha256:<hex>` line each: the 21 toolchain, runtime and
  math-library packages (`cuda-nvcc-13-5` 13.5.37-1, `cuda-cudart-13-5` 13.5.37-1,
  `libcublas-13-5` 13.10.0.20-1, ...) and `cuda-toolkit-config-common` and
  `cuda-toolkit-13-config-common` at 13.5.37-1, which they need upgraded from the base
  image's 13.0.96-1. `version` must be one the Dockerfile has a stage for
  (`CUDA_TOOLKIT_STAGES` in `spark_ds41f/build.py`: only 13.5.37), and the lockfile must
  list its `cuda-nvcc` at that version; `bin/spark doctor` and every build command
  refuse a malformed entry or lockfile.
- `prepare` downloads each file from `upstream` into `context/cuda-toolkit-debs`, keeps
  the files that already match the lock's size and SHA-256, and refuses one that does
  not; `check` reports missing, unlocked and changed files. `prepare --only
  cuda_toolkit` prepares just these.
- `build image` passes the context, `CUDA_STAGE=cuda-13-5` and
  `RUNTIME_CORE_STAGE=runtime-core-cuda-13-5`. Stage `cuda-13-5` installs exactly the
  locked files with `dpkg -i`, without the network, so a dependency the set lacks fails
  the build; it wires `/usr/local/cuda`, `/usr/local/cuda-13`, the alternatives,
  `nvcc.profile`'s `TOP` and the pip `nvidia/cu13` `bin`, `nvvm` and libraries (which
  `libtorch_cuda.so`'s `DT_RPATH` and `cuda.pathfinder` search ahead of
  `/usr/local/cuda`) to `/usr/local/cuda-13.5` with `/usr/local/sbin/spark-wire-cuda-13-5`,
  and sets `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH` and the CUDA version variables.
  `runtime-core-cuda-13-5` runs the wiring again after `runtime-core`'s CuTe DSL wheel
  reinstall, checks `nvcc`/`ptxas` 13.5.37 (also as `cuda.pathfinder` finds them),
  `libcublasLt` 131000 through both paths, the `nvidia/cu13` libraries and `libcudart`
  13050, and labels the image `local.spark-ds41f.cuda.version=13.5.37`, which
  `build image --apply` checks; the B12X and TileLang runtimes build on it.

To refresh the lockfile, list what apt would install in the base image (no payload is
downloaded) and take each file's size and SHA-256 from the repository's signed index:

```sh
image=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
docker run --rm --network host --user root --entrypoint /bin/sh "$image" -c '
  apt-get update
  apt-get install --no-install-recommends --dry-run PACKAGE=VERSION ...  # every Inst line goes in the lock
  for p in PACKAGE=VERSION ...; do                                        # each Inst line, pinned
    apt-cache show --no-all-versions "$p" | grep -E "^(Package|Version|Architecture|Filename|Size|SHA256):"
  done'
```

Write one line per package, its `Filename` without the leading `./`, and record the
index date (the `Date` of the repository's `InRelease`) in the header. `apt-get download
--print-uris` prints the same URLs and sizes with SHA-512. A new toolkit version also
needs its own Dockerfile stages and `CUDA_TOOLKIT_STAGES` entry.

## Torch

The base image's torch (2.13.0+cu130) stays unless a lock lists `sources.torch` beside its
CUDA toolkit; a lock without it builds exactly as before, with no torch stage and no
`torch-wheels` context.

```json
"torch": {
  "kind": "wheel",
  "upstream": "https://download.pytorch.org/whl/nightly/cu135",
  "version": "2.16.0.dev20261113+cu135",
  "lockfile": "requirements/torch-2-16-cu135-aarch64.txt",
  "role": "torch compiled for the lock's CUDA 13.5 toolkit"
}
```

- `requirements/torch-2-16-cu135-aarch64.txt` lists every wheel the stage installs, one
  `filename size sha256:<hex> url` line each: torch, the 16 packages pip resolves for it,
  and torchvision, torchaudio, torchcodec and triton. `bin/spark build prepare` (or
  `--only torch`) downloads each from its URL and verifies it.
- `version` must be one the Dockerfile has stages for (`TORCH_STAGES` in
  `spark_ds41f/build.py`), and the lock's `cuda_toolkit` must be the toolkit that torch was
  compiled for.
- `cuda-13-5-torch-2-16` installs the set over the CUDA 13.5 stage with no index and no
  dependency resolution, and becomes `CUDA_STAGE`: vLLM's extensions, NCCL, TileLang and
  sparknet are all built against this torch.
- torch binds NCCL's symbols when `libtorch_cuda.so` loads, so the lock's NCCL must be the
  release torch was compiled against (2.32.3 here). The build passes the tag's
  `NCCL_VERSION_CODE`, and `runtime-core-cuda-13-5-torch-2-16` checks that torch reports
  that release, loads the patched library and no other, and that torchvision runs.
- The image carries `local.spark-ds41f.torch.version`, which `build image` compares with
  the lock.

## Locks without B12X

`runtime-core` holds everything but B12X (the vLLM tree and its two rebuilt
extensions, NCCL, the CuTe DSL wheels, liburing for the io_uring readers vLLM
builds at first use); `runtime` adds B12X on it, with its labels and its checks. A
lock that lists `b12x` (spark-ds41f-1025 and earlier) builds as before, in the
same `vllm-<12>-b12x-<12>-<input hash>` directory. From the release after next the
lock leaves B12X out: `prepare` makes no `src/b12x` or `context/b12x-source` (the
directory is `vllm-<12>-<input hash>`), `build image` passes
`TILELANG_BASE_STAGE=runtime-core`, so BuildKit never reads a `b12x-source`
context, the image has no `local.spark-ds41f.b12x.*` labels, and the GPU smoke
imports no B12X module and checks that B12X is not installed. A lock without B12X
must list the TileLang sources ([docs/upstreams.md](../docs/upstreams.md#locks-without-b12x)).

## Without FlashInfer's precompiled kernels

An experiment's arm with `"build": {"flashinfer_kernels": false}` builds
`RUNTIME_STAGE=runtime-no-flashinfer-kernels`: the `runtime-tilelang-sparknet`
image with the base image's flashinfer-cubin and flashinfer-jit-cache uninstalled
and checked absent, labelled `local.spark-ds41f.flashinfer-kernels=removed`.
flashinfer-python stays, since vLLM imports it. The image is no smaller, because
the base image's layers still hold the files. Any other runtime is refused
([experiments/2026-10-27-no-flashinfer-kernels](../experiments/2026-10-27-no-flashinfer-kernels/README.md)).

## Resource policy on DGX Spark

`build image --apply` runs only on an idle Spark:

- It refuses while a DS4.1 container is running and holds a host lock against
  overlapping builds.
- It needs 64 GiB MemAvailable to start.
- It picks compile jobs from memory and CPUs: 8 GiB per job, 24 GiB kept in
  reserve, four CPUs reserved, two NVCC threads per job, at most ten jobs.
- It samples MemAvailable every second and cancels the build below 24 GiB.

`build image --max-compile-jobs N` can lower the computed compilation parallelism
for vLLM, TileLang and NCCL; it never raises the host-derived limit or relaxes
the memory floor. With no option, the existing sizing and NCCL's four-job
Dockerfile default stay unchanged. The receipt records the requested cap and
effective job counts; with the option, its separate16GiB/no-extra-swap GPU smoke
also has a CPU limit of at most two. This caps compiler jobs, not the Docker daemon's resources;
an experiment that needs a hard compilation CPU/memory bound must also select
and verify a resource-limited BuildKit worker.

After the build it checks the tree labels, then imports the DS4.1 serving
modules on the GPU in a 16 GiB container. It writes a receipt with the image
ID, job count, elapsed time, and lowest MemAvailable.

The default tag is `container.image` from the cluster configuration. The
command refuses to overwrite an existing tag, so a node never silently holds a
different image under the promoted name. Build once, then load the same image
on every node and confirm every node reports the same ID ([docs/setup.md](../docs/setup.md#6-image)).
Start the service only through `bin/spark cluster start`, with its startup and
steady memory guards.
