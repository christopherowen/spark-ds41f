# vLLM 0.32 native TileLang port

Candidate based on spark-ds41f `6aed292604df01f1c521e828501206f5f32c758b`.
The promoted r6x recipe, source locks and baseline records stay unchanged.
This ports the existing native SM12x family onto official
[vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0),
revision `5df63a7203cdb9328d824a27b682e77c7b9928be`.

## What is retained

The port keeps r6x's TileLang attention, linear, MoE, mHC, vocabulary and
sparknet collectives, its MXFP4 KV/indexer records (288/68 bytes), native
Engram disk reader, interleaved column shards and ring relay, WO-A activation
quantization, FP32 logits, exact routed-expert scale codes, routed SwiGLU,
kernel preparation and adaptive DSpark row cost 0.035. Native greedy
verification works with the V2 runner, including request constraints and
bounded query padding. General sampling keeps the upstream sampler.

The patch dispatches to the native family only for CUDA SM12x with the
TileLang backend. Other devices and backends keep the official model entry.
It does not apply the separate broadcast or rejection-sink experiments.

## Compatibility changes

- Use the official image's matching native extensions and CuTe DSL 4.8.0.
- Install hash-locked B12X 1.5.1 and FlashInfer 0.7.1 wheels. Build the existing
  TileLang, TileKernels, sparknet and patched NCCL sources from their pins.
- Load through `native_b12x` with lazy checkpoint metadata. The loader routes
  mixed shards without materializing Engram tables, and registers their original
  checkpoint paths and byte offsets with the disk reader.
- Select `disk_offload` in the new Engram config and MXFP4 indexer cache in the
  attention config. Use V2, compilation mode `NONE` and `FULL_DECODE_ONLY` graphs.
- Remove obsolete prefill, graph-size and JIT-monitor flags and AOT environment
  settings. Enable the native minimum-token constraint mask explicitly.

Normal per-node checkpoint mounts and upstream fabric/host guards are retained.
There are no shared-weight services, site-local Engram plans, external NCCL
libraries, router hooks or private host/address mappings in this candidate.
The direct loader supports the native DP1 TileLang path; EP and encoder-only
loading are rejected. TP3/TP4 configuration checks pass, but this candidate
recipe targets TP4 and does not inherit TP3 serving or memory qualification.

## Prepare and build

From a checkout configured for the site's normal upstream node map:

```sh
profile=experiments/2026-10-20-vllm032/cluster.json
bin/spark --cluster-config "$profile" doctor
bin/spark --cluster-config "$profile" build prepare
bin/spark --cluster-config "$profile" build check
bin/spark --cluster-config "$profile" build image
bin/spark --cluster-config "$profile" build image --apply
```

The last command requires an idle ARM64 builder with at least 64 GiB available
and runs the existing build guard and GPU import smoke. The candidate lock
selects `docker/Dockerfile.vllm032`; wheel hashes and the Dockerfile hash are
part of build identity. The image is built once and distributed to every rank.
A serving trial still needs a coordinated window and the usual quality/release
checks before promotion. No new baseline or performance claim is made here.

## Validation scope

The public patch is replayed from the official release with the build helper's
fixed committer, checking both commit and tree identity. Repository unit tests,
doctor, prepared-input checks, Docker build checks and the exact native GPU
import smoke cover packaging.
The repository suite passes 454 tests. The focused source suite passes 228
tests, with the existing nonfinite-indexer-query-scale contract marked as an
expected failure. It covers checkpoint routing, stock file offsets,
configuration rejection, greedy verification, DFlash inputs, dead rows,
rejection chunking, preparation, attention, MoE and vocabulary kernels.

These source tests used a compatible ARM64 CUDA environment with B12X 1.5.1,
FlashInfer 0.7.1 and the patched TileLang compiler. The public Docker recipe
has not completed an uncached image build or a full multi-node serving release
gate. The related site deployment uses different storage and communication
settings, so its serving measurements do not qualify this generic recipe.
