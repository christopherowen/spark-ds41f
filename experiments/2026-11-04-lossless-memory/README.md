# Lossless memory experiments

Base deployment: `3095e3f294bf65f5e65c6ba6dd2b21984abe5864`.
Reference: `spark-ds41f-1030`, checkpoint `DeepSeek-V4.1-Flash-1017`
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
The four investigations below are independent variables, not a combined arm.
Promoted configuration, published baselines and recorded reference runs remain unchanged.

## Required invariants

- Every encoded block reconstructs the original bytes, including signed zero,
  subnormals, special floating-point encodings, scales and padding that is read.
- Consumer arithmetic, reduction order, routing, target/draft weights,
  requested effort, sampling and prompt semantics remain unchanged.
- A byte round trip is necessary but does not qualify serving. The changed
  kernels must match the existing kernels bit for bit; greedy digests and
  teacher-forced distributions must match the same workload's recorded reference.
- A prefix checkpoint includes every required cache group, private sliding-window
  suffix, compressor state and target/draft auxiliary state. Only written regions
  may be exported, after a capture fence. A restored prefix becomes visible only
  after all required rank copies complete and admission owns the destination pages.
- Every failure is retained. No precision reduction, pruning, steering, cache
  approximation or sampled quality tolerance is an acceptable speed tradeoff.

## Variables and gates

| Investigation | One intended variable | First gate | Serving gate |
| --- | --- | --- | --- |
| FP8 | Independent N128/K128 weight-tile representation | Compare raw, LZ4, bitplane+LZ4, FP8-field+LZ4 and exact exponent Huffman; include all metadata | Decode within the consuming GEMM without a full main-memory expansion; identical products and accumulation order |
| Expert scales | Physical storage of existing 2/4-bit codes | Compact narrow experts; reconstruct each existing scale word and preserve contiguous loads | Persistent/peak memory and decode/prefill on two candidate boots |
| Engram | Co-location of row values and scales | One rank-local record, correct column map and page-boundary scatter | Actual NVMe reads, exposed waits, decode/prefill, startup packing and memory on two boots |
| Cold prefix | Storage representation of a complete inactive checkpoint | Bounded independent blocks; exact state/identity round trip; atomic publication and corruption refusal | Reuse after KV eviction; identical resumed target/draft state and outputs, TTFT, store/load cost and memory |

`memory_codec.py` is a CPU reference and screening implementation. It works on
integer byte encodings and never casts or rounds floating-point values. RAW is
an explicit format representation when coding does not reduce payload size;
codec errors are raised. Each independent block has a bounded decoded length
and SHA-256 verification. A serving-tile implementation needs its own layout
and GPU measurement; these reference CPU timings are not a GPU prediction.

`cold_store.py` exercises a disk container for already-fenced exports. Its
namespace includes checkpoint, tokenizer, serving tree, tensor-parallel rank,
cache layout and steering identity. Metadata and every record have integrity
checks; publication uses a private temporary file, fsync and atomic replacement.
The store has explicit decoded-size and disk-quota caps. It does **not** publish
prefix hits to vLLM and is **not** a qualified serving adapter. An assertion by
an input producer that its checkpoint is complete is not proof of a GPU fence.

## Existing implementations to reconcile

The newer candidate stack already contains compact expert scale storage
(`rc20/vllm/0595`, originally #727) and a packed Engram reader (originally #754,
with subsequent reader changes). The promoted 1030 recipe predates them.
Reuse their source and immutable receipts where applicable. Do not attribute a
whole candidate's throughput gain to one component, or rerun a released build.
The isolated expert-scale port here starts from the promoted vLLM tree rather
than including unrelated newer-candidate behavior.

Previously recorded component claims are hypotheses to audit: about 1.29 GiB
per TP3 rank from scale compaction, and fewer small-step NVMe reads from packed
Engram rows. They are not fresh serving results for this experiment. A source
revision or layout change needs new candidate receipts.

## CPU commands

Run the collector only against the exact immutable checkpoint. The collector
reads headers and a bounded set of rectangular tiles; it does not initialize
CUDA, clear caches or modify checkpoint files. Keep its output outside Git.

```sh
python3 experiments/2026-11-04-lossless-memory/collect_weights.py \
  SNAPSHOT/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --max-tensors 24 --tiles-per-tensor 2 --out INPUTS/weight-tiles.json

python3 experiments/2026-11-04-lossless-memory/screen.py \
  INPUTS/weight-tiles.json --repeats 5 --out runs/fp8-screen-boot-a.json
python3 experiments/2026-11-04-lossless-memory/screen.py \
  INPUTS/weight-tiles.json --repeats 5 --out runs/fp8-screen-repeat.json

python3 -m unittest tests.test_lossless_memory -v
```

The two CPU invocations are independent process observations, **not** candidate
boots. Output records all per-block timings, stored modes, frame overhead,
input hashes, checkpoint identity, library version, source commit and peak RSS.
Dense projection families and Engram are reported separately. Each encoder
invocation includes its round-trip verification; decoder timings include SHA-256.
The illustrative saved-read budget uses the older recorded 266.6 GB/s marginal
rate and is neither a current release estimate nor a performance verdict.

## Hardware sequence

1. Read `~/spark-hold.json` on dgx1. If another holder owns it, do no GPU work,
   benchmark, sync or serving operation. Request a window only when candidates
   are ready; do not displace an active job during source preparation.
2. Publish a clean commit. Use the coordinated cluster tools and inspect the
   plan's `live:` and `shape:` lines. Every apply names its cluster configuration.
   Serving stops, starts and restarts require the owner's explicit authorization.
3. Run one component's exactness tests and bounded kernel/I/O screen first.
   Record failures, memory and all unfavorable measurements. Reject a consumer
   that changes values or cannot fit decode overhead within its saved traffic.
4. For a viable candidate, build one immutable image and distribute its digest.
   Measure only the candidate on a boot and repeat boot, against the existing
   reference at `manifests/suites/2026-10-30-spark-ds41f-1030-<shape>`.
5. Run the release, single-stream, sampled, mixed-traffic, prefix and admission
   checks, plus cold boots where preparation changed. Check lowest memory and
   long-prefill stalls. Record confidence intervals and every required digest.
6. Restore the window's recorded production profile once at window close.
   A release PR may include only qualifying variables and needs owner acceptance
   before promotion. No new immutable baseline is written without real receipts.

## Status

CPU codec/storage tests pass. Hardware performance and serving integration are
not yet qualified. See `decision.md`, `base.json` and `runs/preparation.json`.
