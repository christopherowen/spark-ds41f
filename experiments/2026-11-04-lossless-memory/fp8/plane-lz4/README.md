# Independent-plane LZ4 screen

Each complete 16 KiB FP8 tile is permuted into eight 2 KiB bitplanes. Each
plane uses an independent LZ4 block only when its aligned payload is smaller
than RAW. A whole tile also falls back to RAW when metadata removes the saving.
All 256 byte encodings remain integer bit patterns; no floating-point operation
is used. Headers, stream offsets, alignment and the nine-byte device descriptor
count toward storage. This is a probe ABI with source-byte verification before
upload, not an authenticated persistent format.

Two independent CPU preparation processes reconstruct every original byte of
the matched 42 dense tiles and produce identical payload hashes. Stored size
saves 6.2997%; all 42 tiles use exactly one compressed plane. The receipts pin
inputs, original-byte hashes, sources and the actual LZ4 library version/hash.
These are size observations, not GPU or serving performance.

A GPU probe must measure bounded independent-stream decode and the inverse
bit permutation against the same word-sized RAW consumer. Every reconstructed
GPU byte and consumer checksum must match before timing. A proxy win still
needs an unchanged-arithmetic GEMM consumer and two serving boots.

The 512-byte stream variant trades extra directory entries and shorter LZ4
history for more warp-level work: its two CPU preparations save 5.2325% and
produce identical payload hashes. The 2 KiB form keeps its original P4L1 ABI;
smaller independent chunks use P4L2. Both reconstruct every source byte.

The CUDA byte-consumer prototype and driver are prepared but have not been
compiled or run on the target. They are not serving integration or a speed claim.

The canonical target matrix pairs 512/2048-byte streams with 128/256 threads,
then repeats each point in a fresh container (eight jobs total). Both arms use
the same eight-byte shared consumer. Absolute times from the older word-sized
controls are not a comparison. Every original byte, encoding/boundary fixture
and source-derived consumer checksum must match before nine shuffled timed
observations. The receipt also pins the LZ4 library actually inside the immutable
image; local CPU preparation ratios do not predict target decoder throughput.

```sh
python3 experiments/2026-11-04-lossless-memory/run_fp8.py \
  --window WINDOW_ID --format plane-lz4 --lanes 128 256 \
  --chunk-bytes 512 2048 \
  --out results/private/lossless-fp8-plane-lz4-round1 --apply
```

Use only an authorized, published, idle FP8 measurement window. The controller
preserves foreign holds, validates every source/library/layout identity, and
restores the recorded Candidate 36 profile at its end. `analyze_gpu.py --format
plane-lz4` verifies original worker and controller copies before archiving them.
