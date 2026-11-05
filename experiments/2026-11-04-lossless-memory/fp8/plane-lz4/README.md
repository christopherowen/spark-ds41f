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
