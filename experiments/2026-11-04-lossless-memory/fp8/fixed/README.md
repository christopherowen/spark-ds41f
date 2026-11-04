# Fixed-field decoder screen

This arm tests an interaction: exact FP8 representation plus a word-sized
shared-memory byte consumer. It is paired with the same word-sized raw consumer.
Absolute times from the earlier byte-at-a-time Huffman control are not a
comparison for this arm. No GEMM, accumulation order or serving kernel changes.

Each 16,384-byte tile splits into four unchanged sign/mantissa bits and a
two-bit tag. Three tags name the most frequent original exponents; the fourth
names an unchanged four-bit exponent in an exception stream. An offset every
128 values supports bounded warp prefix counts. Headers, offsets, alignment,
exceptions and the actual offset/mode device descriptors count toward storage.
Unprofitable tiles use RAW without unused metadata. Every byte, including signed
zeros and special encodings, is represented exactly; there is no float conversion.
The frame is a probe ABI, not a persisted or untrusted-input storage format.

The CPU reference checks structural extents, padding, dictionary uniqueness and
every chunk offset. Before any GPU timing, all decoded checkpoint bytes must
equal the pinned source. Separate compressed fixtures exercise every byte
encoding, zero exceptions, and scattered exceptions at warp/page boundaries.
Both timed arms use four-byte global/shared loads/stores and the same shared
word consumer. First/repeat containers run at 128, 256 and 512 threads with a
128 MiB working set and nine shuffled raw/coded observations each. All original
timings and errors must be retained. A proxy win would still require a fused
GEMM with unchanged arithmetic, followed by full serving qualification.

Local tests pass. Exact deterministic encoding of the matched 42 dense tiles
saves 8.6876%, including every device descriptor. This is a size screen only;
the decoder has not been compiled or run on GB10. Its CPU packing cost and
storage ratio do not predict GPU throughput.

Within the standard authorized, published, idle cluster window bearing the
FP8 measurement holder, after inspecting the coordinated stop plan:

```sh
python3 experiments/2026-11-04-lossless-memory/run_fp8.py \
  --window WINDOW_ID --format fixed-fields --lanes 128 256 512 \
  --out results/private/lossless-fp8-fixed-round-1 --apply
```

The shared controller requires this format's own source hashes and GPU
exactness flag. It closes only its exact owned window and restores Candidate 36
at completion or failure. Foreign holds, older requests and time caps retain
their normal guards. Release promotion remains a separate owner decision.
