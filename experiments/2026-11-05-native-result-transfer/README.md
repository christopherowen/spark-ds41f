# Native result transfer

Base: exact candidate-38 source, recorded in `base.json`. One intended variable:
transfer a strided one-column token-ID result to CPU without a device pack.
No promoted configuration, baseline or serving process changes.

Both original wire inventory boots expose an ATen packing kernel when native
int32/int64 sampled ID columns have row holes and zero alternatives are requested.
The GPU scoring trace alone misses this work. Transfer the physical column span
using PyTorch's existing asynchronous DMA, then retain the logical strided view
on CPU. Dense and multi-column IDs retain the existing transfer. Native dtype and
storage offset are respected; PyTorch retains ownership and stream/allocator
tracking for both device and pinned host memory. No device staging buffer or
raw pointer copy is added. CPU NumPy consumers already support these strides.

The tradeoff is extra transferred hole bytes, recorded per case by the complete
host-call/stream-wait benchmark. It must pass on both boots; a compute-only
measurement cannot qualify it. Twenty-six tests exercise two ID widths, shifted
storage offsets, empty/single/multiple rows, strides through 33, adaptive request
boundaries and a side-stream lifetime. Wire inventories check the actual public
score-to-CPU path in both score modes, layouts, widths and alternative counts.
Both boots pass all 26 transfer cases inside the 1,266-case combined suite.
All 12 changed-path complete transfer cases pass the strict five-standard-error,
zero-margin gate in both boots (mean about 11–16% faster). All 16 public-path
inventories contain no ATen packing kernel on either boot. Dense common-path
host overhead and full serving remain separate checks.

Raw evidence, exact commands, sources, memory observations and both initial and
fixed inventories are retained in `experiments/2026-11-05-tilelang-runtime-closure`
on branch `feat/tilelang-runtime-closure`. The two patched files match that
composed source byte-for-byte. A built-image/full-serving check remains separate;
no serving cluster operation is performed by these isolated sp8 experiments.

## Final combined check

Both final composed boots pass all 1,446 correctness checks. The 16 public
score-to-CPU routes on each wire boot contain no ATen packing kernel, and all
12 changed-path complete host-call/stream-wait cases pass both strict timing
gates. See `final-qualification.json`. Four 43-layer dummy request traces contain
only TileLang compute and preserve token/logprob bits. This does not qualify
dense-path host overhead, real-weight throughput or physical networking.
