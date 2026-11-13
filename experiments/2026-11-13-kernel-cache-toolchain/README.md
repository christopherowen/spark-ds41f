# The kernel cache and the toolchain: candidate-55 on kernels it compiled itself

Base commit: `543e52288`. Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`, CUDA 13.5.37 through
#974), with its TP4 recipe. One variable: `TILELANG_CACHE_DIR`.

## What was found

- **TileLang's kernel cache is not keyed by the compiler.** It has two levels. `KernelCache._generate_key`
  (`tilelang/cache/kernel_cache.py`) keys a finished kernel (`linux-aarch64/kernels/<key>/executable.so`) by its IR,
  arguments, target, backend, pass configuration, flags and the TileLang build. `CUDABinaryCache.make_key`
  (`tilelang/cache/cuda_binary_cache.py`) keys the device binary below it by `nvcc version=13.0.88` or `13.5.37`
  among other things, but is asked only when the kernel cache misses. So a kernel an earlier toolkit compiled is
  found at the first level and its binary is loaded as it is.
- **The nodes' caches were filled before candidate-55.** In the cache directory candidate-55 uses
  (`cache/spark-ds41f/jit/tilelang/0.1.15_snapshot_35f15411…`), 84 of 7,033 files on dgx1 and 56 of 8,699 on sp1 are
  newer than candidate-55's first build (2026-11-11 18:00Z); the oldest are of 2026-11-07. On sp1 that is 8 of 1,245
  kernel entries. The lab's boot notes agree:
  no candidate-55 image compiled a kernel on its first boot on a node.
- **So candidate-55 has run kernels CUDA 13.0.2 compiled**, wherever a kernel's source is candidate-54's. #974's `nvcc`
  and `ptxas` built only the kernels whose source changed. Every gate, check and screen of candidate-55 was taken that
  way, and a node with an empty cache would run something else.
- **Two boots on an empty cache exist, by accident:** the grid attribution's `pdlpostwait` and `no982` arms mounted
  nothing and set `TILELANG_CACHE_DIR` to a new directory. With the default prefetch windows they were 5.9% and 10.8%
  faster at 1x2 and 1x3 than candidate-55 on the usual cache, and the same elsewhere; with the 15/12/12 MB windows
  the same arm was the same as the caps alone
  ([grid-attribution](../2026-10-26-spark-ds41f-candidate/candidate-55/grid-attribution/README.md)).

## What is measured

candidate-55 on a cache its own image compiled, against candidate-55 on the usual cache, at both TP4 testbeds:

| Testbed | `TILELANG_CACHE_DIR` | Where the cache comes from |
| --- | --- | --- |
| `tp4-switched` | `/cache/tilelang/c55-no982` | compiled from empty by the third build's image on sp1..sp4, 2026-11-12 15:42Z to 16:02Z (4,149 files a node); despite its name no overlay was mounted. The fourth build compiles the kernels #738 changed on its first boot |
| `tp4-ring` | `/cache/tilelang/candidate-55-cuda135` | new: the first boot compiles every kernel (about 24 minutes at `tp4-switched`) |

- **Gate first** ([gate-tp4.json](gate-tp4.json), [gate-tp4-switched.json](gate-tp4-switched.json)): two boots, the
  workloads of candidate-55's gate.
- **Engine grid** ([grid-tp4.json](grid-tp4.json), [grid-tp4-switched.json](grid-tp4-switched.json)): two boots each
  of `candidate-55-own-kernels` and `candidate-55`.

## What is asked, declared before the runs

This measures what candidate-55 is, so there is no mark to pass, only questions:

1. **Outputs.** Are the two boots identical to each other, and do they give candidate-54's TP4 hashes
   18d61961/d6568649? #974 was expected to change temperature-0 output; the gates so far found it did not, on
   kernels its compiler did not build.
2. **Speed.** The grid against `candidate-55` on the usual cache, by shape group, and whether the 24- and 32-row
   shapes move.
3. **Boot.** Time to ready once the cache is filled.

Whichever way these come out, the cache has to be made the build's own before candidate-55 is frozen.

## The fix: the device compiler in the kernel cache key

Owner, 2026-11-13: "make the compiler version part of the cache key".
[tilelang/0014-kernel-cache-device-compiler.patch](tilelang/0014-kernel-cache-device-compiler.patch), on the TileLang
of `spark-ds41f-1107` and candidate-54 and 55 (head `92a1d3a0`, tree `e2b00182`; with it, tree `741806c7`):

- `KernelCache._generate_key` adds `device_compiler` for CUDA targets: the string the CUDA binary cache already
  uses, `nvcc version=13.5.37` (or `nvrtc version=…` when `tl.cuda_compiler` selects it). Other targets keep their
  keys. A compiler whose version cannot be read is stamped `unknown`, which matches no other entry.
- **Every CUDA kernel key changes once.** The first boot of an image that carries the patch compiles its whole
  kernel set on each node (about 24 minutes at TP4), as a node with an empty cache does. Entries under the old keys
  are never read again; `bin/spark cache clear` removes them.
- **Tests:** its five unit tests pass in the candidate-55 image on sp8 (the stamp, other targets unchanged, the key
  following the compiler's version and choice, an unreadable compiler).
- **Not in an image yet.** Carrying it changes TileLang's tree, so it enters candidate-55 with a fifth build, whose
  profiles then no longer match the fourth build's image. The runs above measure the fourth build on a cache it
  compiled itself, which is what the patch makes every node do.

## Commands

```bash
bin/spark candidate queue tp4-switched experiments/2026-11-13-kernel-cache-toolchain/gate-tp4-switched.json experiments/2026-11-13-kernel-cache-toolchain/grid-tp4-switched.json --apply
bin/spark candidate queue tp4-ring experiments/2026-11-13-kernel-cache-toolchain/gate-tp4.json experiments/2026-11-13-kernel-cache-toolchain/grid-tp4.json --apply
bin/spark candidate gate tp4-switched kernel-cache-toolchain-gate-tp4-switched --expect 18d61961/d6568649
```

## Results

Not run yet.
