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

## Results: the same outputs, the same speed at tp4-switched, 0.7% slower on the grid at tp4-ring

Jobs at `tp4-switched` 2026-11-13 08:52Z to 09:41Z and at `tp4-ring` 08:50Z to 09:51Z, deployment `85822940`
([receipts](../2026-10-26-spark-ds41f-candidate/candidate-55/receipts/kernel-cache)). All 12 boots reached ready and
no memory guard fired.

**1. Outputs are candidate-54's.** On kernels candidate-55 compiled itself, every gate run is identical on both
boots at both testbeds, with hashes 18d61961/d6568649 (text 0a2dacdd/12fd8257). #974's compiler changes no
temperature-0 output at TP4; until now that had been shown only for its runtime libraries.

| Testbed | Boots | Runs identical | c16 / c8 / c4 distinct streams, tok/s |
| --- | --- | --- | --- |
| `tp4-switched` | 2 | 8 of 8 | 269.93, 269.46 / 199.23, 199.08 / 154.87, 155.38 |
| `tp4-ring` | 2 | 8 of 8 | 271.31, 272.50 / 202.20, 203.28 / 156.56, 155.00 |

**2. Speed, the engine grid against `candidate-55` on the usual cache** (the mean of both boots of each; negative is
faster):

| Testbed | 1x2 | 1x3 | 1x3, long contexts | 6 to 16 rows | 24 rows | 32 rows and more | Grid (the tool) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tp4-switched` | -0.2% | +0.9% | +0.4% | +0.2% | -0.7% | -0.1% | -0.02%, margin 0.33%: the same |
| `tp4-ring` | +1.0% | +1.5% | +0.9% | +0.5% | -0.1% | +0.6% | **+0.68%, margin 0.34%: worse** |

- **At `tp4-ring` candidate-55's own kernels are slower** on the grid: 1.0% to 1.5% at 1x2 and 1x3 and 1.1% at 48
  rows. Against candidate-54's recorded screen, candidate-55 with the recipe was 1.55% faster on the old kernels, so
  it stays ahead on its own, by about 0.9% (cross-session).
- **The 24- and 32-row shapes do not move** with the compiler (-0.1% and +0.6% at `tp4-ring`).
- **With the default windows the same change was a gain** (5.9% and 10.8% at 1x2 and 1x3, the accidental arms at
  `tp4-switched`); with the caps that is gone. The compiler's effect depends on the prefetch windows, like every
  other change to the forward's timing.
- c16, c1 and prefill were not measured on the candidate's own kernels: the screens and c1 checks are the fifth
  build's to take.

**3. Boot.** The first boot on the empty cache at `tp4-ring` was ready in 1,086 s and wrote 4,149 cache files a
node; the boots after it took 114 to 121 s, as on the usual cache (116 s).

## The patch, end to end on one GPU

sp8, the candidate-55 image on a copy of the cache candidate-54's image filled (524 kernels), dummy weights:

| Boot | Kernel entries before | After | Wall |
| --- | --- | --- | --- |
| As built | 524 | 524 | 91 s: it loads candidate-54's kernels |
| With the patched `kernel_cache.py` mounted | 524 | 1,021 | 1,015 s: it compiles its own |
| The same again | 1,021 | 1,021 | 92 s: it reuses them |

([run6.sh](../2026-10-26-spark-ds41f-candidate/candidate-55/receipts/kernel-cache/sp8/run6.sh) and its log.)
